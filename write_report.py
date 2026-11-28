"""Rebuild the measured report from existing result files, without retraining."""
import json
from common import ROOT,MODEL_NAMES,load_prepared


def write_report():
    *_,meta=load_prepared()
    comparison=json.loads((ROOT/'results/comparison.json').read_text())
    audit=json.loads((ROOT/'prepared/overlap_audit.json').read_text())
    if comparison['fingerprint']!=meta['fingerprint'] or audit['fingerprint']!=meta['fingerprint']:
        raise ValueError('Report inputs belong to different splits')
    stats=meta['stats'];counts=meta['counts']
    lines=['# ASL randomized group evaluation results','',
        'These are measured results from the fixed randomized group holdout described in docs/METHOD_AND_LIMITATIONS.md. No earlier/later assignment rule is used. Fixed model settings and split seeds were set before fitting; no hyperparameter or seed search was performed.','',
        f"All {len(meta['classes'])} classes occur in all three partitions. Original samples: {stats['original_samples']:,}. Repeated rows removed: {stats['exact_duplicate_rows_removed']:,}. Conflicting-label rows excluded: {stats['conflicting_label_rows_removed']:,}. Retained unique vectors: {stats['retained_samples']:,}.",'',
        '| Partition | Samples | Retained percentage |','|---|---:|---:|']
    for name,n in counts.items():lines.append(f"| {name} | {n:,} | {meta['actual_percentages'][name]:.2f}% |")
    lines+=['','## Model performance','',
        '| Model | Training accuracy | Validation accuracy | Test accuracy | Gap in pp | Macro F1 | Test errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison['rows']:
        lines.append(f"| {r['model']} | {r['train_accuracy_pct']:.4f}% | {r['validation_accuracy_pct']:.4f}% | {r['test_accuracy_pct']:.4f}% | {r['train_test_gap_pp']:.4f} | {r['macro_f1']:.4f} | {r['test_errors']:,} |")
    selected=comparison['selected_by_validation'];highest=max(comparison['rows'],key=lambda r:r['test_accuracy_pct'])['model']
    lines+=['',f'Validation-selected model: **{selected}**. Highest test accuracy: **{highest}**. The test ranking does not change the validation selection. Saved pipelines retain their original training fits.','',
        'The large training/test gaps show weak generalization to some held-out groups. This is an observed result, not an error corrected by lowering reported scores. The single holdout and large proxy groups also affect variability. Inspect class_counts.csv and per-class errors.','',
        '## Overlap checks','',f"- Exact vectors shared between retained partitions: {audit['exact_vectors_shared_between_partitions']}.",
        f"- Original same-class timestamp edges checked: {audit['original_timestamp_edges_checked']:,}.",
        f"- Those timestamp edges crossing partitions: {audit['original_timestamp_edges_crossing_partitions']}.",
        '- Retained known same-class <=1-second neighbour matches: '+str(audit['retained_timestamp_neighbour_counts'])+'.','',
        'These checks pass for the specific equality/time rules. They do not establish signer/session independence, rule out nonidentical near-duplicate vectors, or establish temporal independence for unknown timestamps.','',
        '## Timestamp availability','',
        '| Model | Known timestamp test accuracy | Unknown timestamp test accuracy |','|---|---:|---:|']
    for name in MODEL_NAMES:
        m=json.loads((ROOT/'results'/name/'metrics.json').read_text())
        if m['fingerprint']!=meta['fingerprint']:raise ValueError('Mismatched model metrics')
        gs=m['timestamp_status_scores']
        cells=[f"{r['accuracy_pct']:.4f}% on {r['samples']:,} rows" if r['samples'] else 'Not applicable' for r in gs]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['',f"Original rows without timestamps: {audit['missing_timestamp_samples_original']:,}; retained rows without timestamps: {audit['missing_timestamp_samples_retained']:,}. Both subsets may differ in class composition and capture conditions. The subgroup comparison is descriptive and does not isolate the cause of the score difference.",'',
        '## Class and group weighting','',
        'The headline accuracy weights each retained test observation equally. Macro F1 weights classes equally. Mean proxy-group accuracy is also saved, but most groups are small or singleton and it can be much higher than sample accuracy. It must not replace the headline score. Correlated rows within large groups mean the test row count is not a count of independent recording sessions.','',
        '## Sanity controls','']
    path=ROOT/'results/sanity_checks.json'
    if path.exists():
        s=json.loads(path.read_text())
        if s['fingerprint']!=meta['fingerprint']:raise ValueError('Mismatched controls')
        lines += [f"Majority-class baseline test accuracy: {s['majority_baseline_test_accuracy_pct']:.4f}%.",
            f"One tree fitted on shuffled training labels: {s['shuffled_training_labels_tree_test_accuracy_pct']:.4f}% test accuracy against true labels.",
            '',s['explanation']]
    else:lines+=['Run sanity_checks.py to generate the majority baseline and single shuffled-label control.']
    lines+=['','## Reproduction and interpretation','',
        'Use README.md for the four-member workflow. Python scripts produced these measurements; the notebook cells are supplied unexecuted entry points. Software versions, estimator settings, timing, predictions, class reports and confusion matrices are saved per model. Verification checks data/split integrity, training-only scaler fitting and saved-pipeline consistency.','',
        'The scoring population differs from a raw random row split because duplicates are removed and groups are kept intact. Differences from historical scores cannot be interpreted as a causal amount of leakage. No accuracy range was targeted.','',
        'For a stronger final claim, collect fresh data with explicit signer and recording/session IDs and keep those identities separate from training. Further model tuning should use training/validation data with grouping preserved; these test results have now been seen.','',
        'Data/split SHA-256 fingerprint: `'+meta['fingerprint']+'`.']
    path=ROOT/'results/REPORT.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report saved:',path);return path

if __name__=='__main__':write_report()
