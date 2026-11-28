# ASL randomized group project

This is the revised project for four members: one preprocessing contribution and three model contributions. It trains Random Forest, KNN and Decision Tree using a fixed randomized group holdout. It includes the dataset, prepared split, trained pipelines, measured results, notebooks, checks and explanations.

Start with the measured results in `results/REPORT.md`. Read `docs/METHOD_AND_LIMITATIONS.md` before describing the scores to your lecturer. This project has no chronological training protocol and no SVM model.

## What changed

- Construct connected groups from exact coordinate matches and known same-class filename timestamps at most 1,000 ms apart.
- Build groups on all original rows before removing repeated vectors, so removed observations cannot break temporal connections.
- Remove repeated single-label vectors once, using a deterministic representative. Quarantine all rows of a conflicting-label exact vector if one is found.
- Assign whole groups to partitions using fixed shuffled stratified group folds. There is no earlier/later assignment rule.
- Use shared partitions and training-only scaling for all three algorithms.
- Run an independent overlap audit before each model fits.
- Select a model using validation accuracy, then report all three models' test scores.

Known overlap checks pass. Unknown timestamps and absent signer/session IDs remain limitations. The project does not claim all possible leakage is eliminated.

## 1 Install

Extract the entire ZIP. Open a terminal in `ASL_Random_Group_Project`.
Use Python 3.12; the completed run used Python 3.12.14.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS use `python3.12 -m venv .venv`, then `.venv/bin/python` for subsequent commands.
Each member creates their own environment; do not copy another computer's `.venv` directory.
Optional notebooks:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
```

Select `.venv` as the notebook kernel. Notebook cells are supplied unexecuted. The included results came from running the Python scripts.

## 2 Assign the contributions

| Member | Contribution | Python file | Notebook |
|---|---|---|---|
| 1 | Data preparation and audit | `preprocessing.py` | `00_preprocessing.ipynb` |
| 2 | Random Forest | `random_forest.py` | `01_random_forest.ipynb` |
| 3 | KNN | `knn.py` | `02_knn.ipynb` |
| 4 | Decision Tree | `decision_tree.py` | `03_decision_tree.ipynb` |

Every member needs the complete project folder. The model files call shared helpers; they are separate contributions within this project, not standalone files.

## 3 Prepare once

Member 1 runs:

```powershell
.\.venv\Scripts\python.exe preprocessing.py
.\.venv\Scripts\python.exe audit.py
```

Check `prepared/metadata.json`, `class_counts.csv`, `split_manifest.csv` and `overlap_audit.json`.
Share the source, `data/` and the same `prepared/` folder with the model members.
The package already contains this prepared split; running preprocessing reproduces it with the pinned versions.

The original raw dataset ZIP can be used without extracting its thousands of CSV files:

```powershell
.\.venv\Scripts\python.exe preprocessing.py --data "C:\path\media_pipe_keypoints_dataset.zip"
```

This caches the loaded arrays as `data/landmarks.npz`. Preprocessing changes require rerunning all three models.

## 4 Train your assigned model

Each member runs one command:

```powershell
.\.venv\Scripts\python.exe random_forest.py
.\.venv\Scripts\python.exe knn.py
.\.venv\Scripts\python.exe decision_tree.py
```

Editing is optional. Each script already defines a working estimator. It loads the common partitions, fits its scaler and classifier on training samples, evaluates all three partitions and saves results.

Outputs are separate: `results/random_forest/`, `results/knn/`, `results/decision_tree/`.
A rerun overwrites that model's saved baseline outputs. Use `--output experiments/my_run` for exploratory outputs if you want to preserve the packaged baseline.

## 5 Combine and compare

Copy each member's model result folder into the same `results/` folder, then run:

```powershell
.\.venv\Scripts\python.exe compare_models.py
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe write_report.py
```

The comparison rejects results with mismatched data/split fingerprints. It selects by validation accuracy and writes `comparison.csv`, `comparison.json` and `selection.json`. It does not retrain the winner.

In the completed run KNN wins on validation; Random Forest is highest on test. Keep the validation-selected KNN rather than switching after seeing test results.

## 6 One command option

To reproduce preparation, all three models, comparison, control checks, verification and report:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

For a four-member contribution, use the separate commands above. Both paths call the same source.

## 7 Predict one CSV

```powershell
.\.venv\Scripts\python.exe predict.py data/example_landmarks.csv
```

The example is one held-out dataset sample, included only to demonstrate the input format. Its actual label is A; the packaged selected KNN predicts N. This demonstrates a real error, even with a high neighbour vote score, rather than claiming the sample proves live accuracy. It is not a new external evaluation. For your own sample, provide 21 rows with `Landmark,X,Y,Z` columns in the same coordinate convention. The script loads the selected fitted pipeline. Do not scale its input manually.

No webcam capture interface is included. This project evaluates extracted landmark vectors, not motion sequences or sentence translation.

## 8 Check the work

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe sanity_checks.py
```

Optional full comparison with original CSVs:

```powershell
.\.venv\Scripts\python.exe verify_project.py --raw "C:\path\media_pipe_keypoints_dataset.zip"
```

The completed delivery includes `results/verification.json`, `sanity_checks.json` and `unit_tests.txt`.
These checks do not require running a chronological experiment or any fourth classifier.

## Files to read next

- `docs/METHOD_AND_LIMITATIONS.md`: exact split method, audit scope and interpretation.
- `docs/STUDY_GUIDE.md`: preprocessing, model explanations, metrics and viva questions.
- `docs/GROUP_WORK_PLAN.md`: responsibilities and next steps.
- `results/REPORT.md`: only the new measured randomized-group results.

## Original sources

Original team repository: https://github.com/Amindra-Edirisuriya/American-Sign-Language-Ditection
Data: supplied MediaPipe CSV ZIP, numerically checked against the compact input.
Dataset publisher/license information must be verified separately before public redistribution.
The referenced proposal PDF was not provided, so proposal-specific alignment is not claimed.
