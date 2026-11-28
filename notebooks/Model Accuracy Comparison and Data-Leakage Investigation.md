American Sign Language (ASL) Detection Project
Model Accuracy Comparison and Data-Leakage Investigation
Project repository:
https://github.com/Amindra-Edirisuriya/American-Sign-Language-Ditection\
Dataset:
https://www.kaggle.com/datasets/ironmanomk/sign-language-mediapipe-keypoints\
Notebook: notebooks/05_model_accuracy_comparison.ipynb
1. Project and dataset overview
The project uses MediaPipe hand-landmark coordinates to classify
American Sign Language (ASL) signs.
- Total samples: 64,376
- Number of classes found in the downloaded dataset: 28
- Features per sample: 63
- Feature representation: 21 hand landmarks × 3 coordinates (X, Y,
  Z)
The dataset documentation describes 29 classes (A--Z and additional
labels), but the downloaded dataset contains 28 class folders. The
downloaded labels include Confirm rather than stop. This discrepancy
should be mentioned accurately in the project report rather than
silently assuming the documentation and downloaded data match.

The dataset contains individual CSV files with MediaPipe coordinates.
The available documentation does not define a predefined train/test
split and does not provide clear signer IDs, recording/session IDs, or
raw-video boundaries.
2. Project paths and processed data
Local project directory:
C:\Users\ASUS\Desktop\AMINDRA\American Sign Language
Raw dataset directory:
PROJECT_ROOT / "media_pipe_keypoints_dataset"
Processed data directory:
PROJECT_ROOT / "dataset" / "processed"
Processed files identified during the work:
X_train_original.npy
X_test_original.npy
X_train_scaled.npy
X_test_scaled.npy
y_train.npy
y_test.npy
standard_scaler.pkl
Important: the raw dataset was not located in dataset/raw; that
directory contained no CSV files. The correct source directory used for
reconstruction was media_pipe_keypoints_dataset.
3. Preprocessing and original train/test split
The dataset was loaded by iterating through the class folders and
reading each class folder's CSV files. The order of this iteration
matters because it determines the relationship between each sample and
its sample_index.

The original split was a stratified 80/20 train/test split using:
train_test_split(
    all_indices,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)
The reconstructed split was checked against the saved processed arrays.
All four checks returned True:
- Reconstructed training features matched X_train_original.
- Reconstructed testing features matched X_test_original.
- Reconstructed training labels matched y_train.
- Reconstructed testing labels matched y_test.
This verified that the reconstructed indices corresponded to the saved
original split.
4. Model comparison results
The recorded model results were:

  Model                            Accuracy
  Random Forest                    99.8680%
K-Nearest Neighbours (KNN)       99.8524%
Support Vector Machine (SVM)     99.8447%
Decision Tree                    99.0758%

Recorded training and prediction times:

  Model             Training time (s)   Prediction time (s)
  KNN                        0.022517              1.296660
SVM                       13.081911             17.632233
Decision Tree              8.270700              0.006001
Random Forest              8.548501              0.135869

The Random Forest model achieved the highest recorded accuracy among
these models. Its recorded training accuracy was 100%.

These are the results from the original random stratified split. They
should not automatically be described as performance on a fully
independent recording-level test set.
5. Why data leakage was investigated
The dataset consists of many individual samples, and some samples may be
identical or very similar to samples in the training set.

If highly related samples are divided between training and testing, the
test set may not represent genuinely independent data. This can make a
model's measured test performance appear more reliable than it would be
on new recordings.

The investigation considered:
1. Exact duplicate feature vectors across train and test.
2. Similar samples recorded close together in time.
3. Whether temporally adjacent samples were split between training and
   testing.
4. How model performance changed when potentially affected test samples
   were excluded.
The available metadata does not identify recording sessions or signers.
Therefore, temporal proximity is a warning signal to investigate, not
proof that two samples came from the same recording session.
6. Exact duplicate investigation
Exact duplicate detection was performed using the original, unscaled
feature vectors.

Results:
- Exact duplicate test samples also found in training: 192
- Test-set size: 12,876
- Percentage of test samples: 1.4911%
- Duplicate samples with matching labels: 192
- Duplicate samples with conflicting labels: 0
After excluding the 192 exact-duplicate test samples and evaluating the
existing Random Forest predictions:

  Evaluation                              Accuracy
  Original test set                       99.8680%
Test set excluding exact duplicates     99.8660%

The measured difference was approximately 0.002 percentage points.

Interpretation: Exact duplicates were present across the train/test
split, but removing them had a very small effect on the recorded Random
Forest accuracy. This result only measures the effect of exact
duplicates; it does not rule out other forms of similarity or leakage.
7. Timestamp investigation
The filename timestamp extraction used this pattern:
r"_(\d{13})$"
The timestamp analysis found:
- Timestamped samples: 29,282
- Samples without timestamps: 35,094
- Timestamped samples span two UTC dates:
  - 2025-02-21
  - 2025-03-24
The timestamps allow some temporal-neighbour analysis, but they do not
establish recording/session boundaries.
8. Temporal-neighbour similarity
Timestamped samples were sorted by label and timestamp. Distances
between consecutive samples within each class were calculated.

Results:
- Total consecutive timestamped pairs: 29,258
- Pairs with a time gap greater than 0 and no more than 1 second:
  29,224
- Median distance between consecutive samples: 0.022439
- Median distance between randomly selected same-class samples:
  0.799432
The median distance for consecutive samples is much smaller than the
median distance for randomly selected same-class samples.

Interpretation: Samples captured close together in time tend to have
much more similar feature vectors than randomly selected samples of the
same class. This is evidence of strong temporal similarity in the
dataset. It does not, by itself, prove that the samples belong to the
same recording session or that the model has memorized them.
9. Temporal neighbours across the train/test split
For consecutive timestamped pairs within one second, the original
train/test assignment was checked.

Results:
- Cross-split consecutive pairs: 9,388
- Percentage of close consecutive pairs: 32.12%
- Unique test samples with at least one adjacent timestamped training
  neighbour: 5,629
- Percentage of all test samples: 43.72%
For an independently assigned random 80/20 split, the probability that a
pair is divided across train and test is:
P(cross-split) = P(train, test) + P(test, train)
               = (0.8 × 0.2) + (0.2 × 0.8)
               = 0.32
               = 32%
The observed cross-split proportion of 32.12% is very close to the 32%
expected under a random 80/20 split.

Interpretation: The percentage of cross-split adjacent pairs is not
unusually high compared with what the random split itself would produce.
However, 5,629 test samples had at least one temporally adjacent
training neighbour. This indicates that a substantial part of the test
set has nearby timestamped samples in training.

This is temporal adjacency evidence, not confirmed recording-level
leakage. The dataset does not provide the session metadata needed to
establish that these samples came from the same recording.
10. Temporal-removal sensitivity experiment
A sensitivity experiment excluded test samples that had at least one
consecutive timestamped training neighbour within one second.

Results:
- Original test samples: 12,876
- Excluded potentially affected test samples: 5,629
- Remaining test samples: 7,247
- Original Random Forest accuracy: 99.8680%
- Accuracy on the remaining subset: 99.9448%
The accuracy increased after the potentially affected samples were
removed.

However, the remaining subset had a substantially different class
distribution. For example, the number of Confirm samples changed from
557 to 22. Some classes, including Q, Space, X, and Y, were unchanged
because they had no timestamps.

Important limitation: The 99.9448% result is a selected-subset
sensitivity result. It is not a replacement for the original test
accuracy and must not be presented as a definitive leakage-free
accuracy. The class distribution changed substantially, so the two
accuracy values are not a fair like-for-like comparison.

The result does not show that temporal similarity inflated the measured
accuracy. It also does not prove that temporal similarity had no effect.
11. Why a date-based split is not sufficient
Only four classes with timestamps appeared on both recorded dates:
- Confirm
- E
- I
- R
Other timestamped classes appeared on only one of the two dates. Four
additional classes---Q, Space, X, and Y---had no timestamps.

Therefore, a date-based split using the available timestamps cannot
provide a complete evaluation across all 28 classes.

A date-based experiment could potentially be explored for the four
classes that appear on both dates, but it would not be a valid full
28-class evaluation.
12. Overall findings and limitations
Findings supported by the investigation
1. The original stratified random split produced very high model
   accuracy.
2. There were 192 exact duplicate test samples also present in
   training.
3. All detected exact duplicates had matching labels.
4. Removing exact duplicates changed Random Forest accuracy only
   slightly.
5. Consecutive timestamped samples were much more similar than random
   same-class samples.
6. A substantial number of test samples had a temporally adjacent
   training sample within one second.
7. The observed proportion of cross-split consecutive pairs was close
   to the 32% expected from a random 80/20 split.
8. Removing temporally affected test samples produced a higher
   accuracy, but also changed the class distribution substantially.
What cannot currently be concluded
- It cannot be confirmed that temporally adjacent samples came from
  the same recording session.
- It cannot be confirmed that the dataset contains signer overlap,
  because signer IDs are not provided.
- It cannot be claimed that the original 99.8680% accuracy is
  definitely inflated by recording-level leakage.
- It cannot be claimed that the 99.9448% selected-subset accuracy is
  the true independent-test accuracy.
- A complete recording-level or signer-independent evaluation cannot
  be constructed from the currently identified metadata alone.
13. Recommended next steps
1. Re-run the notebook from the beginning so all variables are
   recreated.
2. Complete the planned comparison between test samples with and
   without temporal neighbours, using the same model predictions and
   clearly reporting each subset's class distribution.
3. Keep the original random-split results as the baseline.
4. Report exact-duplicate and temporal-neighbour investigations
   separately.
5. Clearly state that recording/session IDs and signer IDs are
   unavailable.
6. If a future dataset version provides signer IDs or recording/session
   IDs, use a group-based split so that complete signers or recording
   sessions are kept separate between training and testing.
7. Avoid describing the current temporal-removal subset as a fully
   leakage-free test set.
14. Notebook recovery note
A later diagnostic showed that the current Jupyter kernel did not
contain these variables:
train_indices: MISSING
test_indices: MISSING
metadata_df: MISSING
consecutive_distances_df: MISSING
model_instance: MISSING
X_test_scaled: MISSING
y_test: MISSING
This means the kernel's in-memory state had been cleared or the setup
cells had not been run in the current session. It does not mean the
saved .npy processed files have been deleted.

To restore the notebook state, use:

Run → Restart Kernel and Run All Cells

Then check for execution errors. Once the notebook has recreated the
data, split indices, trained model, and predictions, continue with the
remaining comparison. Do not assume that variables from an earlier
kernel session still exist.
15. Suggested wording for the project report
The initial model evaluation used a stratified random 80/20 train/test
split and achieved a Random Forest accuracy of 99.8680%. To
investigate the reliability of this result, exact duplicate and
temporal-neighbour analyses were conducted. A total of 192 exact
duplicate test samples were identified in the training set, all with
matching labels. Excluding these samples produced only a very small
change in accuracy.

The timestamp analysis also showed that consecutive samples captured
close in time were substantially more similar than randomly selected
samples from the same class. A number of test samples had temporally
adjacent training samples. However, the observed proportion of
cross-split consecutive pairs was close to the proportion expected
under a random 80/20 split. Since the dataset does not provide signer
IDs, recording/session IDs, or raw-video boundaries, recording-level
leakage could not be confirmed. A temporal-removal sensitivity
experiment was also performed, but the resulting subset had a
different class distribution and therefore cannot be treated as a
definitive independent-test evaluation.

Consequently, the reported random-split accuracy should be interpreted
in the context of these dataset limitations. A future evaluation using
signer-independent or recording-session-independent splitting would
provide stronger evidence of generalization to unseen signers or
recordings.