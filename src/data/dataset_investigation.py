
from pathlib import Path
import re

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "media_pipe_keypoints_dataset"

print("=" * 60)
print("ASL DATASET INVESTIGATION")
print("=" * 60)

# Check dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# Get class folders
class_folders = [
    folder for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
]

print(f"\nNumber of classes: {len(class_folders)}")

# Inspect filenames
all_files = []

for folder in sorted(class_folders):
    csv_files = list(folder.glob("*.csv"))
    all_files.extend(csv_files)

    print(f"\nClass: {folder.name}")
    print(f"Samples: {len(csv_files)}")

    print("Example filenames:")
    for file in csv_files[:5]:
        print("  ", file.name)

# Total samples
print("\nTotal CSV samples:", len(all_files))

# Investigate numeric filename patterns
timestamp_count = sum(
    bool(re.search(r"\d{10,}", file.stem))
    for file in all_files
)

print(
    "Filenames containing long numeric sequences:",
    timestamp_count
)

from collections import Counter
from datetime import datetime, timezone

print("\n" + "=" * 60)
print("FILENAME PATTERN ANALYSIS")
print("=" * 60)

# Extract numeric suffixes
numeric_suffixes = []

for file in all_files:
    match = re.search(r"_(\d{1,})$", file.stem)

    if match:
        numeric_suffixes.append(int(match.group(1)))

print("\nNumeric suffixes found:", len(numeric_suffixes))

# Examine large numeric suffixes
large_numbers = [
    number for number in numeric_suffixes
    if number >= 1_000_000_000_000
]

print("13+ digit numeric suffixes:", len(large_numbers))

# Convert timestamp-like values to dates
print("\nExample timestamp conversions:")

for number in large_numbers[:10]:
    try:
        dt = datetime.fromtimestamp(
            number / 1000,
            tz=timezone.utc
        )
        print(number, "->", dt)
    except (ValueError, OSError, OverflowError):
        print(number, "-> Invalid timestamp")

# Check filename prefixes against folder labels
print("\nFilename prefix mismatches:")

mismatches = []

for file in all_files:
    prefix = file.stem.split("_")[0]
    folder_label = file.parent.name

    if prefix != folder_label:
        mismatches.append((folder_label, file.name))

print("Total mismatches:", len(mismatches))

for folder, filename in mismatches[:20]:
    print(f"Folder: {folder} | Filename: {filename}")

from collections import defaultdict
import statistics

print("\n" + "=" * 60)
print("TIMESTAMP SEQUENCE INVESTIGATION")
print("=" * 60)

timestamp_groups = defaultdict(list)

for file in all_files:
    match = re.search(r"_(\d{13})$", file.stem)

    if match:
        timestamp = int(match.group(1))

        timestamp_groups[file.parent.name].append({
            "timestamp": timestamp,
            "filename": file.name
        })

for label, samples in sorted(timestamp_groups.items()):

    timestamps = sorted(
        sample["timestamp"] for sample in samples
    )

    if len(timestamps) < 2:
        continue

    intervals = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    print(f"\nClass: {label}")
    print("Timestamp samples:", len(timestamps))
    print("First timestamp:", timestamps[0])
    print("Last timestamp:", timestamps[-1])
    print("Median interval (ms):", statistics.median(intervals))
    print("Minimum interval (ms):", min(intervals))
    print("Maximum interval (ms):", max(intervals))

print("\n" + "=" * 60)
print("GLOBAL TIMESTAMP GAP ANALYSIS")
print("=" * 60)

all_timestamped_samples = []

for file in all_files:
    match = re.search(r"_(\d{13})$", file.stem)

    if match:
        timestamp = int(match.group(1))

        all_timestamped_samples.append({
            "timestamp": timestamp,
            "label": file.parent.name,
            "filename": file.name
        })

all_timestamped_samples.sort(key=lambda x: x["timestamp"])

print("Total timestamped samples:", len(all_timestamped_samples))

# Calculate gaps between consecutive timestamps
gaps = []

for i in range(len(all_timestamped_samples) - 1):
    current = all_timestamped_samples[i]
    following = all_timestamped_samples[i + 1]

    gap = following["timestamp"] - current["timestamp"]

    gaps.append({
        "gap": gap,
        "previous": current,
        "following": following
    })

print("\nLargest 20 timestamp gaps:")

largest_gaps = sorted(
    gaps,
    key=lambda x: x["gap"],
    reverse=True
)[:20]

for item in largest_gaps:
    print("\nGap:", round(item["gap"] / 1000, 2), "seconds")
    print("Previous:", item["previous"]["label"],
          item["previous"]["filename"])
    print("Following:", item["following"]["label"],
          item["following"]["filename"])

# Count gaps above different thresholds
print("\nGap threshold analysis:")

for threshold in [1, 5, 10, 30, 60, 300]:
    count = sum(
        1 for item in gaps
        if item["gap"] > threshold * 1000
    )

    print(f"Gaps greater than {threshold} seconds: {count}")

print("\n" + "=" * 60)
print("PRELIMINARY RECORDING BATCH ANALYSIS")
print("=" * 60)

# Sort all timestamped samples chronologically
samples = sorted(
    all_timestamped_samples,
    key=lambda x: x["timestamp"]
)

# Start a new batch when the gap exceeds 10 seconds
batches = []
current_batch = [samples[0]]

for i in range(1, len(samples)):

    gap = (
        samples[i]["timestamp"]
        - samples[i - 1]["timestamp"]
    )

    if gap > 10000:
        batches.append(current_batch)
        current_batch = [samples[i]]
    else:
        current_batch.append(samples[i])

batches.append(current_batch)

print("Preliminary batches:", len(batches))

for batch_number, batch in enumerate(batches, start=1):

    labels = sorted(set(
        sample["label"] for sample in batch
    ))

    duration = (
        batch[-1]["timestamp"]
        - batch[0]["timestamp"]
    ) / 1000

    print(f"\nBatch {batch_number}")
    print("Samples:", len(batch))
    print("Duration (seconds):", round(duration, 2))
    print("Classes:", labels)
    print("First sample:", batch[0]["filename"])
    print("Last sample:", batch[-1]["filename"])
