"""Read MediaPipe sample CSVs from a directory or directly from a ZIP."""
import csv
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

STAMP = re.compile(r"_(\d{13})$")


def read_sample(stream, identifier):
    with io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text:
        reader = csv.DictReader(text)
        if not {"Landmark", "X", "Y", "Z"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{identifier}: expected Landmark, X, Y and Z headers")
        try:
            rows = sorted((int(r["Landmark"]), *(float(r[k]) for k in "XYZ")) for r in reader)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{identifier}: invalid landmark or coordinate") from exc
    if [r[0] for r in rows] != list(range(21)):
        raise ValueError(f"{identifier}: expected one row each for landmarks 0 through 20")
    vector = np.asarray([r[1:] for r in rows], dtype=np.float64).ravel()
    if not np.isfinite(vector).all():
        raise ValueError(f"{identifier}: non-finite coordinate")
    return vector


def load_dataset(source):
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            features, labels, ids, times = (data[k] for k in ("X", "y", "ids", "t"))
        count = len(ids)
        if (not count or features.shape != (count, 63) or labels.shape != (count,)
                or times.shape != (count,) or ids.shape != (count,)
                or ids.dtype.kind != "U" or labels.dtype.kind != "U"
                or times.dtype.kind != "i" or not np.isfinite(features).all()
                or len(np.unique(ids)) != count):
            raise ValueError("Invalid landmark dataset arrays.")
        return features, labels, ids, times
    features, labels, ids, times = [], [], [], []

    def read_all(paths, opener):
        samples = []
        for path in paths:
            if path.suffix.lower() != ".csv" or "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            label = path.parent.name
            if not label:
                raise ValueError(f"{path}: expected a class folder")
            samples.append((f"{label}/{path.name}", path))
        samples.sort(key=lambda pair: (pair[0].casefold(), pair[0]))
        if len({key for key, _ in samples}) != len(samples):
            raise ValueError("Repeated class/filename IDs in input. Select a single dataset copy.")
        for key, path in samples:
            with opener(path) as stream:
                features.append(read_sample(stream, key))
            ids.append(key)
            labels.append(path.parent.name)
            stamp = STAMP.search(path.stem)
            times.append(int(stamp.group(1)) if stamp else -1)

    if source.is_dir():
        read_all((p.relative_to(source) for p in source.rglob("*") if p.is_file()),
                 lambda p: (source / p).open("rb"))
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            read_all((PurePosixPath(name) for name in archive.namelist() if not name.endswith("/")),
                     lambda p: archive.open(p.as_posix()))
    else:
        raise FileNotFoundError(f"Expected the original dataset ZIP or its class-folder directory: {source}")
    if not ids:
        raise ValueError("No sample CSVs found.")
    return np.stack(features), np.array(labels), np.array(ids), np.array(times, dtype=np.int64)

