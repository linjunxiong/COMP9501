"""CPU-only readers for training logs and local checkpoint metadata."""
import csv
import io
from pathlib import Path
import pickletools
import zipfile


class MetricsSnapshotError(RuntimeError):
    """The CSV snapshot is incomplete, changing, or structurally invalid."""


def read_metrics_snapshot(path):
    """Read one complete CSV snapshot; never remove NULs or invent missing values."""
    path = Path(path)
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise MetricsSnapshotError(f"cannot read {path}: {error}") from error
    signature = lambda stat: (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature(before) != signature(after):
        raise MetricsSnapshotError(f"CSV changed while being read: {path}")
    if not data or b"\x00" in data or not data.endswith((b"\n", b"\r")):
        raise MetricsSnapshotError(f"CSV is empty, contains NUL, or has an unfinished line: {path}")
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader)
        if "train/loss_sds" not in header or len(set(header)) != len(header):
            raise MetricsSnapshotError(f"CSV training header is incomplete: {path}")
        rows = []
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise MetricsSnapshotError(f"CSV row width does not match its header: {path}")
            row = dict(zip(header, values))
            for key, value in row.items():
                if key.startswith("train/") and value:
                    float(value)  # malformed values are unreadable; NaN/Inf stay visible to callers
            rows.append(row)
        return rows
    except (UnicodeError, csv.Error, StopIteration, ValueError) as error:
        raise MetricsSnapshotError(f"cannot parse CSV snapshot {path}: {error}") from error


def checkpoint_global_step(path):
    """Read Lightning's leading scalar metadata without torch, unpickling, or CUDA."""
    with zipfile.ZipFile(path) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        pending = False
        for opcode, value, _ in pickletools.genops(archive.read(metadata)):
            if pending:
                if opcode.name in {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT", "FRAME"}:
                    continue
                if opcode.name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG"}:
                    if isinstance(value, int) and value >= 0:
                        return value
                break
            if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                if value == "global_step":
                    pending = True
                elif value == "state_dict":
                    break
    raise ValueError(f"Lightning global_step metadata is missing or invalid: {path}")
