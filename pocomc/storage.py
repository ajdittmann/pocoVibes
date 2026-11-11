"""
Checkpoint storage for ``pocoMC``.

A checkpoint consists of two parts, both stored as NumPy ``.npz`` archives:

* A small **state file** holding everything that changes during a run (current
  particles, normalizing flow weights, geometry, scaler, counters, random number
  generator states, and -- when the checkpoint is taken in the middle of the MCMC
  mutation step -- the full state of the MCMC loop). State files are always
  written atomically (written to a temporary file, flushed to disk and then
  renamed over the target), so a crash during a save never corrupts an existing
  state file.

* An append-only **history store**, a directory ``{label}_history`` holding the
  initial prior samples (``static.npz``) and one file per past iteration
  (``row_XXXXXXXX.npz``), each written atomically. Rows are only ever appended,
  so the cost of a checkpoint does not grow with the number of iterations. A
  state file records how many history rows belong to it; rows beyond that number
  are ignored.

Values are stored natively wherever possible (numeric NumPy arrays and scalars),
together with a tag that records the original Python type so that values are
restored with exactly the same type and bits. Values that cannot be represented
natively (``None`` is tagged; strings are stored as UTF-8 bytes; object arrays
and other objects are serialized with ``dill``).
"""

import os
import json
from pathlib import Path

import dill
import numpy as np

__all__ = []

FORMAT_VERSION = 1
SEP = "/"
_ZIP_MAGIC = b"PK\x03\x04"


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------

def _native_dtype(dtype):
    """Return True if arrays of ``dtype`` can be stored natively (without pickling)."""
    if dtype.hasobject:
        return False
    if dtype.fields is not None:
        return all(_native_dtype(dtype.fields[name][0]) for name in dtype.names)
    if dtype.subdtype is not None:
        return _native_dtype(dtype.subdtype[0])
    return dtype.kind in "biufcS"


def _encode(value):
    """
    Encode a value as ``(numpy array, kind)``.

    The ``kind`` tag is used by ``_decode`` to restore the original type.
    """
    if value is None:
        return np.zeros(0, dtype=np.uint8), "none"
    if isinstance(value, bool):
        return np.asarray(value), "bool"
    if isinstance(value, int):
        return np.asarray(value, dtype=np.int64), "int"
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float64), "float"
    if isinstance(value, str):
        return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy(), "str"
    if isinstance(value, np.generic) and _native_dtype(value.dtype):
        return np.asarray(value), "npscalar"
    if isinstance(value, np.ndarray) and _native_dtype(value.dtype):
        return np.ascontiguousarray(value), "array"
    return np.frombuffer(dill.dumps(value), dtype=np.uint8).copy(), "pickle"


def _decode(array, kind):
    """Inverse of ``_encode``."""
    if kind == "none":
        return None
    if kind == "bool":
        return bool(array[()])
    if kind == "int":
        return int(array[()])
    if kind == "float":
        return float(array[()])
    if kind == "str":
        return np.asarray(array, dtype=np.uint8).tobytes().decode("utf-8")
    if kind == "npscalar":
        return np.asarray(array)[()]
    if kind == "array":
        return np.asarray(array)
    if kind == "pickle":
        return dill.loads(np.asarray(array, dtype=np.uint8).tobytes())
    raise ValueError(f"Unknown stored value kind '{kind}'.")


def _flatten(tree, prefix=""):
    """
    Flatten a nested dictionary into ``{path: value}``. Only dictionaries are
    descended into; every other value is a leaf. Keys must be strings without
    the separator.
    """
    flat = {}
    for key, value in tree.items():
        if not isinstance(key, str) or SEP in key:
            raise ValueError(f"Invalid key for checkpoint storage: {key!r}")
        path = prefix + key
        if isinstance(value, dict):
            if len(value) == 0:
                flat[path] = {}
            else:
                flat.update(_flatten(value, path + SEP))
        else:
            flat[path] = value
    return flat


def _unflatten(flat):
    """Inverse of ``_flatten``."""
    tree = {}
    for path, value in flat.items():
        parts = path.split(SEP)
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------

def _fsync_path(path):
    """Flush a file (already closed) to disk."""
    fd = os.open(str(path), os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_replace(tmp_path, path, fsync):
    """Flush ``tmp_path`` to disk and atomically move it to ``path``."""
    if fsync:
        _fsync_path(tmp_path)
    os.replace(str(tmp_path), str(path))


def _tmp_path(path):
    path = Path(path)
    return path.with_name(path.name + ".tmp")


# ---------------------------------------------------------------------------
# Flat dictionary <-> npz file
# ---------------------------------------------------------------------------

def _npz_pack(flat):
    """Convert ``{path: value}`` to the arrays stored in an npz archive."""
    arrays = {}
    kinds = {}
    for i, (path, value) in enumerate(flat.items()):
        name = f"v{i}"
        if isinstance(value, dict):  # empty dictionary
            arrays[name] = np.zeros(0, dtype=np.uint8)
            kinds[name] = (path, "emptydict")
            continue
        array, kind = _encode(value)
        arrays[name] = array
        kinds[name] = (path, kind)
    arrays["__index__"] = np.frombuffer(json.dumps(kinds).encode("utf-8"), dtype=np.uint8).copy()
    return arrays


def _npz_unpack(npz):
    """Inverse of ``_npz_pack`` for an open ``np.load`` archive."""
    kinds = json.loads(npz["__index__"].tobytes().decode("utf-8"))
    flat = {}
    for name, (path, kind) in kinds.items():
        if kind == "emptydict":
            flat[path] = {}
        else:
            flat[path] = _decode(npz[name], kind)
    return flat


def write_flat_file(path, flat, fsync=True):
    """Atomically write ``{path: value}`` to an npz file."""
    path = Path(path)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as f:
        np.savez(f, **_npz_pack(flat))
    _atomic_replace(tmp, path, fsync)


def read_flat_file(path):
    """Read ``{path: value}`` from a file written by ``write_flat_file``."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if not magic.startswith(_ZIP_MAGIC):
        raise ValueError(f"File '{path}' is not a pocoMC checkpoint. State files written by "
                         f"older pocoMC versions (dill pickles) are not supported.")
    with np.load(path, allow_pickle=False) as npz:
        return _npz_unpack(npz)


# ---------------------------------------------------------------------------
# History store
# ---------------------------------------------------------------------------

def history_name(label):
    """Directory name of the history store."""
    return f"{label}_history"


class HistoryStore:
    """
    Append-only store of per-iteration particles and the static prior samples.

    Parameters
    ----------
    path : ``str`` or ``Path``
        Directory of the history store.
    fsync : ``bool``
        Flush writes to disk before returning (default ``True``).
    """

    def __init__(self, path, fsync=True):
        self.path = Path(path)
        self.fsync = fsync

    @staticmethod
    def _row_name(k):
        return f"{int(k):08d}"

    def reset(self, run_id, static):
        """
        Start a new (empty) history for run ``run_id`` with static data ``static``
        (a flat dictionary). Any previous content is ignored.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        flat = dict(static)
        flat["__run_id__"] = run_id
        flat["__format_version__"] = FORMAT_VERSION
        write_flat_file(self.path / "static.npz", flat, fsync=self.fsync)

    def append(self, start, rows):
        """
        Write ``rows`` (a list of flat dictionaries) as history rows
        ``start, start+1, ...``. Existing rows with these indices are replaced.
        """
        for i, row in enumerate(rows):
            write_flat_file(self.path / f"row_{self._row_name(start + i)}.npz", row, fsync=self.fsync)

    def read(self, n_rows, run_id):
        """
        Read the static data and the first ``n_rows`` rows of the history.
        Raises ``ValueError`` if the history belongs to a different run.
        """
        static = read_flat_file(self.path / "static.npz")
        if static.pop("__run_id__", None) != run_id:
            raise ValueError(f"History directory '{self.path}' belongs to a different run "
                             f"(it has been overwritten since this state was saved).")
        static.pop("__format_version__", None)
        rows = []
        for k in range(n_rows):
            row_path = self.path / f"row_{self._row_name(k)}.npz"
            if not row_path.exists():
                raise ValueError(f"History directory '{self.path}' is missing iteration {k}.")
            rows.append(read_flat_file(row_path))
        return static, rows
