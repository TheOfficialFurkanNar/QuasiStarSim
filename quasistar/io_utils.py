import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

from quasistar.config_logging import get_logger

logger = get_logger(__name__)


class UnsupportedFormatError(ValueError):
    """Raised when an unsupported format is requested."""
    pass


#  Numpy array için özel YAML representer ve constructor
def _ndarray_representer(dumper: yaml.Dumper, array: np.ndarray):
    return dumper.represent_sequence('!numpy.ndarray', array.tolist())

def _ndarray_constructor(loader: yaml.Loader, node: yaml.nodes.SequenceNode):
    return np.array(loader.construct_sequence(node))

yaml.add_representer(np.ndarray, _ndarray_representer)
yaml.add_constructor('!numpy.ndarray', _ndarray_constructor)


def save_results(
    data: Dict[str, Any],
    sim_id: str,
    output_dir: str = "results",
    fmt: str = "json",
    overwrite: bool = False,
    version: Optional[str] = None
) -> str:
    """
    Save simulation results to disk in various formats.

    Parameters
    ----------
    data : Dict[str, Any]
        Keys to simple types or numpy arrays.
    sim_id : str
        Base filename prefix for the outputs.
    output_dir : str
        Directory where files are saved.
    fmt : str
        One of 'json', 'yaml', 'npz', 'pickle', 'hdf5' (if h5py installed).
    overwrite : bool
        If False and file exists, raise FileExistsError.
    version : Optional[str]
        Custom version tag appended to sim_id. If None, use UTC timestamp.

    Returns
    -------
    filepath : str
        Full path to the saved file.

    Raises
    ------
    FileExistsError
        When file exists and overwrite is False.
    UnsupportedFormatError
        When fmt is not recognized.
    """
    # version tag ekle
    tag = version or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    sim_tagged = f"{sim_id}_{tag}"

    os.makedirs(output_dir, exist_ok=True)
    ext = fmt.lower()
    filename = f"{sim_tagged}_results.{ext}"
    filepath = os.path.join(output_dir, filename)

    # var olan dosya yönetimi
    if os.path.exists(filepath):
        if not overwrite:
            msg = f"save_results: '{filepath}' already exists and overwrite=False."
            logger.error(msg)
            raise FileExistsError(msg)
        # overwrite=True → yedeğini al
        bak = filepath + ".bak_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
        shutil.move(filepath, bak)
        logger.info(f"save_results: Backup created at {bak}")

    try:
        if ext == "json":
            serial = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in data.items()
            }
            with open(filepath, "w") as f:
                json.dump(serial, f, indent=2)

        elif ext == "yaml":
            serial = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in data.items()
            }
            with open(filepath, "w") as f:
                yaml.dump(serial, f, default_flow_style=False)

        elif ext == "npz":
            np.savez_compressed(filepath, **data)

        elif ext == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

        elif ext in ("hdf5", "h5"):
            if not H5PY_AVAILABLE:
                raise UnsupportedFormatError("h5py is not installed")
            with h5py.File(filepath, "w") as hf:
                for key, val in data.items():
                    hf.create_dataset(key, data=val)

        else:
            raise UnsupportedFormatError(f"Unsupported format '{fmt}'")

        logger.info(f"save_results: Saved to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"save_results: Error while saving: {e}")
        raise


def load_results(
    sim_id: str,
    output_dir: str = "results",
    fmt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load previously saved simulation results.

    Parameters
    ----------
    sim_id : str
        Base filename prefix (with version tag).
    output_dir : str
        Directory to search for files.
    fmt : Optional[str]
        Force specific format: 'json', 'yaml', 'npz', 'pickle', 'hdf5'.

    Returns
    -------
    data : Dict[str, Any]
        Loaded data.

    Raises
    ------
    FileNotFoundError
        When no matching file is found.
    UnsupportedFormatError
        When fmt is not recognized.
    """
    if not os.path.isdir(output_dir):
        msg = f"load_results: '{output_dir}' is not a directory."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # sim_id ile başlayan tüm dosyalar
    candidates = sorted([
        fn for fn in os.listdir(output_dir)
        if fn.startswith(f"{sim_id}_results.")
    ])
    if not candidates:
        msg = f"load_results: No files for '{sim_id}_results.*'"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # ext seçimi
    if fmt:
        ext = fmt.lower()
        filename = f"{sim_id}_results.{ext}"
        if filename not in candidates:
            msg = f"load_results: '{filename}' not found."
            logger.error(msg)
            raise FileNotFoundError(msg)
    else:
        filename = candidates[0]
        ext = filename.split(".")[-1]

    filepath = os.path.join(output_dir, filename)

    try:
        if ext == "json":
            with open(filepath, "r") as f:
                raw = json.load(f)
            data = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in raw.items()
            }

        elif ext == "yaml":
            with open(filepath, "r") as f:
                raw = yaml.safe_load(f)
            data = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in raw.items()
            }

        elif ext == "npz":
            npzf = np.load(filepath, allow_pickle=True)
            data = {k: npzf[k] for k in npzf.files}

        elif ext == "pickle":
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        elif ext in ("hdf5", "h5"):
            if not H5PY_AVAILABLE:
                raise UnsupportedFormatError("h5py is not installed")
            with h5py.File(filepath, "r") as hf:
                data = {k: hf[k][()] for k in hf.keys()}

        else:
            raise UnsupportedFormatError(f"Unsupported format '{ext}'")

        logger.info(f"load_results: Loaded from {filepath}")
        return data

    except Exception as e:
        logger.error(f"load_results: Error while loading: {e}")
        raise