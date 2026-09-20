"""Reproducibility and environment manifests."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import platform
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    return int(seed)

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def hash_array(array):
    return hash_bytes(np.asarray(array).tobytes())

def hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def environment_manifest(packages=None):
    return {
        "python":platform.python_version(),
        "platform":platform.platform(),
        "packages":packages or {},
    }

def dataset_manifest(X,y=None,source=None):
    X=np.asarray(X)
    manifest={
        "shape":list(X.shape),
        "dtype":str(X.dtype),
        "feature_hash":hash_array(X),
        "source":source,
    }
    if y is not None:
        y=np.asarray(y)
        manifest.update({
            "target_shape":list(y.shape),
            "target_hash":hash_array(y),
        })
    return manifest

def save_manifest(manifest,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    return path
