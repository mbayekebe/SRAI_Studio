"""Data and model versioning utilities."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import shutil
import time

def file_digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def version_file(source,registry_dir,name=None):
    source=Path(source)
    registry=Path(registry_dir)
    registry.mkdir(parents=True,exist_ok=True)
    digest=file_digest(source)
    version_name=f"{name or source.stem}-{digest[:12]}{source.suffix}"
    target=registry/version_name
    shutil.copy2(source,target)
    metadata={
        "name":name or source.stem,
        "digest":digest,
        "source":str(source),
        "artifact":str(target),
        "created_at":time.time(),
    }
    meta_path=target.with_suffix(target.suffix+".json")
    meta_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

def registry_index(registry_dir):
    registry=Path(registry_dir)
    records=[]
    for meta in registry.glob("*.json"):
        records.append(json.loads(meta.read_text(encoding="utf-8")))
    return sorted(records,key=lambda x:x["created_at"])

def latest_version(registry_dir,name):
    matches=[r for r in registry_index(registry_dir) if r["name"]==name]
    return matches[-1] if matches else None
