"""MLOps project structure and configuration helpers."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class ProjectConfig:
    name:str
    version:str
    random_seed:int=42
    environment:str="development"
    data_path:str="data"
    artifact_path:str="artifacts"

    def to_dict(self):
        return asdict(self)

def create_project_structure(root):
    root=Path(root)
    paths=[
        root/"data"/"raw",
        root/"data"/"processed",
        root/"artifacts",
        root/"models",
        root/"notebooks",
        root/"src",
        root/"tests",
        root/"configs",
    ]
    for p in paths:
        p.mkdir(parents=True,exist_ok=True)
    return paths

def save_config(config,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(config.to_dict(),indent=2),encoding="utf-8")
    return path

def load_config(path):
    return ProjectConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
