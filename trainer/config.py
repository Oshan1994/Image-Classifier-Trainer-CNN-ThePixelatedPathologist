from dataclasses import dataclass, field
from typing import List

@dataclass
class ClassEntryData:
    label: str
    dirs: List[str] = field(default_factory=list)

@dataclass
class ProjectState:
    project_dir: str = ""
    model_config: dict = field(default_factory=dict)
    hyper_config: dict = field(default_factory=dict)
    split_config: dict = field(default_factory=dict)
    class_entries: List[ClassEntryData] = field(default_factory=list)
    comparison_data: List[dict] = field(default_factory=list)
