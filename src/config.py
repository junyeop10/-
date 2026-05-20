"""Application configuration loading and versioning helpers."""

from __future__ import annotations

import json
from dataclasses import MISSING
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CONFIG_VERSION = "2.0"
DEFAULT_CONFIG_PATH = Path("data/app_config.json")


@dataclass
class ScoringConfig:
    rule_weight: float = 0.45
    embedding_weight: float = 0.2
    metadata_weight: float = 0.1
    filename_weight: float = 0.1
    feedback_weight: float = 0.1
    duplicate_weight: float = 0.05
    llm_override_threshold: float = 0.8
    review_threshold: float = 0.65
    ambiguity_margin: float = 0.08
    low_confidence_threshold: float = 0.55


@dataclass
class OCRConfig:
    backend: str = "rapidocr"
    enabled: bool = True
    min_text_chars: int = 100
    max_pages: int = 5
    timeout_seconds: int = 30
    cache_enabled: bool = True


@dataclass
class LLMProviderConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 45
    api_key_env: str = ""


@dataclass
class EmbeddingStoreConfig:
    enabled: bool = True
    backend: str = "hdf5"
    path: str = "data/embeddings.h5"
    use_legacy_sqlite_cache: bool = True
    migrate_legacy_cache_on_hit: bool = True
    dual_write_legacy_sqlite: bool = False
    lock_timeout_seconds: float = 15.0


@dataclass
class FileSystemConfig:
    managed_root: str = "organized_files"
    preview_only_default: bool = True
    allow_overwrite: bool = False
    snapshot_dir: str = "data/snapshots"
    manifest_dir: str = "data/manifests"


@dataclass
class AppConfig:
    version: str = CONFIG_VERSION
    taxonomy_path: str = "data/categories.json"
    database_path: str = "data/classifier.db"
    default_input_dir: str = "input_files"
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    embedding: EmbeddingStoreConfig = field(default_factory=EmbeddingStoreConfig)
    filesystem: FileSystemConfig = field(default_factory=FileSystemConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(data: dict[str, Any], cls: type[Any]) -> Any:
    field_values: dict[str, Any] = {}
    for field_name, field_info in cls.__dataclass_fields__.items():
        if field_name not in data:
            continue
        value = data[field_name]
        nested_default = None
        if field_info.default_factory is not MISSING and callable(field_info.default_factory):
            try:
                nested_default = field_info.default_factory()
            except TypeError:
                nested_default = None
        if nested_default is not None and hasattr(nested_default, "__dataclass_fields__") and isinstance(value, dict):
            field_values[field_name] = _merge_dataclass(value, type(nested_default))
        else:
            field_values[field_name] = value
    return cls(**field_values)


def default_config() -> AppConfig:
    return AppConfig()


def load_app_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        config = default_config()
        save_app_config(config, config_path)
        return config

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = _merge_dataclass(raw, AppConfig)
    if not getattr(config, "version", ""):
        config.version = CONFIG_VERSION
    return config


def save_app_config(config: AppConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path
