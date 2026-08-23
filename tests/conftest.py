"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture()
def modified_config_dir(tmp_path: Path) -> callable:
    """Factory: write the 4 policy YAMLs into a tmp dir with overrides.

    Overrides are deep-merged dicts keyed by file stem, e.g.
    ``{"fairness": {"gates": {"disparate_impact_min": 0.99}}}``.
    """

    def _make(overrides: dict[str, dict]) -> Path:
        for stem, patch in overrides.items():
            doc = yaml.safe_load((CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))

            def deep_merge(base: dict, delta: dict) -> None:
                for key, value in delta.items():
                    if isinstance(value, dict) and isinstance(base.get(key), dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value

            deep_merge(doc, patch)
            (tmp_path / f"{stem}.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
        for stem in ("data", "model", "fairness", "monitoring"):
            if not (tmp_path / f"{stem}.yaml").exists():
                (tmp_path / f"{stem}.yaml").write_text(
                    (CONFIG_DIR / f"{stem}.yaml").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        return tmp_path

    return _make
