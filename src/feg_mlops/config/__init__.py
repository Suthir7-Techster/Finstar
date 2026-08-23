"""Typed configuration loading (policy-as-code).

Every threshold that gates a model (fairness limits, performance floors,
drift alert levels, decision bands) lives in ``configs/*.yaml`` and is
validated by pydantic models at load time. Changing policy therefore never
requires touching code — this is the "governance as code" mechanism the
source paper calls for (Section 4.3).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BiasConfig(_FrozenModel):
    label_flip_good_to_bad_unprivileged: float = Field(ge=0.0, le=1.0)
    label_flip_good_to_bad_privileged: float = Field(ge=0.0, le=1.0)
    credit_history_gap_months: float = Field(ge=0.0)


class DataConfig(_FrozenModel):
    seed: int
    n_train: int = Field(gt=0)
    n_eval: int = Field(gt=0)
    n_monitor_window: int = Field(gt=0)
    protected_attribute: str
    privileged_group: str
    unprivileged_group: str
    unprivileged_share: float = Field(gt=0.0, lt=1.0)
    bias: BiasConfig


class QualityConfig(_FrozenModel):
    max_null_fraction: float = Field(ge=0.0, le=1.0)
    max_duplicate_fraction: float = Field(ge=0.0, le=1.0)
    min_class_share: float = Field(gt=0.0, lt=1.0)
    min_group_share: float = Field(gt=0.0, lt=1.0)
    min_rows: int = Field(gt=0)


class DriftScenario(_FrozenModel):
    """Parameters for a simulated production window."""

    income_shift_log: float = 0.0
    utilization_add: float = 0.0
    unprivileged_share: float | None = None
    label_flip_good_to_bad_unprivileged: float | None = None

    def merged_over(self, base: BiasConfig, base_share: float) -> tuple[BiasConfig, float]:
        """Overlay this scenario onto the base data-generation bias settings."""
        flip = self.label_flip_good_to_bad_unprivileged
        bias = (
            base
            if flip is None
            else base.model_copy(update={"label_flip_good_to_bad_unprivileged": flip})
        )
        share = base_share if self.unprivileged_share is None else self.unprivileged_share
        return bias, share


class DataSettings(_FrozenModel):
    data: DataConfig
    quality: QualityConfig
    drift_scenarios: dict[str, DriftScenario] = Field(default_factory=dict)


class LogisticRegressionParams(_FrozenModel):
    C: float = Field(gt=0.0)
    max_iter: int = Field(gt=0)
    calibrate: bool = True


class GradientBoostingParams(_FrozenModel):
    learning_rate: float = Field(gt=0.0, le=1.0)
    max_iter: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    early_stopping: bool = True


class ModelSettings(_FrozenModel):
    class Models(_FrozenModel):
        logistic_regression: LogisticRegressionParams
        gradient_boosting: GradientBoostingParams

    class Selection(_FrozenModel):
        min_auc: float = Field(gt=0.5, lt=1.0)
        threshold_objective: str = "balanced_accuracy"

    class Decisions(_FrozenModel):
        approve_below: float = Field(ge=0.0, le=1.0)
        refer_between: tuple[float, float]
        decline_above: float = Field(ge=0.0, le=1.0)
        max_reason_codes: int = Field(gt=0, le=10)

        @field_validator("refer_between")
        @classmethod
        def _band_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
            if not v[0] < v[1]:
                raise ValueError("refer band must be (low, high) with low < high")
            return v

    models: Models
    selection: Selection
    decisions: Decisions


class FairnessGates(_FrozenModel):
    disparate_impact_min: float = Field(gt=0.0, le=1.0)
    statistical_parity_abs_max: float = Field(gt=0.0, le=1.0)
    equal_opportunity_abs_max: float = Field(gt=0.0, le=1.0)
    min_subgroup_recall: float = Field(ge=0.0, le=1.0)
    min_subgroup_count: int = Field(gt=0)


class MitigationConfig(_FrozenModel):
    reweighing: bool = True
    group_threshold_optimization: bool = True


class FairnessSettings(_FrozenModel):
    protected_attribute: str
    privileged_group: str
    unprivileged_group: str
    gates: FairnessGates
    mitigation: MitigationConfig


class PsiThresholds(_FrozenModel):
    warning: float = Field(ge=0.0)
    critical: float = Field(ge=0.0)


class JsThresholds(_FrozenModel):
    warning: float = Field(ge=0.0)
    critical: float = Field(ge=0.0)


class KsThresholds(_FrozenModel):
    warning: float = Field(ge=0.0, le=1.0)
    critical: float = Field(ge=0.0, le=1.0)


class DriftThresholds(_FrozenModel):
    psi: PsiThresholds
    js_divergence: JsThresholds
    ks: KsThresholds


class FairnessDriftThresholds(_FrozenModel):
    disparate_impact_abs_drop_warning: float = Field(ge=0.0, le=1.0)
    disparate_impact_abs_drop_critical: float = Field(ge=0.0, le=1.0)
    statistical_parity_abs_delta_warning: float = Field(ge=0.0, le=1.0)
    statistical_parity_abs_delta_critical: float = Field(ge=0.0, le=1.0)


class AlertConfig(_FrozenModel):
    min_features_for_dataset_alert: int = Field(gt=0)


class MonitoringSettings(_FrozenModel):
    drift: DriftThresholds
    fairness_drift: FairnessDriftThresholds
    alerts: AlertConfig


class FegConfig(_FrozenModel):
    """Aggregate of all policy documents."""

    data_settings: DataSettings = Field(alias="data_settings")
    model_settings: ModelSettings = Field(alias="model_settings")
    fairness_settings: FairnessSettings = Field(alias="fairness_settings")
    monitoring_settings: MonitoringSettings = Field(alias="monitoring_settings")

    @property
    def data(self) -> DataConfig:
        return self.data_settings.data

    @property
    def fairness(self) -> FairnessSettings:
        return self.fairness_settings


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return loaded


def load_configs(config_dir: str | Path) -> FegConfig:
    """Load and validate the four policy documents in ``config_dir``."""
    config_dir = Path(config_dir)
    return FegConfig(
        data_settings=DataSettings.model_validate(_read_yaml(config_dir / "data.yaml")),
        model_settings=ModelSettings.model_validate(_read_yaml(config_dir / "model.yaml")),
        fairness_settings=FairnessSettings.model_validate(_read_yaml(config_dir / "fairness.yaml")),
        monitoring_settings=MonitoringSettings.model_validate(
            _read_yaml(config_dir / "monitoring.yaml")
        ),
    )
