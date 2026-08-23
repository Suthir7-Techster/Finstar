"""Data schema validation and quality gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feg_mlops.config import load_configs
from feg_mlops.data.quality import QualityChecker
from feg_mlops.data.schema import validate_schema
from feg_mlops.data.synthetic import SyntheticKycGenerator


def test_generated_data_passes_schema(config_dir):
    cfg = load_configs(config_dir)
    df = SyntheticKycGenerator(cfg.data).eval_frame()
    assert validate_schema(df) == []


def test_schema_detects_out_of_bounds_and_missing(config_dir):
    cfg = load_configs(config_dir)
    df = SyntheticKycGenerator(cfg.data).eval_frame()
    broken = df.copy()
    broken.loc[0, "credit_utilization"] = 7.5  # out of bounds
    issues = validate_schema(broken)
    assert any(i.column == "credit_utilization" and "out of" in i.issue for i in issues)

    missing = df.drop(columns=["income_annual"])
    assert any(i.column == "income_annual" for i in validate_schema(missing))


def test_quality_checker_passes_clean_data(config_dir):
    cfg = load_configs(config_dir)
    df = SyntheticKycGenerator(cfg.data).eval_frame()
    report = QualityChecker(cfg.data_settings).check(df, "eval")
    assert report.passed, report.issues


def test_quality_checker_catches_nulls_imbalance_and_rarity(config_dir):
    cfg = load_configs(config_dir)
    df = SyntheticKycGenerator(cfg.data).eval_frame().copy()

    df.loc[df.index[:50], "income_annual"] = np.nan  # nulls
    df.loc[df.index, "defaulted"] = 0  # class imbalance (no defaults at all)
    df.loc[df.index, "age_band"] = "35_plus"  # protected group vanishes

    report = QualityChecker(cfg.data_settings).check(df, "eval")
    checks = {i.check for i in report.errors}
    assert not report.passed
    assert "nulls" in checks
    assert "class_balance" in checks
    assert "group_representation" in checks


def test_determinism(config_dir):
    cfg = load_configs(config_dir)
    gen = SyntheticKycGenerator(cfg.data)
    pd.testing.assert_frame_equal(gen.train_frame(), gen.train_frame())


def test_label_bias_present_in_train_not_eval(config_dir):
    cfg = load_configs(config_dir)
    gen = SyntheticKycGenerator(cfg.data)
    train, eval_df = gen.train_frame(), gen.eval_frame()
    # Observed labels are biased in train; ground truth matches observed in eval.
    assert (train["defaulted"] != train["defaulted_ground_truth"]).any()
    assert (eval_df["defaulted"] == eval_df["defaulted_ground_truth"]).all()
