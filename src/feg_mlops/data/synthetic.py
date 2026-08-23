"""Deterministic synthetic KYC applicant generator with configurable bias.

The generator has two jobs:

1. Produce realistic KYC/credit applicants whose *ground-truth* default risk
   depends only on legitimate financial features.
2. Inject documented historical bias into the *training labels* (good
   outcomes flipped to bad for the unprivileged group), reproducing the
   discriminatory-lending history the source paper describes (Section 3.2).

Because the generator is seeded, every run of the pipeline trains on the
exact same data — a reproducibility property the paper flags as missing in
current practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feg_mlops.config import DataConfig, DriftScenario
from feg_mlops.data.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMN,
)

_AGE_THRESHOLD = 35


@dataclass(frozen=True)
class GenerationContext:
    """Everything that varies between splits / drift scenarios."""

    seed_offset: int
    income_shift_log: float = 0.0
    utilization_add: float = 0.0
    unprivileged_share: float | None = None
    label_flip_unprivileged: float = 0.0
    label_flip_privileged: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=float)


class SyntheticKycGenerator:
    """Generates KYC applicants with a latent ground-truth default process."""

    def __init__(self, config: DataConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_frame(self) -> pd.DataFrame:
        """Training data with historical label bias injected (by design)."""
        ctx = GenerationContext(
            seed_offset=1,
            label_flip_unprivileged=self._cfg.bias.label_flip_good_to_bad_unprivileged,
            label_flip_privileged=self._cfg.bias.label_flip_good_to_bad_privileged,
        )
        return self._generate(self._cfg.n_train, ctx)

    def eval_frame(self) -> pd.DataFrame:
        """Hold-out data reflecting true outcomes (no historical bias)."""
        return self._generate(self._cfg.n_eval, GenerationContext(seed_offset=2))

    def monitoring_window(self, scenario: DriftScenario, index: int) -> pd.DataFrame:
        """Simulated production window under a drift scenario.

        Windows reflect true outcomes (no historical label bias) except where
        the scenario explicitly corrupts outcome recording — keeping the
        mild/severe scenarios comparable in their label process.
        """
        share = (
            scenario.unprivileged_share
            if scenario.unprivileged_share is not None
            else self._cfg.unprivileged_share
        )
        ctx = GenerationContext(
            seed_offset=100 + index,
            income_shift_log=scenario.income_shift_log,
            utilization_add=scenario.utilization_add,
            unprivileged_share=share,
            label_flip_unprivileged=scenario.label_flip_good_to_bad_unprivileged or 0.0,
        )
        return self._generate(self._cfg.n_monitor_window, ctx)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate(self, n: int, ctx: GenerationContext) -> pd.DataFrame:
        cfg = self._cfg
        rng = np.random.default_rng(cfg.seed + ctx.seed_offset)

        share = (
            ctx.unprivileged_share if ctx.unprivileged_share is not None else cfg.unprivileged_share
        )
        is_unprivileged = rng.random(n) < share
        age = np.where(
            is_unprivileged,
            rng.uniform(21, _AGE_THRESHOLD - 0.01, n),
            rng.uniform(_AGE_THRESHOLD, 74, n),
        )
        age_band = np.where(age < _AGE_THRESHOLD, "under_35", "35_plus").astype(object)

        # --- Legitimate financial features --------------------------------
        # The unprivileged group is *represented* differently (shorter credit
        # histories and employment tenure, slightly lower income) — the proxy
        # channel through which label bias reaches the model.
        # Histories and tenure grow with age, so the protected attribute leaks
        # into legitimate-looking features (the proxy channel); the configured
        # gap adds the historical-discrimination component on top.
        history_gap = cfg.bias.credit_history_gap_months
        credit_history_months = np.clip(
            (age - 21) * 12 * 0.62
            + rng.normal(0, 30, n)
            - np.where(is_unprivileged, history_gap, 0.0),
            0,
            720,
        )

        log_income = rng.normal(np.log(52000), 0.55, n) - 0.06 * is_unprivileged
        log_income = log_income + ctx.income_shift_log
        income_annual = np.exp(log_income)

        employment_tenure_months = np.clip(
            (age - 21) * 12 * rng.uniform(0.25, 0.55, n) - np.where(is_unprivileged, 14.0, 0.0),
            0,
            600,
        )

        debt_to_income = np.clip(rng.beta(4, 9, n) * 0.9, 0, 1)
        num_open_accounts = rng.poisson(6, n).clip(0, 30)
        num_delinquencies_24m = rng.poisson(0.55, n).clip(0, 12)
        has_delinq = num_delinquencies_24m > 0
        months_since_last_delinquency = np.where(
            has_delinq, np.clip(rng.uniform(1, 24, n), 0, 240), 240.0
        )
        kyc_docs_verified = (rng.random(n) < 0.93).astype(float)
        device_trust_score = np.clip(rng.beta(9, 3, n), 0, 1)
        ip_country_risk = np.clip(rng.beta(1.5, 12, n), 0, 1)
        velocity_apps_90d = rng.poisson(1.1, n).clip(0, 20)
        avg_balance_6m = np.clip(income_annual * rng.uniform(0.15, 1.6, n) / 12, 0, 2_000_000)
        balance_volatility = np.clip(rng.beta(3, 8, n), 0, 1)
        pct_on_time_payments = np.clip(rng.beta(14, 3, n) * 0.35 + 0.65, 0, 1)
        credit_utilization = np.clip(rng.beta(5, 7, n) + ctx.utilization_add, 0, 1)
        has_previous_default = (rng.random(n) < 0.12).astype(float)

        features = pd.DataFrame(
            {
                "income_annual": income_annual,
                "debt_to_income": debt_to_income,
                "credit_history_months": credit_history_months,
                "num_open_accounts": num_open_accounts.astype(float),
                "num_delinquencies_24m": num_delinquencies_24m.astype(float),
                "employment_tenure_months": employment_tenure_months,
                "months_since_last_delinquency": months_since_last_delinquency,
                "kyc_docs_verified": kyc_docs_verified,
                "device_trust_score": device_trust_score,
                "ip_country_risk": ip_country_risk,
                "velocity_apps_90d": velocity_apps_90d.astype(float),
                "avg_balance_6m": avg_balance_6m,
                "balance_volatility": balance_volatility,
                "pct_on_time_payments": pct_on_time_payments,
                "credit_utilization": credit_utilization,
                "has_previous_default": has_previous_default,
            }
        )

        # --- Ground-truth default process (legitimate features only) ------
        p_default = self._latent_default_probability(features)
        defaulted = (rng.random(n) < p_default).astype(int)

        # --- Historical label bias (training / observed labels only) ------
        flips = np.zeros(n, dtype=bool)
        good = defaulted == 0
        flips |= good & (~is_unprivileged) & (rng.random(n) < ctx.label_flip_privileged)
        flips |= good & is_unprivileged & (rng.random(n) < ctx.label_flip_unprivileged)
        defaulted_observed = np.where(flips, 1, defaulted)

        df = features.copy()
        df[ID_COLUMN] = [f"KYC-{ctx.seed_offset:03d}-{i:06d}" for i in range(n)]
        df["age"] = age
        df[cfg.protected_attribute] = age_band
        df[LABEL_COLUMN] = defaulted_observed
        # Ground-truth label retained for honest evaluation even when the
        # observed labels are corrupted by simulated historical bias.
        df["defaulted_ground_truth"] = defaulted
        return df[
            list(
                (
                    ID_COLUMN,
                    "age",
                    cfg.protected_attribute,
                    *FEATURE_COLUMNS,
                    LABEL_COLUMN,
                    "defaulted_ground_truth",
                )
            )
        ]

    @staticmethod
    def _latent_default_probability(features: pd.DataFrame) -> np.ndarray:
        """Logistic ground-truth risk from standardized legitimate features."""
        z = {
            "income_annual": (np.log(features["income_annual"]) - np.log(52000)) / 0.55,
            "debt_to_income": (features["debt_to_income"] - 0.28) / 0.14,
            "credit_history_months": (features["credit_history_months"] - 120) / 80,
            "num_open_accounts": (features["num_open_accounts"] - 6) / 3,
            "num_delinquencies_24m": (features["num_delinquencies_24m"] - 0.55) / 0.9,
            "employment_tenure_months": (features["employment_tenure_months"] - 80) / 60,
            "months_since_last_delinquency": (features["months_since_last_delinquency"] - 100) / 60,
            "kyc_docs_verified": features["kyc_docs_verified"] - 0.93,
            "device_trust_score": (features["device_trust_score"] - 0.75) / 0.15,
            "ip_country_risk": (features["ip_country_risk"] - 0.11) / 0.09,
            "velocity_apps_90d": (features["velocity_apps_90d"] - 1.1) / 1.1,
            "avg_balance_6m": (features["avg_balance_6m"] - 4300) / 3000,
            "balance_volatility": (features["balance_volatility"] - 0.27) / 0.13,
            "pct_on_time_payments": (features["pct_on_time_payments"] - 0.83) / 0.08,
            "credit_utilization": (features["credit_utilization"] - 0.42) / 0.16,
            "has_previous_default": features["has_previous_default"] - 0.12,
        }
        weights = {
            "income_annual": -0.50,
            "debt_to_income": 0.65,
            "credit_history_months": -0.15,
            "num_open_accounts": -0.05,
            "num_delinquencies_24m": 0.55,
            "employment_tenure_months": -0.10,
            "months_since_last_delinquency": -0.10,
            "kyc_docs_verified": -0.25,
            "device_trust_score": -0.18,
            "ip_country_risk": 0.28,
            "velocity_apps_90d": 0.22,
            "avg_balance_6m": -0.15,
            "balance_volatility": 0.18,
            "pct_on_time_payments": -0.75,
            "credit_utilization": 0.80,
            "has_previous_default": 0.45,
        }
        logit = -0.55 + sum(z[k] * w for k, w in weights.items())
        return _sigmoid(np.asarray(logit, dtype=float))
