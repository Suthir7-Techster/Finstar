# Contributing

Thanks for improving feg-mlops. This project exists to demonstrate *operationalized*
responsible AI — contributions that weaken the governance guarantees will not be merged,
even if they improve raw model metrics.

## Setup

```bash
git clone <your-fork> && cd feg-mlops
python -m pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Branch from `main` (`feat/...`, `fix/...`).
2. Make changes with tests; keep or raise coverage (≥85% enforced).
3. Ensure the full quality gate passes locally:

   ```bash
   make lint && make typecheck && make coverage
   ```

4. Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:` ...). CI runs the same
   gates plus the FEG pipeline job — a model that fails fairness policy fails CI.
5. PRs require green CI and a review.

## Ground rules specific to this repo

- **Policy changes belong in `configs/*.yaml`.** Thresholds must not be hard-coded in
  source. If a change alters governance semantics, document it in an ADR.
- **Every pipeline stage emits audit events.** New stages must append to the audit trail
  before doing their work and after completing it.
- **Tests pin fairness semantics.** If you change metric math, update the hand-computed
  expectations in `tests/unit/test_fairness_metrics.py` and justify in the PR.
- **No PII in logs.** Decision events carry feature content hashes, never raw values.
- New dependencies require justification: the reference implementation stays lean
  (numpy/pandas/scikit-learn/shap/pydantic/FastAPI).
