# Contributing

This repository preserves a frozen scientific experiment. Contributions are
welcome when they improve reproducibility without silently changing the
prespecified design or archived results.

1. Open an issue describing the proposed change and its scientific effect.
2. Keep raw provider data and credentials outside the repository.
3. Add or update tests for any code change.
4. Run `make compile`, `make validate`, `make test`, and `make mass-balance`.
5. Regenerate `MANIFEST.csv` only after all checks pass.
6. Record any result-changing modification in `CHANGELOG.md` and preserve the
   preceding tagged release.

Changes to model structure, split assignment, thresholds, observation
operators, or target-data access are new experiments and must receive a new
frozen protocol and version.
