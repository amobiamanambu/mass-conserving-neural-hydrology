.PHONY: validate test mass-balance compile manifest

validate:
	python workflow/validate_release.py

test:
	pytest -q

mass-balance:
	python conus_zero_shot_law_transfer_2026-09-30/scripts/00_test_mass_conservation.py

compile:
	python -m compileall -q conus_zero_shot_law_transfer_2026-09-30 conus_daily_process_observability_2026-10-12 workflow tests

manifest:
	python workflow/build_manifest.py
