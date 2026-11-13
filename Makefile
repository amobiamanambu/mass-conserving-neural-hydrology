.PHONY: validate test manifest

validate:
	python workflow/validate_release.py

test:
	pytest -q

manifest:
	python workflow/build_manifest.py
