.PHONY: check eval

# Offline contract. Recomputes ECE/Brier from committed diagnostic
# probabilities and fails if README / DEEP_DIVE drift from the raw files.
check:
	python3 checks/check_offline.py
	python3 -m pytest -q tests

eval:
	python3 -m pytest -q tests
