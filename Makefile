.PHONY: install test lint validate seed curve eval-seeded eval-ceiling repair clean

PY ?= python
CONFIG ?= configs/default.yaml

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests scripts

validate:
	$(PY) -m toolsmith.cli validate

seed:
	$(PY) -m toolsmith.cli seed

curve:
	$(PY) scripts/make_repair_curve.py

eval-seeded:
	$(PY) -m toolsmith.cli eval --config $(CONFIG) \
		--schemas data/schemas/seeded --out results/seeded --name seeded

eval-ceiling:
	$(PY) -m toolsmith.cli eval --config configs/hand_tuned.yaml \
		--schemas data/schemas/clean --defects "" --out results/hand_tuned --name hand_tuned

repair:
	$(PY) -m toolsmith.cli repair --config $(CONFIG) --rounds 4

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
