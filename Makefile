.PHONY: smoke train reproduce extension figures test clean-results

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

smoke:
	bash scripts/smoke.sh

train:
	$(PYTHON) src/train.py --config configs/reproduce.yaml --seeds 0 1 2 --output-dir checkpoints/reproduce --results results/training.jsonl

reproduce:
	bash scripts/reproduce.sh

extension:
	bash scripts/extension.sh

figures:
	bash scripts/figures.sh

test:
	pytest -q

clean-results:
	rm -f results/*.jsonl figures/*.png

