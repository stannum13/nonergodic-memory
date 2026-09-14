.PHONY: smoke train reproduce extension figures sweep-overlap sweep-length sweep-components sweep-width test clean-results

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

sweep-overlap:
	bash scripts/sweep_overlap.sh

sweep-length:
	bash scripts/sweep_length.sh

sweep-components:
	bash scripts/sweep_components.sh

sweep-width:
	bash scripts/sweep_width.sh

test:
	pytest -q

clean-results:
	rm -f results/*.jsonl figures/*.png
