.PHONY: smoke train reproduce extension figures sweep-overlap sweep-length sweep-components sweep-width sweep-depth sweep-interaction sweep-context-restart sweep-position-restart sweep-short-context sweep-budget-context test clean-results

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

sweep-depth:
	bash scripts/sweep_depth.sh

sweep-interaction:
	bash scripts/sweep_interaction.sh

sweep-context-restart:
	bash scripts/sweep_context_restart.sh

sweep-position-restart:
	bash scripts/sweep_position_restart.sh

sweep-short-context:
	bash scripts/sweep_short_context.sh

sweep-budget-context:
	bash scripts/sweep_budget_context.sh

test:
	pytest -q

clean-results:
	rm -f results/*.jsonl figures/*.png
