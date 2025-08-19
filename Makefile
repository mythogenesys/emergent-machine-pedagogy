# ===== Makefile (FINAL, Reorganized v3 - PYTHONPATH fix) =====
IMAGE_NAME = emergent-pedagogy
CONFIG_FILE = experiments/config.yaml
VOLUME_MOUNT = -v $(shell pwd):/app
# --- FIX: Set the PYTHONPATH environment variable for all python commands ---
PYTHON_ENV = -e PYTHONPATH=/app

.PHONY: all build shell clean experiments analyze dry-run

all: build experiments analyze

build:
	docker build -t $(IMAGE_NAME) .

experiments:
	@echo "--- Running FULL experiment for ENTROPY ---"
	docker run --rm $(VOLUME_MOUNT) $(PYTHON_ENV) $(IMAGE_NAME) python -u experiments/run_full_suite.py --task entropy --config $(CONFIG_FILE) --mode full
	# ... (add other tasks here)

dry-run:
	@echo "--- Running DRY-RUN experiment for ENTROPY ---"
	docker run --rm $(VOLUME_MOUNT) $(PYTHON_ENV) $(IMAGE_NAME) python -u experiments/run_full_suite.py --task entropy --config $(CONFIG_FILE) --mode dry-run

analyze:
	@echo "--- Analyzing results ---"
	docker run --rm $(VOLUME_MOUNT) $(PYTHON_ENV) $(IMAGE_NAME) python -u experiments/analyze_results.py --output final_summary_results.csv

shell:
	@echo "--- Starting interactive shell with PYTHONPATH set ---"
	docker run -it --rm $(VOLUME_MOUNT) $(PYTHON_ENV) $(IMAGE_NAME) bash

clean:
	rm -rf results/* results_ssca/* final_summary_results.csv