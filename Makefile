# -------- Settings --------
PYTHON ?= python
PKG    ?= lc_soliton

# Ruff/pytest config can also live in pyproject.toml
TEST_ARGS ?= -q
RUFF_ARGS ?= .

# -------- Help --------
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install     - Editable install with dev tools (pytest, ruff)"
	@echo "  test        - Run pytest ($(TEST_ARGS))"
	@echo "  lint        - Static checks via ruff (lint only)"
	@echo "  fmt         - Auto-format via ruff (format + organize imports)"
	@echo "  cuda-info   - Print CUDA driver/toolkit/CuPy summary"
	@echo "  clean       - Remove build artifacts and caches"

# -------- Dev setup --------
.PHONY: install
install:
	@$(PYTHON) -m pip install -U pip
	@$(PYTHON) -m pip install -e .[dev]

# -------- Tests --------
.PHONY: test
test:
	@$(PYTHON) -m pytest $(TEST_ARGS)

# -------- Lint & Format --------
.PHONY: lint
lint:
	@$(PYTHON) -m ruff check $(RUFF_ARGS)

.PHONY: fmt
fmt:
	@$(PYTHON) -m ruff check --select I --fix $(RUFF_ARGS)   # organize imports
	@$(PYTHON) -m ruff format $(RUFF_ARGS)                   # format files

# -------- CUDA info --------
.PHONY: cuda-info
cuda-info:
	@echo "=== Checking CUDA environment ==="
	@$(PYTHON) -m $(PKG).utils.env_info

# -------- Clean --------
.PHONY: clean
clean:
	@rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
# --------- Demos ---------	
.PHONY: run-demo plot-demo

run-demo:
	python examples/run_theta2d.py --Nx 128 --Ny 128 --xaper 10.0 \
	  --steps 500 --dt 1e-3 --b 1.0 --bi 0.3 --intensity 1.0 \
	  --mobility 1.0 --save theta_out.npz

plot-demo:
	python examples/plot_field.py theta_out.npz --save theta_out.png
