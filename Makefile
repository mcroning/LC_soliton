# ==============================================================================
# LC_soliton — unified Makefile (local + cluster friendly)
# Place this file at the repo root.
# ==============================================================================

# -------- Settings --------
# -------- Settings --------
# Prefer python3; fall back to python. Allow override: make <target> PYTHON=/path/to/python3
PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python)
PIP    ?= $(PYTHON) -m pip

PKG    ?= lc_soliton
TEST_ARGS ?= -q
RUFF_ARGS ?= .
NPZ    ?= theta_out.npz
PNG    ?= theta_out.png
LATEST_NPZ := $(shell ls -t *.npz 2>/dev/null | head -n1)


# -------- Help --------
.PHONY: help list
help: list
list:
	@echo "Targets:"
	@printf "  %-15s %s\n" \
	  install "Editable install with dev tools (pytest, ruff)" \
	  test "Run pytest ($(TEST_ARGS))" \
	  lint "Static checks via ruff" \
	  fmt "Auto-format (ruff format + organize imports)" \
	  cuda-info "Print CUDA driver/toolkit/CuPy summary" \
	  run-demo "Short demo GPU run (writes theta_out.npz)" \
	  plot-demo "Plot theta_out.npz -> theta_out.png" \
	  run-long "Longer, finer GPU run" \
	  plot-latest "Plot newest .npz -> $(PNG)" \
	  plot "Plot NPZ=<file>.npz -> PNG=<file>.png" \
	  sweep "Sequential parameter sweep over B_LIST & BI_LIST" \
	  slurm-run "Submit a single run to Slurm (GPU)" \
	  slurm-sweep "Submit a sweep job to Slurm (GPU)" \
	  clean "Remove outputs/logs" \
	  clean-all "Also remove caches/builds"

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
.PHONY: lint fmt
lint:
	@$(PYTHON) -m ruff check $(RUFF_ARGS)

fmt:
	@$(PYTHON) -m ruff check --select I --fix $(RUFF_ARGS)   # organize imports
	@$(PYTHON) -m ruff format $(RUFF_ARGS)                   # format files

# -------- CUDA info --------
.PHONY: cuda-info
cuda-info:
	@echo "=== CUDA / CuPy info from $(PYTHON) ==="
	@$(PYTHON) -m $(PKG).utils.env_info || (echo "Note: env_info module prints driver/toolkit/CuPy summary"; exit 0)

# -------- Demos --------
.PHONY: run-demo plot-demo
run-demo:
	$(PYTHON) examples/run_theta2d.py --Nx 128 --Ny 128 --xaper 10.0 \
	  --steps 500 --dt 1e-3 --b 1.0 --bi 0.3 --intensity 1.0 \
	  --mobility 1.0 --save $(NPZ)

plot-demo:
	$(PYTHON) examples/plot_field.py $(NPZ) --save $(PNG)

# -------- Extended run --------
.PHONY: run-long
run-long:
	@echo "▶ Running a longer 2-D theta evolution on GPU…"
	$(PYTHON) examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0 \
	  --steps 10000 --dt 5e-4 --b 1.05 --bi 0.35 --intensity 1.0 \
	  --mobility 1.0 --save $(NPZ)

# -------- Plot helpers --------
.PHONY: plot-latest plot
plot-latest:
	@if [ -z "$(LATEST_NPZ)" ]; then \
	  echo "No .npz files found. Run 'make run-demo' or set NPZ=yourfile.npz"; exit 1; \
	fi; \
	echo "▶ Plotting $(LATEST_NPZ) → $(PNG)"; \
	$(PYTHON) examples/plot_field.py "$(LATEST_NPZ)" --save "$(PNG)"

plot:
	@if [ ! -f "$(NPZ)" ]; then echo "Missing NPZ=$(NPZ)"; exit 1; fi
	@echo "▶ Plotting $(NPZ) → $(PNG)"
	$(PYTHON) examples/plot_field.py "$(NPZ)" --save "$(PNG)"

# -------- Parameter sweep (sequential) --------
# Override on the command line as needed:
#   make sweep B_LIST="0.9 1.0 1.1" BI_LIST="0.2 0.3" STEPS=6000 DT=7.5e-4
B_LIST ?= 0.95 1.00 1.05
BI_LIST ?= 0.25 0.30 0.35
STEPS ?= 4000
DT ?= 1e-3

.PHONY: sweep
sweep:
	@echo "▶ Sweeping b in [$(B_LIST)] and bi in [$(BI_LIST)]"
	@for B in $(B_LIST); do \
	  for BI in $(BI_LIST); do \
	    OUT=theta_b$${B}_bi$${BI}.npz; \
	    echo "  → b=$${B}, bi=$${BI}  -> $${OUT}"; \
	    $(PYTHON) examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0 \
	      --steps $(STEPS) --dt $(DT) --b $${B} --bi $${BI} --intensity 1.0 \
	      --mobility 1.0 --save $${OUT}; \
	    $(PYTHON) examples/plot_field.py $${OUT} --save "$${OUT%.npz}.png"; \
	  done; \
	done
	@echo "✔ Sweep complete"

# -------- Slurm helpers (cluster) --------
# Override if your site uses different modules/partitions:
#   make slurm-run CUDA_MOD=cuda/11.8 ANA_MOD=anaconda/2024.10 SLURM_PART=gpu-a100
SLURM_PART ?= gpu
SLURM_TIME ?= 00:30:00
SLURM_GPUS ?= 1
CUDA_MOD   ?= cuda/12.2
ANA_MOD    ?= anaconda/2024.10
ENV_NAME   ?= lc_soliton

.PHONY: slurm-run
slurm-run:
	@echo "▶ Submitting Slurm run: partition=$(SLURM_PART) time=$(SLURM_TIME)"
	@sbatch --parsable <<'SB'
	#!/bin/bash
	#SBATCH -p $(SLURM_PART)
	#SBATCH --gres=gpu:$(SLURM_GPUS)
	#SBATCH -t $(SLURM_TIME)
	#SBATCH -J lc-theta
	#SBATCH -o lc-theta.%j.out
	set -xeu
	module purge
	module load $(ANA_MOD)
	module load $(CUDA_MOD)
	source "$$(conda info --base)/etc/profile.d/conda.sh"
	conda activate $(ENV_NAME)
	$(PIP) install -e . >/dev/null 2>&1 || true
	$(PYTHON) examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0 \
	--steps 8000 --dt 7.5e-4 --b 1.05 --bi 0.35 --intensity 1.0 \
	--mobility 1.0 --save theta_out.npz
	$(PYTHON) examples/plot_field.py theta_out.npz --save theta_out.png
	echo "Saved theta_out.npz and theta_out.png"
	SB

.PHONY: slurm-sweep
slurm-sweep:
	@echo "▶ Submitting Slurm sweep job: b in [$(B_LIST)], bi in [$(BI_LIST)]"
	@sbatch --parsable <<'SB'
	#!/bin/bash
	#SBATCH -p $(SLURM_PART)
	#SBATCH --gres=gpu:$(SLURM_GPUS)
	#SBATCH -t 02:00:00
	#SBATCH -J lc-sweep
	#SBATCH -o lc-sweep.%j.out
	set -xeu
	module purge
	module load $(ANA_MOD)
	module load $(CUDA_MOD)
	source "$$(conda info --base)/etc/profile.d/conda.sh"
	conda activate $(ENV_NAME)
	$(PIP) install -e . >/dev/null 2>&1 || true
	B_LIST="$(B_LIST)"
	BI_LIST="$(BI_LIST)"
	for B in $$B_LIST; do
	for BI in $$BI_LIST; do
		OUT=theta_b$${B}_bi$${BI}.npz
		$(PYTHON) examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0 \
		--steps $(STEPS) --dt $(DT) --b $${B} --bi $${BI} --intensity 1.0 \
		--mobility 1.0 --save $${OUT}
		$(PYTHON) examples/plot_field.py $${OUT} --save "$${OUT%.npz}.png"
	done
	done
	SB

# ==== Unified clean target (replaces ALL previous `clean:` rules) ===========
FIND        ?= find
RM          ?= rm -rf
CLEAN_FILES ?= *.npz *.png slurm-*.out lc-*.out lc-*.%j.out lc-theta.*.out lc-sweep.*.out \
               .coverage .pytest_cache *.egg-info
CLEAN_DIRS  ?= build dist .ruff_cache .venv venv \
               docs/__pycache__ scripts/__pycache__ lc_soliton/__pycache__ tests/__pycache__

.PHONY: clean
clean:
	@echo "▶ Cleaning outputs and logs"
	@rm -f *.npz *.png slurm-*.out lc-*.out lc-*.%j.out lc-theta.*.out lc-sweep.*.out || true
	@echo "[clean] removing files: $(CLEAN_FILES)"
	@$(FIND) . -maxdepth 1 -type f \( $(foreach f,$(CLEAN_FILES),-name "$(f)" -o) -false \) -print -delete || true
	@echo "[clean] removing directories: $(CLEAN_DIRS)"
	@$(foreach d,$(CLEAN_DIRS), [ -d "$(d)" ] && echo "$(d)" && $(RM) "$(d)" || true; )
	@echo "[clean] removing per-module __pycache__ (recursive)"
	@$(FIND) . -type d -name "__pycache__" -print -exec $(RM) {} + || true
# ==== End unified clean =====================================================


# Merge with your existing help target or keep this if you don't have one
help:
	@echo "make install         - Editable install with dev tools"
	@echo "make test            - Run pytest suite"
	@echo "make lint            - Ruff static checks"
	@echo "make cuda-info       - GPU/CuPy diagnostics"
	@echo "make clean           - Remove temp build artifacts and caches (safe)"
	@echo "make clean-dry       - Preview what 'clean' would remove"
	@echo "make clean-cupy-cache- Remove CuPy caches"
	@echo "make clean-slurm     - Remove Slurm outputs"
	@echo "make superclean      - clean + extra cleanup (safe)"
	@echo "make distclean       - SUPER aggressive cleanup (requires FORCE=1)"


# --- BEGIN AUTO-ADDED DOCS TARGETS ---
# ==== Docs pipeline (append-only block) ======================================
# Variables
DOCS_DIR     ?= docs
SCRIPTS_DIR  ?= scripts
DOCS_SCRIPT  ?= $(SCRIPTS_DIR)/generate_docs.py



# --- END AUTO-ADDED DOCS TARGETS ---

# Include shared documentation targets (safe if missing, due to leading '-')
-include scripts/Makefile.docs.addon

