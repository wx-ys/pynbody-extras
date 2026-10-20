

VENV ?= .venv
UV ?= uv
PY ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

# Set to 0 to disable colors: `make help COLOR=0`
COLOR ?= 1

ifeq ($(COLOR),1)
C_RESET  := \033[0m
C_BOLD   := \033[1m
C_DIM    := \033[2m
C_BLUE   := \033[34m
C_GREEN  := \033[32m
C_YELLOW := \033[33m
C_CYAN   := \033[36m
else
C_RESET  :=
C_BOLD   :=
C_DIM    :=
C_BLUE   :=
C_GREEN  :=
C_YELLOW :=
C_CYAN   :=
endif

.PHONY: help
help:
	@printf "pynbody-extras Makefile: \n"
	@printf "\n"
	@printf "$(C_BOLD)$(C_GREEN)Usage$(C_RESET): $(C_CYAN)make <target>$(C_RESET)\n"
	@printf "$(C_BOLD)$(C_GREEN)target$(C_RESET):\n"
	@printf "  $(C_CYAN)test$(C_RESET)               Run fast pytest (no testdata)$(C_RESET)\n"
	@printf "  $(C_CYAN)lint$(C_RESET)               Run ruff lint$(C_RESET)\n"
	@printf "  $(C_CYAN)format$(C_RESET)             Run ruff format$(C_RESET)\n"
	@printf "  $(C_CYAN)typecheck$(C_RESET)          Run mypy$(C_RESET)\n"
	@printf "  $(C_CYAN)check$(C_RESET)              lint + typecheck + fast tests$(C_RESET)\n"
	@printf "  $(C_CYAN)test-all$(C_RESET)           Run the whole pytest suite$(C_RESET)\n"
	@printf "  $(C_CYAN)test-data$(C_RESET)          Run only tests needing testdata$(C_RESET)\n"
	@printf "  $(C_CYAN)setup$(C_RESET)              Create .venv and install git hooks$(C_RESET)\n"
	@printf "  $(C_CYAN)docs-check$(C_RESET)         Validate work-item pairs and index$(C_RESET)\n"
	@printf "  $(C_CYAN)docs-index$(C_RESET)         Regenerate docs/work/INDEX.md$(C_RESET)\n"
	@printf "  $(C_CYAN)work-item$(C_RESET)          Create a bilingual work item$(C_RESET)\n"
	@printf "  $(C_CYAN)archive-item$(C_RESET)       Archive a done item into docs/history$(C_RESET)\n"
	@printf "\n"
	@printf "  $(C_CYAN)bench-check$(C_RESET)        asv check (import/discover only)$(C_RESET)\n"
	@printf "  $(C_CYAN)bench-setup$(C_RESET)        asv setup (create envs if not using existing)$(C_RESET)\n"
	@printf "  $(C_CYAN)bench$(C_RESET)              Run a benchmark (ASV)$(C_RESET)\n"
	@printf "  $(C_CYAN)bench-v$(C_RESET)            Run benchmark with -v$(C_RESET)\n"
	@printf "  $(C_CYAN)bench-head$(C_RESET)         Run ASV for latest commit only (HEAD)$(C_RESET)\n"
	@printf "  $(C_CYAN)bench-continuous$(C_RESET)   Regression gate: BASE..HEAD (FACTOR)$(C_RESET)\n"


.PHONY: setup require-dev lint format typecheck test test-all test-data check docs-check docs-index work-item archive-item

# One-command bootstrap for a fresh clone. `.python-version` pins the interpreter.
setup:
	$(UV) python install
	$(UV) sync --all-extras
	$(VENV)/bin/pre-commit install
	@printf "Ready. Run 'make check' for the fast loop.\n"

require-dev:
	@$(PY) -c "import ruff, mypy" >/dev/null 2>&1 || { echo "Missing dev tools. Run 'make setup' first."; exit 1; }

lint: require-dev
	$(PY) -m ruff check ./pynbodyext

format: require-dev
	$(PY) -m ruff format ./pynbodyext

typecheck: require-dev
	$(PY) -m mypy pynbodyext

# Fast inner loop for agents: no testdata download, no long-running cases.
test:
	$(PY) -m pytest -m "not data and not slow"

test-all:
	$(PY) -m pytest

test-data:
	$(PY) -m pytest -m data

docs-check:
	$(PY) scripts/check_docs.py

docs-index:
	$(PY) scripts/render_index.py

work-item:
	$(PY) scripts/new_work_item.py --type "$(TYPE)" --title-en "$(TITLE_EN)" --title-zh "$(TITLE_ZH)"

archive-item:
	$(PY) scripts/archive_work_item.py --id "$(ID)"

check: lint typecheck test docs-check

# OpenMP flags for cpp-test. Apple clang rejects a bare -fopenmp, so on macOS
# use Homebrew's keg-only libomp when it is installed (single-threaded
# otherwise); elsewhere use -fopenmp. Override with `make cpp-test OMP_FLAGS=`.
ifeq ($(shell uname -s),Darwin)
LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
OMP_FLAGS ?= $(if $(wildcard $(LIBOMP_PREFIX)/include/omp.h),-Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp)
else
OMP_FLAGS ?= -fopenmp
endif

.PHONY: cpp-test
cpp-test:
	g++ -std=c++17 -O2 $(OMP_FLAGS) -Icpp -o /tmp/pynbodyext_cpp_test \
		cpp/gravity/tests/test_core.cpp \
		$(wildcard cpp/gravity/*.cpp cpp/gravity/multipole/*.cpp) \
		-lm && /tmp/pynbodyext_cpp_test


.PHONY: bench-check bench-setup bench bench-v bench-continuous
bench-check:
	$(PY) -m asv check

bench-setup:
	$(PY) -m asv -vv setup

# ---- Generic benchmark entrypoints ----
# Usage:
#   make bench BENCH="bench_gravity.TimeGravityRealData"
BENCH ?=
ASV_ARGS ?=

# Only pass --bench when BENCH is non-empty
BENCH_FLAG := $(if $(strip $(BENCH)),--bench "$(BENCH)",)

bench:
	$(PY) -m asv run $(BENCH_FLAG) -- $(ASV_ARGS)

bench-v:
	$(PY) -m asv run -v $(BENCH_FLAG) -- $(ASV_ARGS)

# Run benchmarks at latest commit only (single revision)
bench-head:
	$(PY) -m asv run $(BENCH_FLAG) HEAD^! -- $(ASV_ARGS)


# Continuous regression gate
BASE ?= origin/main
HEAD ?= HEAD
FACTOR ?= 1.10

bench-continuous:
	$(PY) -m asv continuous --factor "$(FACTOR)" --split "$(BASE)" "$(HEAD)" $(BENCH_FLAG) -- $(ASV_ARGS)
