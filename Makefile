

PY ?= python

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
	@printf "  $(C_CYAN)test$(C_RESET)               Run pytest$(C_RESET)\n"
	@printf "  $(C_CYAN)lint$(C_RESET)               Run ruff lint$(C_RESET)\n"
	@printf "  $(C_CYAN)format$(C_RESET)             Run ruff format$(C_RESET)\n"
	@printf "  $(C_CYAN)typecheck$(C_RESET)          Run mypy$(C_RESET)\n"
	@printf "\n"
	@printf "  $(C_CYAN)asv-check$(C_RESET)          asv check (import/discover only)$(C_RESET)\n"
	@printf "  $(C_CYAN)asv-setup$(C_RESET)          asv setup (create envs if not using existing)$(C_RESET)\n"
	@printf "  $(C_CYAN)bench$(C_RESET)              Run a benchmark (ASV)$(C_RESET)\n"
	@printf "  $(C_CYAN)bench-v$(C_RESET)            Run benchmark with -v$(C_RESET)\n"
	@printf "  $(C_CYAN)bench-continuous$(C_RESET)   Regression gate: BASE..HEAD (FACTOR)$(C_RESET)\n"


.PHONY: lint format typecheck test

lint:
	ruff check ./pynbodyext

format:
	ruff format ./pynbodyext

typecheck:
	mypy pynbodyext

test:
	pytest


.PHONY: asv-check asv-setup bench bench-v bench-continuous
asv-check:
	asv check

asv-setup:
	asv -vv setup

# ---- Generic benchmark entrypoints ----
# Usage:
#   make bench BENCH="bench_gravity.TimeGravityRealData"
BENCH ?=
ASV_ARGS ?=

# Only pass --bench when BENCH is non-empty
BENCH_FLAG := $(if $(strip $(BENCH)),--bench "$(BENCH)",)

bench:
	./scripts/asv_run.sh $(BENCH_FLAG) -- $(ASV_ARGS)

bench-v:
	./scripts/asv_run.sh -v $(BENCH_FLAG) -- $(ASV_ARGS)

# Continuous regression gate
BASE ?= origin/main
HEAD ?= HEAD
FACTOR ?= 1.10

bench-continuous:
	./scripts/asv_run.sh --continuous $(BENCH_FLAG) --base "$(BASE)" --head "$(HEAD)" --factor "$(FACTOR)" -- $(ASV_ARGS)