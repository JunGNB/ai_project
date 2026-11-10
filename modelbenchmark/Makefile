# Developer entry points.  `make check` is the same gate as CI.
PY ?= .venv/bin/python
UV ?= uv

.PHONY: help setup check gate test build tasks live docs docs-write verify-data clean

help:
	@echo "make setup    - create .venv and install package (portable bootstrap)"
	@echo "make check    - full harness gate (imports/compile/tests/validate/dataset_integrity/perf_plan/docs/secrets/hygiene)"
	@echo "make gate     - same as check"
	@echo "make test     - pytest only"
	@echo "make verify-data - read-only dataset/structure/SHA-256 verification"
	@echo "make build    - build wheel into output/build/dist"
	@echo "make tasks    - list AI task board"
	@echo "make live     - gate including a real live smoke run"
	@echo "make docs     - check docs links + docs/code contract"
	@echo "make docs-write - regenerate docs/design/33-project-structure.md"

setup:
	./modelbench setup

check: gate

gate:
	$(PY) -m harness.gate

test:
	$(PY) -m pytest test/

verify-data:
	$(PY) tool/dataset/performance/fetch.py --verify
	$(PY) tool/dataset/performance/build.py --check
	$(PY) tool/dataset/accuracy/fetch.py --verify
	$(PY) tool/dataset/accuracy/build.py --check

build:
	mkdir -p output/build
	UV_CACHE_DIR=$$PWD/.uv-cache $(UV) build --wheel --no-build-isolation --out-dir output/build/dist

tasks:
	$(PY) -m harness.taskctl list

live:
	$(PY) -m harness.gate --live

docs:
	$(PY) -m harness.gate --only docs_links,docs_contract

docs-write:
	$(PY) -m harness.gen_structure --write

clean:
	rm -rf harness/reports/*.json .pytest_cache output/build src/*.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
