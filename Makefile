CFG ?= configs/quick.yaml
RES ?= results/merged
PY  ?= python

.PHONY: test zoo replacements graphs validate probes all tables figures paper lm interpbench

test:
	$(PY) -m pytest -q

zoo:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stage zoo
replacements:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stage replacements
graphs:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stage graphs
validate:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stage validate
probes:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stage probes
logit-bypass:
	$(PY) scripts/logit_bypass.py --results $(RES) --ckpt $(subst results,checkpoints,$(RES))
all:
	$(PY) scripts/run_pipeline.py --config $(CFG) --stage all
	$(PY) scripts/logit_bypass.py --results $(RES) --ckpt $(subst results,checkpoints,$(RES))

tables:
	$(PY) scripts/make_tables.py --results $(RES) --paper paper/generated
figures:
	$(PY) scripts/make_figures.py --results $(RES) --out paper/figures
paper: tables figures
	cd paper && latexmk -pdf main.tex

lm:  ## GPT-2 two-hop validation (GPU)
	PYTHONPATH=src $(PY) -m cdc.realdata.lm_twohop --model gpt2 --out results/gpt2_twohop
interpbench:
	PYTHONPATH=src $(PY) -m cdc.realdata.interpbench_eval --list

merge:  ## combine the three parts into results/merged
	$(PY) scripts/merge_results.py results/part_A results/part_B results/part_C --out results/merged
