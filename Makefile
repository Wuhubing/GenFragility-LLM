SHELL := /bin/bash

PYTHON ?= /root/miniconda3/envs/genfragility/bin/python
BASE_MODEL ?= meta-llama/Llama-2-7b-hf

HF_TOKEN_FILE ?= keys/hf_key.txt
OPENAI_KEY_FILE ?= keys/openai_key.txt
HF_CACHE ?= /tmp/hf_cache

NODE_COUNT ?= 20000
GRAPH_FILE ?= /root/GenFragility-LLM/latest.pkl
RIPPLE_OUTPUT_DIR ?= results/experiments_ripples_fast_20k
NUM_EXPERIMENTS ?= 15
MAX_DISTANCE ?= 5
NUM_PROCESSES ?= 4

EXPERIMENT_FILE ?= $(RIPPLE_OUTPUT_DIR)/ripple_experiment_003.json
RUN_MAX_DISTANCE ?= d3
EPOCHS ?= 1
NUM_POISON ?= 12
NUM_NEUTRAL ?= 20
NUM_IRRELEVANT ?= 6
CONCURRENCY ?= 16

REPORT ?=
RIPPLE_METRICS_OUT ?=
DIAGNOSE_OUT ?=

.PHONY: help build-graph gen-ripples run-single run-exp003-d3 detect diagnose

help:
	@echo "Targets:"
	@echo "  make build-graph           Build graph database checkpoint (latest.pkl style outputs)"
	@echo "  make gen-ripples           Generate ripple_experiment_*.json from GRAPH_FILE"
	@echo "  make run-single            Run one integrated experiment (uses EXPERIMENT_FILE)"
	@echo "  make run-exp003-d3         Convenience target for exp_003 at d3"
	@echo "  make detect                Detect ripple effect from REPORT (or latest report)"
	@echo "  make diagnose              Output ripple metrics + clean accuracy by distance in one JSON"
	@echo ""
	@echo "Common overrides:"
	@echo "  EXPERIMENT_FILE=... RUN_MAX_DISTANCE=d2 CONCURRENCY=8"
	@echo "  GRAPH_FILE=... NUM_EXPERIMENTS=3 MAX_DISTANCE=3 NUM_PROCESSES=2"
	@echo "  REPORT=... RIPPLE_METRICS_OUT=... DIAGNOSE_OUT=..."

build-graph:
	@mkdir -p "$(HF_CACHE)"
	@$(PYTHON) -c "import run_1to1_fast as r; r.generate_1to1_graph_fast(node_count=$(NODE_COUNT))"

gen-ripples:
	@mkdir -p "$(RIPPLE_OUTPUT_DIR)"
	@$(PYTHON) -c "import src.generate_ripple_experiments as g; g.GRAPH_FILE='$(GRAPH_FILE)'; g.OUTPUT_DIR='$(RIPPLE_OUTPUT_DIR)'; g.NUM_EXPERIMENTS=$(NUM_EXPERIMENTS); g.MAX_DISTANCE=$(MAX_DISTANCE); g.NUM_PROCESSES=$(NUM_PROCESSES); g.main()"

run-single:
	@mkdir -p "$(HF_CACHE)"
	@HF_TOKEN="$$(cat '$(HF_TOKEN_FILE)')" \
	HUGGING_FACE_HUB_TOKEN="$$(cat '$(HF_TOKEN_FILE)')" \
	HUGGINGFACEHUB_API_TOKEN="$$(cat '$(HF_TOKEN_FILE)')" \
	OPENAI_API_KEY="$$(cat '$(OPENAI_KEY_FILE)')" \
	HF_HOME="$(HF_CACHE)" \
	TRANSFORMERS_CACHE="$(HF_CACHE)" \
	$(PYTHON) main.py \
		--mode single \
		--experiment_file "$(EXPERIMENT_FILE)" \
		--run_poison_pipeline \
		--base_model "$(BASE_MODEL)" \
		--poison_method factual \
		--max_distance "$(RUN_MAX_DISTANCE)" \
		--epochs "$(EPOCHS)" \
		--num_poison "$(NUM_POISON)" \
		--num_neutral "$(NUM_NEUTRAL)" \
		--num_irrelevant "$(NUM_IRRELEVANT)" \
		--concurrency_limit "$(CONCURRENCY)"

run-exp003-d3:
	@$(MAKE) run-single \
		EXPERIMENT_FILE="$(RIPPLE_OUTPUT_DIR)/ripple_experiment_003.json" \
		RUN_MAX_DISTANCE=d3 \
		CONCURRENCY=16

detect:
	@report="$(REPORT)"; \
	if [[ -z "$$report" ]]; then \
		report=$$(ls -t main_output/integrated_experiment_*/ripple_experiment_*/comparison_reports/*_comparison_*.json 2>/dev/null | head -n1); \
	fi; \
	if [[ -z "$$report" ]]; then \
		echo "No comparison report found. Set REPORT=<path/to/report.json>"; \
		exit 1; \
	fi; \
	out="$(RIPPLE_METRICS_OUT)"; \
	if [[ -z "$$out" ]]; then \
		out="$$(dirname "$$report")/ripple_metrics_v2.json"; \
	fi; \
	echo "Using report: $$report"; \
	echo "Output metrics: $$out"; \
	$(PYTHON) tools/report/detect_ripple_effect.py --report "$$report" --out "$$out"

diagnose:
	@report="$(REPORT)"; \
	if [[ -z "$$report" ]]; then \
		report=$$(ls -t main_output/integrated_experiment_*/ripple_experiment_*/comparison_reports/*_comparison_*.json 2>/dev/null | head -n1); \
	fi; \
	if [[ -z "$$report" ]]; then \
		echo "No comparison report found. Set REPORT=<path/to/report.json>"; \
		exit 1; \
	fi; \
	metrics="$(RIPPLE_METRICS_OUT)"; \
	if [[ -z "$$metrics" ]]; then \
		metrics="$$(dirname "$$report")/ripple_metrics_v2.json"; \
	fi; \
	out="$(DIAGNOSE_OUT)"; \
	if [[ -z "$$out" ]]; then \
		out="$$(dirname "$$report")/diagnose_summary.json"; \
	fi; \
	echo "Using report: $$report"; \
	echo "Output metrics: $$metrics"; \
	echo "Output summary: $$out"; \
	$(PYTHON) tools/report/detect_ripple_effect.py --report "$$report" --out "$$metrics"; \
	REPORT_PATH="$$report" METRICS_PATH="$$metrics" OUT_PATH="$$out" $(PYTHON) -c "import json,os;from collections import defaultdict;report=os.environ['REPORT_PATH'];metrics=os.environ['METRICS_PATH'];out=os.environ['OUT_PATH'];d=json.load(open(report,'r',encoding='utf-8'));m=json.load(open(metrics,'r',encoding='utf-8'));u=d.get('unified_results',[]);by=defaultdict(list);[by[r.get('distance','unknown')].append(r) for r in u];mean=lambda xs: sum(xs)/len(xs) if xs else 0.0;clean={k:{'count':len(v),'clean_accuracy_mean':mean([x.get('clean_accuracy',0.0) for x in v]),'poisoned_accuracy_mean':mean([x.get('poisoned_accuracy',0.0) for x in v]),'avg_confidence_change':mean([x.get('confidence_change',0.0) for x in v])} for k,v in by.items()};payload={'report':report,'total_samples':len(u),'ripple':m,'clean_accuracy_by_distance':clean,'clean_accuracy_overall':mean([x.get('clean_accuracy',0.0) for x in u]),'poisoned_accuracy_overall':mean([x.get('poisoned_accuracy',0.0) for x in u])};json.dump(payload,open(out,'w',encoding='utf-8'),ensure_ascii=False,indent=2);print('Saved:',out)"
