# Reproducible pipeline. Run from the repo root. Needs uv and Java 17.
RUN = uv run python

all: data train gate benchmark notebook

setup:
	uv sync --python 3.12
data:        ## ~115 MB download (ACS California 2018, NYC TLC Jan 2023)
	$(RUN) pipeline/download_data.py
train:       ## commit code first: lineage records the git commit
	$(RUN) pipeline/train_models.py
gate:        ## exit 1 if any model is blocked; rewrites REGISTRY.md
	$(RUN) -m governance.gate > REGISTRY.md; status=$$?; cat REGISTRY.md | head -8; exit $$status
test:
	uv run pytest -q
benchmark:   ## Spark vs pandas scaling, writes results/
	$(RUN) -m pipeline.benchmark_spark
notebook:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
report:
	$(RUN) report/build_pdf.py

.PHONY: all setup data train gate test benchmark notebook report
