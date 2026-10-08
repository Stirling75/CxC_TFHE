# Regenerate the paper's figure data and tables from ../results.
# Usage: make            (figure data and tables, python3 only)
#        make check      (compare against the data used in the paper)
#        make sweep      (recompute the limit sweep of Figure 4, several minutes)
PY ?= python3
SUMMARIES := $(shell grep -v '^\#' summaries.txt)

all: data table breakdown estimates mvb noise limit ratio threads

data:
	$(PY) export_data.py $(SUMMARIES)

table:
	$(PY) latency_table.py $(SUMMARIES)

breakdown:
	$(PY) breakdown_data.py

estimates:
	$(PY) estimates.py

mvb:
	$(PY) mvb_table.py

noise:
	$(PY) noise_table.py

limit:
	$(PY) limit_csv.py

sweep:
	$(PY) limit_sweep.py

ratio: data
	$(PY) ratio_data.py out/data

threads: data
	$(PY) threads_table.py out/data out/threads-table.tex

check: all
	@sh compare.sh

clean:
	rm -rf out

.PHONY: all data table breakdown estimates mvb noise limit sweep ratio threads check clean
