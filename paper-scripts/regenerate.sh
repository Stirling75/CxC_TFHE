#!/bin/sh
# Same as `make check` for systems without make.
set -eu
cd "$(dirname "$0")"
FILES=$(grep -v '^#' summaries.txt)
python3 export_data.py $FILES
python3 latency_table.py $FILES
python3 breakdown_data.py
python3 estimates.py
python3 mvb_table.py
python3 noise_table.py
python3 limit_csv.py
python3 ratio_data.py out/data
python3 threads_table.py out/data out/threads-table.tex
sh compare.sh
