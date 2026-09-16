#!/bin/bash
# Fetch the knowledge-graph triples that run_tucker.py reads.
#
# TuckERLoader(base_path="data") opens three files this repository does not carry:
#
#     data/train.txt
#     data/valid.txt
#     data/test.txt
#
# Without them the driver fails at startup on a missing path. This script materializes them so
# run_tucker.py can be run from a fresh checkout. data/README.md already names the source; this
# performs what it describes.
#
# PATHS ARE RELATIVE TO THIS DIRECTORY, not to the repository root, because TuckERLoader resolves
# base_path against the current directory. Run run_tucker.py from here, the way this script stages
# the data, or the loader will look for data/ beside wherever you started instead and report the
# files absent while they sit correctly staged.
set -e
cd "$(dirname "$0")"

# Idempotent, because callers may run it once per repetition: cheap and silent when the data is
# already staged.
[ -f data/train.txt ] && [ -f data/valid.txt ] && [ -f data/test.txt ] && exit 0

mkdir -p data
for f in train.txt valid.txt test.txt; do
	[ -f "data/$f" ] || curl -sfL -o "data/$f" \
		"https://raw.githubusercontent.com/kyzhouhzau/NLPGNNDATA/master/KG2E/$f"
done
