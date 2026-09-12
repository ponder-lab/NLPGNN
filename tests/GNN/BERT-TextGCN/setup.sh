#!/bin/bash
# Fetch the R8 text corpus that build_graph.py reads.
#
# build_graph.py expects two files that this repository does not carry:
#
#     data/R8/r8-train-all-terms.txt
#     data/R8/r8-test-all-terms.txt
#
# Without them it fails at startup on a missing path. This script materializes them so the
# driver can be run from a fresh checkout.
#
# IT DOES NOT FETCH THE BERT CHECKPOINT, which build_graph.py also needs by way of
# LoadCheckpoint(language='en', cased=False). That download is large, and LoadCheckpoint skips
# it whenever uncased_L-12_H-768_A-12/ already exists and is non-empty, so an automated fetch
# placed here would be silently bypassed in the common case and would duplicate a step the
# sibling recipe in ../gnn_for_nlp/setup.sh already performs. Obtain the checkpoint by whichever
# of those routes suits you; this script is only about the text inputs.
set -e
cd "$(git rev-parse --show-toplevel)"

# Idempotent, because callers may run it once per repetition: cheap and silent when the data is
# already staged.
[ -f data/R8/r8-train-all-terms.txt ] && [ -f data/R8/r8-test-all-terms.txt ] && exit 0

mkdir -p data/R8
for f in r8-train-all-terms.txt r8-test-all-terms.txt; do
	[ -f "data/R8/$f" ] || curl -sfL -o "data/R8/$f" \
		"https://raw.githubusercontent.com/Cynwell/Text-Level-GNN/main/$f"
done
