#!/bin/bash
# Materialize the R8 inputs for text_gcn.py (recipe: Python-Subjects#14).
set -e
cd "$(git rev-parse --show-toplevel)"
[ -f data/R8/train_nodes.npy ] && exit 0
mkdir -p data/R8
for f in r8-train-all-terms.txt r8-test-all-terms.txt; do
    [ -f "data/R8/$f" ] || curl -sfL -o "data/R8/$f" "https://raw.githubusercontent.com/Cynwell/Text-Level-GNN/main/$f"
done
D=uncased_L-12_H-768_A-12
B=https://huggingface.co/google/bert_uncased_L-12_H-768_A-12/resolve/main
mkdir -p "$D"
if [ ! -f "$D/bert_config.json" ]; then
    curl -sfL -o "$D/bert_model.ckpt.data-00000-of-00001" "$B/bert_model.ckpt.data-00000-of-00001"
    curl -sfL -o "$D/bert_model.ckpt.index" "$B/bert_model.ckpt.index"
    curl -sfL -o "$D/vocab.txt" "$B/vocab.txt"
    curl -sfL -o "$D/bert_config.json" "$B/config.json"
fi
PYTHONPATH=. python3.10 tests/GNN/BERT-TextGCN/build_graph_gen.py
mv data/train_*.npy data/test_*.npy data/R8/
