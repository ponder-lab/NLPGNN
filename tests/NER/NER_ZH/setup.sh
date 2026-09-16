#!/bin/bash
# Fetch the Chinese NER corpus that bert_ner_crf_train.py and bert_ner_crf_test.py read.
#
# TFWriter opens its input as `os.path.join("Input", mode)`, so both CRF drivers need
#
#     Input/train
#     Input/valid
#
# and this repository carries neither: Input/ holds only a README naming the source. Without
# them the driver dies inside TFWriter before any model code runs. This performs what that
# README describes.
#
# THIS IS NOT THE ONLY THING BLOCKING THESE DRIVERS. They also call
# LoadCheckpoint(language='zh'), and that archive answers 403, so staging the data is necessary
# and not sufficient. Running them still fails at the checkpoint. The data
# is staged anyway because the two blockers are independent and this one is closable.
#
# PATHS ARE RELATIVE TO THIS DIRECTORY, because TFWriter resolves "Input" against the current
# directory. Run the drivers from here, the way this script stages the data, or the writer looks
# for Input/ beside wherever you started and reports the file absent while it sits correctly
# staged.
set -e
cd "$(dirname "$0")"

# Idempotent, because callers may run it once per repetition: cheap and silent when the data is
# already staged.
[ -f Input/train ] && [ -f Input/valid ] && exit 0

# EXCLUDED LOCALLY RATHER THAN VIA .gitignore, and the difference is deliberate. KG2E's data
# lands in data/, which this repository's .gitignore already covers with `**/data/`. Input/ is a
# tracked directory holding a tracked README, so 20 MB fetched into it would show as untracked
# in a subject whose working tree is otherwise clean, on every branch, forever. Editing
# .gitignore would fix that and would be a change to the subject's own source, which this is
# deliberately not: .git/info/exclude is per-clone, untracked, and identical in effect.
exclude="$(git rev-parse --git-dir 2>/dev/null)/info/exclude"
if [ -n "$exclude" ] && [ -w "$(dirname "$exclude")" ]; then
	for p in tests/NER/NER_ZH/Input/train tests/NER/NER_ZH/Input/valid; do
		grep -qxF "$p" "$exclude" 2>/dev/null || printf '%s\n' "$p" >> "$exclude"
	done
fi

mkdir -p Input
for f in train valid; do
	[ -f "Input/$f" ] || curl -sfL -o "Input/$f" \
		"https://raw.githubusercontent.com/kyzhouhzau/NLPGNNDATA/master/NER_CHINESE/$f"
done
