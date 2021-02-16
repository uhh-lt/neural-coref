#!/usr/bin/env python3
"""
For tuebadz 11 we create a custom split.
"""
import sys
import math

# Tüba-DZ 11 has a total of 3816 documents
# There are a couple of obvious ways to split them.
# Following GerHotCoref we might use the first few docs as test and dev.
# But we don't want to have any of the SemEval 2010 docs in our test sets.
# That way we can use SemEval's singleton mentions for pre-training without worrying.
f = open(sys.argv[1])


total_size = 3816
dev_size = 382
test_size = 382
train_size = total_size - dev_size - test_size

assert dev_size + test_size + train_size == total_size
splits = [(train_size, "train.german.tuebdz_gold_conll"), (dev_size, "dev.german.tuebdz_gold_conll"), (test_size, "test.german.tuebdz_gold_conll")]

out_files = []
for _, split in splits:
    out_files.append(open(split, "w"))


def get_current_file(out_files, splits, doc_idx):
    total = 0
    for i, (required, _) in enumerate(splits):
        total += required
        if total > doc_idx:
            return out_files[i]


doc_idx = 0
for line in f:
    out = get_current_file(out_files, splits, doc_idx)
    out.write(line)
    if line.strip() == "#end document":
        doc_idx += 1
