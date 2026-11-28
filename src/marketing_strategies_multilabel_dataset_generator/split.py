"""Leakage-safe train/val/test split by `group` for the multilabel dataset.

Multilabel variant of `tip_multiclass_dataset_generator.split.stratified_group_split`: groups
are atomic, and each group is described by its positive count per label instead of one class.
Labels are processed from the one with the fewest carrier groups to the most; each carrier
goes to the split with the largest remaining deficit for that label, then for total frames.
Groups with no positive label fill the frame deficit. A repair pass then guarantees every
label has a carrier in the splits given by `support_order`, when its groups allow it.
"""
import numpy as np
import pandas as pd

SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SUPPORT_ORDER = ("train", "test", "val")


def positives(frame: pd.DataFrame, label_cols) -> pd.DataFrame:
    """0/1 matrix from label cells ('1'/1/1.0 = positive; '0', blanks, NBSP and NaN = negative)."""
    return frame[label_cols].apply(lambda c: pd.to_numeric(c.astype(str).str.strip(), errors="coerce").eq(1)).astype(int)


def multilabel_group_split(frame, label_cols, group_col="group", fractions=SPLIT_FRACTIONS,
                           support_order=SUPPORT_ORDER, seed=SEED):
    """Returns a dict group -> split name."""
    splits = list(fractions)
    y = positives(frame, label_cols)
    counts = y.groupby(frame[group_col]).sum()
    sizes = frame.groupby(group_col).size()
    order = counts.index.to_numpy().copy()
    np.random.default_rng(seed).shuffle(order)
    counts, sizes = counts.loc[order], sizes.loc[order]
    labels = [c for c in counts.columns if counts[c].sum() > 0]

    label_target = {s: counts[labels].sum(axis=0) * f for s, f in fractions.items()}
    frame_target = {s: sizes.sum() * f for s, f in fractions.items()}
    filled = {s: pd.Series(0.0, index=labels) for s in splits}
    filled_frames = dict.fromkeys(splits, 0.0)
    assignment = {}

    def assign(g, split):
        assignment[g] = split
        filled[split] += counts.loc[g, labels]
        filled_frames[split] += sizes[g]

    n_carriers = (counts[labels] > 0).sum(axis=0).sort_values(kind="stable")
    for label in n_carriers.index:
        carriers = [g for g in counts.index if g not in assignment and counts.loc[g, label] > 0]
        carriers.sort(key=lambda g: -counts.loc[g, label])
        for g in carriers:
            assign(g, max(splits, key=lambda s: (label_target[s][label] - filled[s][label],
                                                 frame_target[s] - filled_frames[s])))
    for g in sorted((g for g in counts.index if g not in assignment), key=lambda g: -sizes[g]):
        assign(g, max(splits, key=lambda s: frame_target[s] - filled_frames[s]))

    def required(label):
        return [s for s in support_order if s in splits][:min(int(n_carriers[label]), len(splits))]

    def carriers_in(label, split):
        return sum(1 for g, s in assignment.items() if s == split and counts.loc[g, label] > 0)

    def can_leave(g, split):
        return all(carriers_in(l, split) > 1 or split not in required(l)
                   for l in labels if counts.loc[g, l] > 0)

    changed = True
    while changed:
        changed = False
        for label in n_carriers.index:
            for split in required(label):
                if carriers_in(label, split):
                    continue
                donors = [g for g, s in assignment.items()
                          if s != split and counts.loc[g, label] > 0 and can_leave(g, s)]
                if donors:
                    assignment[min(donors, key=lambda g: sizes[g])] = split
                    changed = True

    for label in n_carriers.index:
        missing = [s for s in required(label) if not carriers_in(label, s)]
        if missing:
            print(f"  NOTE: label {label!r} has no group in {missing} (its groups cannot be moved)")
    sparse = [l for l in n_carriers.index if n_carriers[l] < len(splits)]
    if sparse:
        print(f"  NOTE: {len(sparse)} label(s) have fewer groups than splits: {', '.join(sparse)}")
    return assignment
