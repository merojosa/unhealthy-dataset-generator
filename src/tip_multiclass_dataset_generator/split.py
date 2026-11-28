"""Leakage-safe train/val/test split by `group`.

Verbatim port of `stratified_group_split` from the junk-food-ads-classifiers-v2 multiclass
notebooks (same seed and fractions), so the `split` column written here matches the split
the notebooks compute. Keep both copies in sync.
"""
import numpy as np
import pandas as pd

SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# When a class has fewer groups than splits, its groups go to the splits in this order:
# 1 group -> train; 2 groups -> train + test (so the class is still evaluated); >= 3 -> all splits.
SUPPORT_ORDER = ("train", "test", "val")


def stratified_group_split(frame, label_col, group_col="group", fractions=SPLIT_FRACTIONS,
                           support_order=SUPPORT_ORDER, seed=SEED):
    """Assign whole groups to splits while keeping the per-class frame shares close to `fractions`.

    Classes are processed from the one with the fewest groups to the one with the most; each
    carrier group (largest first, seeded order on ties) goes to the split with the largest
    remaining deficit for that class, then for total frames. A repair pass then guarantees
    every class has at least one group in the splits given by `support_order`.

    Returns a dict group -> split name.
    """
    splits = list(fractions)
    counts = pd.crosstab(frame[group_col], frame[label_col])
    order = counts.index.to_numpy().copy()
    np.random.default_rng(seed).shuffle(order)
    counts = counts.loc[order]
    classes = list(counts.columns)
    sizes = counts.sum(axis=1)

    class_target = {s: counts.sum(axis=0) * f for s, f in fractions.items()}
    frame_target = {s: sizes.sum() * f for s, f in fractions.items()}
    filled = {s: pd.Series(0.0, index=classes) for s in splits}
    assignment = {}

    n_carriers = (counts > 0).sum(axis=0).sort_values(kind="stable")
    for cls in n_carriers.index:
        carriers = [g for g in counts.index if g not in assignment and counts.loc[g, cls] > 0]
        carriers.sort(key=lambda g: -counts.loc[g, cls])
        for g in carriers:
            split = max(splits, key=lambda s: (class_target[s][cls] - filled[s][cls],
                                               frame_target[s] - filled[s].sum()))
            assignment[g] = split
            filled[split] += counts.loc[g]

    def required(cls):
        return [s for s in support_order if s in splits][:min(int(n_carriers[cls]), len(splits))]

    def carriers_in(cls, split):
        return sum(1 for g, s in assignment.items() if s == split and counts.loc[g, cls] > 0)

    def can_leave(g, split):
        return all(carriers_in(c, split) > 1 or split not in required(c)
                   for c in classes if counts.loc[g, c] > 0)

    changed = True
    while changed:
        changed = False
        for cls in n_carriers.index:
            for split in required(cls):
                if carriers_in(cls, split):
                    continue
                donors = [g for g, s in assignment.items()
                          if s != split and counts.loc[g, cls] > 0 and can_leave(g, s)]
                if donors:
                    assignment[min(donors, key=lambda g: sizes[g])] = split
                    changed = True

    for cls in n_carriers.index:
        missing = [s for s in required(cls) if not carriers_in(cls, s)]
        if missing:
            print(f"  NOTE: class {cls!r} has no group in {missing} (its groups cannot be moved)")
        if n_carriers[cls] < len(splits):
            present = [s for s in splits if carriers_in(cls, s)]
            print(f"  NOTE: class {cls!r} has only {int(n_carriers[cls])} group(s); present in {present}")
    return assignment
