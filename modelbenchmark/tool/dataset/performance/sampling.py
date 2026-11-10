#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic sampling primitives shared by the performance builder.

* ``deterministic_pick``: order a pool by ``sha256(seed|id)`` and take the first
  k.  Independent of file/row order and of Python's ``random`` state.
* ``allocate``: largest-remainder quota allocation over strata, with an optional
  per-stratum floor (so long-tail categories are represented).

Kept as an independent copy of ``tool/dataset/accuracy/sampling.py`` so the two
test sets can evolve separately (no cross-package import).
"""
from __future__ import annotations

import math
from typing import Dict, List

from tool.common.fs import sha256_text


def deterministic_pick(ids: List[str], k: int, seed: int = 42) -> List[str]:
    if k < 0 or k > len(ids):
        raise ValueError(f"cannot pick {k} from pool of {len(ids)}")
    return sorted(ids, key=lambda x: sha256_text(f"{seed}|{x}"))[:k]


def allocate(pool: Dict[str, int], k: int, min_each: int = 0, seed: int = 42) -> Dict[str, int]:
    """Largest-remainder allocation of ``k`` rows over strata ``pool``.

    ``pool[stratum]`` is the available count; the returned quota is always
    ``<= pool[stratum]`` and sums to ``k``.
    """
    strata = sorted(pool)
    total = sum(pool.values())
    if k > total:
        raise ValueError(f"allocate {k} > pool {total}")
    quota: Dict[str, int] = {s: 0 for s in strata}
    if min_each:
        for s in strata:
            quota[s] = min(min_each, pool[s])
        while sum(quota.values()) > k:
            s = max(strata, key=lambda x: (quota[x], sha256_text(str(x))))
            if quota[s] == 0:
                break
            quota[s] -= 1
    remaining = k - sum(quota.values())
    if remaining:
        weights = {s: pool[s] - quota[s] for s in strata}
        available = sum(weights.values())
        if available < remaining:
            raise ValueError("quota exceeds available pool")
        raw = {s: remaining * weights[s] / available for s in strata}
        base = {s: int(math.floor(raw[s])) for s in strata}
        left = remaining - sum(base.values())
        order = sorted(strata, key=lambda s: (-(raw[s] - base[s]), sha256_text(str(s))))
        for s in order[:left]:
            base[s] += 1
        for s in strata:
            quota[s] += base[s]
    assert sum(quota.values()) == k, (quota, k)
    for s in strata:
        assert quota[s] <= pool[s], (s, quota[s], pool[s])
    return quota
