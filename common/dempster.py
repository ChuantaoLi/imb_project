"""common.dempster — Dempster-Shafer evidence combination.

Used by MC-EVHS and E-EVRS (the two evidential methods in the survey). A basic
belief assignment (bba / mass function) is represented as a dict mapping a
frozenset of class indices (a focal element) to a mass in [0, 1]; the masses of
one bba sum to 1. The universal set ``Theta`` (frame of discernment) is the
frozenset of all classes under consideration.

Dempster's rule of combination for two bbas m1, m2:

    m(A) = (1 / (1 - K)) * sum_{B ∩ C = A} m1(B) * m2(C)
    K    = sum_{B ∩ C = ∅}   m1(B) * m2(C)            (conflict mass)

When K == 1 (total conflict) the combination is undefined; we fall back to a
simple mixture to remain numerically safe.
"""
import numpy as np
from functools import reduce


def _theta(classes):
    return frozenset(classes)


def combine_pair(m1, m2):
    """Combine two bbas with Dempster's rule. Returns a new bba dict."""
    # union of focal elements
    focal = set(m1) | set(m2)
    combined = {}
    conflict = 0.0
    for b1, mb1 in m1.items():
        for b2, mb2 in m2.items():
            inter = b1 & b2
            w = mb1 * mb2
            if len(inter) == 0:
                conflict += w
            else:
                combined[inter] = combined.get(inter, 0.0) + w
    if conflict >= 1.0 - 1e-12:
        # total conflict: fall back to plain mixture (no normalization possible)
        out = {}
        keys = set(m1) | set(m2)
        for k in keys:
            out[k] = 0.5 * m1.get(k, 0.0) + 0.5 * m2.get(k, 0.0)
        return out
    norm = 1.0 - conflict
    return {k: v / norm for k, v in combined.items()}


def dempster_combine(masses):
    """Fold Dempster's rule over a list of bbas. Empty/single list is identity."""
    if not masses:
        return {}
    if len(masses) == 1:
        return dict(masses[0])
    return reduce(combine_pair, masses)


def pignistic_proba(bba, classes):
    """BetP pignistic transformation -> a normalized probability vector over
    `classes`. Each focal element's mass is split evenly among its members."""
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    p = np.zeros(len(classes))
    for focal, mass in bba.items():
        members = [idx[c] for c in focal if c in idx]
        if not members:
            continue
        share = mass / len(members)
        for j in members:
            p[j] += share
    s = p.sum()
    if s <= 0:
        return np.full(len(classes), 1.0 / len(classes))
    return p / s


def mass_from_proba(p, classes):
    """Build a bba from a class-probability vector.

    Evidential mapping: preserve the FULL posterior as soft singleton evidence,
    then reserve the residual ignorance for the universal set Theta.

    Let ``conf = max(p_norm)`` and ``p_norm = p / sum(p)``. We assign:
      * ``m({c_j}) = conf * p_norm[j]`` for every class with non-zero support;
      * ``m(Theta) = 1 - conf``.

    This keeps the mapping normalized, retains the whole soft label rather than
    only its argmax, and still leaves explicit ignorance mass for uncertain
    predictions.
    """
    classes = list(classes)
    p = np.asarray(p, dtype=float).ravel()
    bba = {}
    if len(p) == 0:
        return bba
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return {_theta(classes): 1.0}
    p = p / total
    conf = float(np.max(p))
    for j, pj in enumerate(p):
        if pj <= 0.0:
            continue
        bba[frozenset((classes[j],))] = conf * float(pj)
    bba[_theta(classes)] = bba.get(_theta(classes), 0.0) + (1.0 - conf)
    return bba
