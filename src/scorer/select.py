"""Exact selection: among one photo's candidates, the set maximising sum(s - tau) such that no two kept candidates
conflict (overlap too much). Candidates with s <= tau are dropped first; the conflict graph of the others splits into
connected components; a single-candidate component is kept as is, a larger one is solved as a 0/1 integer program
(one variable per candidate, one x_a + x_b <= 1 per conflict) with scipy.optimize.milp (HiGHS). Raises RuntimeError
unless the solver proves an optimal solution."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TAU = 0.24


def _solve_component(w: np.ndarray, ea: np.ndarray, eb: np.ndarray, options: dict | None, results: list | None):
    """Max sum(w x) over x in {0, 1}^n with x[ea[e]] + x[eb[e]] <= 1 for every edge e; returns the 0/1 vector."""
    n = len(w)
    A = coo_matrix((np.ones(2 * len(ea)), (np.repeat(np.arange(len(ea)), 2), np.stack([ea, eb], 1).ravel())),
                   shape=(len(ea), n))
    # One HiGHS thread per process (many photos are solved in parallel); the optimum value is unchanged.
    opts = dict(options or {})
    opts.setdefault("threads", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # scipy: "unrecognized options ... passed to HiGHS verbatim"
        r = milp(c=-w, constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(n), bounds=Bounds(0, 1),
                 options=opts)
    if results is not None:
        results.append(r)
    if r.status != 0:
        raise RuntimeError(f"milp did not prove an optimal solution: status {r.status}, {r.message}")
    return r.x > 0.5


def select(sub: dict, s: np.ndarray, tau: float = TAU, options: dict | None = None,
           results: list | None = None) -> np.ndarray:
    """Indices of the conflict-free set maximising sum(s - tau); conflicts are the index pairs in sub["conf"]."""
    w = s - tau
    pos = np.flatnonzero(w > 0)
    if not len(pos):
        return pos
    loc = -np.ones(len(s), np.int64); loc[pos] = np.arange(len(pos))
    c = sub["conf"]
    c = c[(loc[c[:, 0]] >= 0) & (loc[c[:, 1]] >= 0)]
    if not len(c):
        return pos
    a, b = loc[c[:, 0]], loc[c[:, 1]]
    G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(pos), len(pos)))
    ncc, lab = connected_components(G, directed=False)
    out = []
    for cc in range(ncc):
        nodes = np.flatnonzero(lab == cc)
        if len(nodes) == 1:
            out.append(pos[nodes[0]]); continue
        ln = -np.ones(len(pos), np.int64); ln[nodes] = np.arange(len(nodes))
        e = (ln[a] >= 0)
        keep = _solve_component(w[pos[nodes]], ln[a[e]], ln[b[e]], options, results)
        out.extend(pos[nodes[keep]])
    return np.asarray(out, np.int64)
