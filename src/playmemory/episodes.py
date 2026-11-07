"""Scene ("episode") segmentation of the timeline.

Consecutive frames about the same thing form an episode. Boundaries are placed
where the embedding of the next few frames stops resembling the previous few
(a TextTiling-style similarity valley), plus forced boundaries when a metadata
key (default "session") changes or a time gap is exceeded. Derived data: it is
recomputable from the vectors and cached in the file checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class EpisodeConfig:
    window: int = 3  # frames averaged on each side of a candidate boundary
    alpha: float = 0.5  # boundary needs similarity < mean - alpha*std
    min_len: int = 3
    max_len: int = 40
    boundary_key: str | None = "session"  # meta key whose change forces a boundary
    gap_seconds: float | None = None  # time gap that forces a boundary


def _window_sims(V: np.ndarray, w: int) -> np.ndarray:
    """sims[i] = cosine(mean(V[i-w:i]), mean(V[i:i+w])) for w <= i <= n-w, else +inf
    (position i means 'a boundary just before frame i')."""
    n = len(V)
    sims = np.full(n, np.inf)
    if n < 2 * w:
        return sims
    cs = np.vstack([np.zeros((1, V.shape[1]), V.dtype), np.cumsum(V, axis=0, dtype=np.float64)])
    idx = np.arange(w, n - w + 1)
    L, R = cs[idx] - cs[idx - w], cs[idx + w] - cs[idx]
    den = np.linalg.norm(L, axis=1) * np.linalg.norm(R, axis=1)
    sims[idx] = np.einsum("ij,ij->i", L, R) / np.maximum(den, 1e-12)
    return sims


def _valleys(sims: np.ndarray, w: int, thr: float) -> np.ndarray:
    """Positions that are the minimum within +-w and below thr."""
    n = len(sims)
    padded = np.concatenate([np.full(w, np.inf), sims, np.full(w, np.inf)])
    local_min = sliding_window_view(padded, 2 * w + 1).min(axis=1)
    return np.flatnonzero((sims <= local_min) & (sims < thr))


def _forced(metas: list[dict], times: list[float], cfg: EpisodeConfig) -> list[int]:
    out = []
    for i in range(1, len(metas)):
        if cfg.boundary_key and metas[i].get(cfg.boundary_key) != metas[i - 1].get(cfg.boundary_key):
            out.append(i)
        elif cfg.gap_seconds is not None and times[i] - times[i - 1] > cfg.gap_seconds:
            out.append(i)
    return out


def segment(V: np.ndarray, metas: list[dict], times: list[float], cfg: EpisodeConfig,
            thr: float | None) -> tuple[list[int], float | None]:
    """Episode start positions (always begins with 0) for the slice V.
    Returns (starts, thr); thr is computed from V when not given."""
    n = len(V)
    if n == 0:
        return [], thr
    sims = _window_sims(V, cfg.window)
    finite = sims[np.isfinite(sims)]
    if thr is None and len(finite) >= 4:
        thr = float(finite.mean() - cfg.alpha * finite.std())
    cand = {int(i): float(sims[i]) for i in (_valleys(sims, cfg.window, thr) if thr is not None else [])}
    forced = set(_forced(metas, times, cfg))
    cuts = sorted(set(cand) | forced)
    starts = [0]
    for c in cuts:  # enforce min_len (forced boundaries always win)
        if c <= 0 or c >= n:
            continue
        if c - starts[-1] < cfg.min_len and c not in forced:
            continue
        if starts[-1] != 0 and c - starts[-1] < cfg.min_len and starts[-1] not in forced and c in forced:
            starts[-1] = c  # previous cut was a weak semantic one: forced replaces it
            continue
        starts.append(c)
    # enforce max_len: split long episodes at their weakest internal valley
    out: list[int] = []
    bounds = starts + [n]
    for s, e in zip(bounds[:-1], bounds[1:]):
        out.append(s)
        while e - s > cfg.max_len:
            lo, hi = s + cfg.min_len, min(s + cfg.max_len, e - cfg.min_len)
            if hi <= lo:
                break
            window = sims[lo : hi + 1]
            cut = lo + (int(np.argmin(window)) if np.isfinite(window).any() else hi - lo)
            out.append(cut)
            s = cut
    return out, thr


class EpisodeIndex:
    """Episode boundaries over the whole timeline, maintained incrementally."""

    def __init__(self, cfg: EpisodeConfig | None = None):
        self.cfg = cfg or EpisodeConfig()
        self.starts: list[int] = []
        self.thr: float | None = None
        self._n = 0  # frames covered
        self._thr_n = 0  # frame count when thr was computed

    def __len__(self) -> int:
        return len(self.starts)

    def sync(self, V: np.ndarray, metas: list[dict], times: list[float]) -> None:
        n = len(V)
        if n == self._n and self.starts:
            return
        if n == 0:
            self.starts, self._n = [], 0
            return
        full = (not self.starts) or self.thr is None or n >= 2 * max(self._thr_n, 1) or n < self._n
        if full:
            self.starts, self.thr = segment(V, metas, times, self.cfg, None)
            self._thr_n = n
        else:
            # re-segment only from the last two episodes onward, thr held fixed
            r = self.starts[max(0, len(self.starts) - 2)]
            sub, _ = segment(V[r:], metas[r:], times[r:], self.cfg, self.thr)
            self.starts = [s for s in self.starts if s < r] + [r + s for s in sub]
        self._n = n

    def bounds(self) -> list[tuple[int, int]]:
        ends = self.starts[1:] + [self._n]
        return list(zip(self.starts, ends))

    def episode_of(self, fids) -> np.ndarray:
        return np.searchsorted(np.asarray(self.starts), np.asarray(fids), side="right") - 1

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "ep_starts": np.asarray(self.starts, dtype=np.int64),
            "ep_meta": np.asarray([np.nan if self.thr is None else self.thr, self._thr_n, self._n]),
        }

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray], cfg: EpisodeConfig | None = None) -> "EpisodeIndex":
        self = cls(cfg)
        self.starts = a["ep_starts"].tolist()
        thr, thr_n, n = a["ep_meta"].tolist()
        self.thr = None if thr != thr else thr
        self._thr_n, self._n = int(thr_n), int(n)
        return self
