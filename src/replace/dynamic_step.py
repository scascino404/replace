"""RePlAce dynamic step size adaptation (-ds), Section IV of the paper.

A trial global placement records the HPWL curve. Its shape is split into three
phases by two 2nd-order transition points (TP2, where the slope changes most),
and each phase holds one 1st-order transition point (TP1, Algorithm 3). The
actual placement then bounds the per-iteration growth of the density penalty
lambda by cof_max (Eq. 10):

    cof_max = min_cof_max + cof_range * |HPWL - HPWL_TP2| / |HPWL_TP1 - HPWL_TP2|

so steps are smallest at the TP2s, where solution quality is decided, and
largest at the TP1s.

Details the paper leaves open follow the original RePlAce code: the phase
boundaries (trial start, TP2s, trial end) all act as "TP2" anchors; the TP2
in Eq. 10 is the boundary on the current side of the phase's TP1; the ratio
is capped at 1; progress along the curve is tracked through the density
potential, which keeps decreasing even where HPWL is flat; and after the
trial's end point cof_max stays at the last phase's minimum.
"""

from dataclasses import dataclass

import torch

# Per phase (paper, Section IV-A).
MIN_COF_MAX = (1.0001, 1.001, 1.005)
COF_RANGE = (0.0009, 0.024, 0.045)


def _chord_gap(hpwl: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """HPWL minus the straight line from point a to point b, on [a, b]."""
    t = torch.linspace(0, 1, b - a + 1, dtype=hpwl.dtype)
    return hpwl[a : b + 1] - (hpwl[a] + t * (hpwl[b] - hpwl[a]))


def _farthest(hpwl: torch.Tensor, a: int, b: int) -> int:
    """Point of [a, b] farthest from the chord between a and b."""
    return a + int(_chord_gap(hpwl, a, b).abs().argmax())


@dataclass
class TransitionPoints:
    """Trial iteration indices of the phase anchors:
    [start, TP1, TP2, TP1, TP2, TP1, end], i.e. phase p spans
    anchors 2p .. 2p + 2 with its TP1 at 2p + 1."""

    index: list[int]


def find_transition_points(hpwl: list[float]) -> TransitionPoints:
    """Algorithm 3 on the trial HPWL curve."""
    h = torch.tensor(hpwl, dtype=torch.float64)
    end = len(h) - 1
    # TP2s: farthest point from the primary line (start -> end), then the
    # farthest point from the two secondary lines through the first TP2.
    tp2a = _farthest(h, 0, end)
    gaps = torch.cat([_chord_gap(h, 0, tp2a)[:-1], _chord_gap(h, tp2a, end)]).abs()
    tp2b = int(gaps.argmax())
    tp2a, tp2b = sorted((tp2a, tp2b))
    bounds = [0, tp2a, tp2b, end]

    anchors = [0]
    for phase in range(3):
        a, b = bounds[phase], bounds[phase + 1]
        if b - a < 2:
            tp1 = (a + b) // 2  # degenerate phase
        elif phase == 1:
            # 2nd phase: where the curve passes from above to below the phase line.
            gap = _chord_gap(h, a, b)
            cross = ((gap[:-1] >= 0) & (gap[1:] < 0)).nonzero()
            tp1 = a + 1 + int(cross[0]) if len(cross) else _farthest(h, a, b)
        else:
            # 1st/3rd phase: where the slope equals that of the phase line,
            # i.e. the point farthest from it.
            tp1 = _farthest(h, a, b)
        anchors += [tp1, b]
    return TransitionPoints(anchors)


class DynamicStepSize:
    """cof_max schedule (Eq. 10) for the actual placement."""

    def __init__(self, tps: TransitionPoints, trial_hpwl: list[float], trial_potential: list[float]):
        self.hpwl = [trial_hpwl[i] for i in tps.index]
        self.potential = [trial_potential[i] for i in tps.index]
        self.segment = 0  # current half-phase: between anchors segment and segment + 1

    def _advance(self, potential: float) -> None:
        """Move to the next half-phase once the potential passes its end anchor."""
        while self.segment < len(self.potential) - 1:
            start, stop = self.potential[self.segment], self.potential[self.segment + 1]
            passed = potential <= stop if stop <= start else potential >= stop
            if not passed:
                break
            self.segment += 1

    def cof_max(self, hpwl: float, potential: float) -> float:
        self._advance(potential)
        k = self.segment
        if k >= 6:  # beyond the trial's end point
            return MIN_COF_MAX[2]
        phase = k // 2
        tp1 = self.hpwl[2 * phase + 1]
        tp2 = self.hpwl[k] if k % 2 == 0 else self.hpwl[k + 1]
        ratio = abs(hpwl - tp2) / max(abs(tp1 - tp2), 1e-12)
        return MIN_COF_MAX[phase] + COF_RANGE[phase] * min(ratio, 1.0)
