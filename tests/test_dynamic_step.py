import pytest

from replace.dynamic_step import COF_RANGE, MIN_COF_MAX, DynamicStepSize, TransitionPoints, find_transition_points


def test_second_order_transition_points_are_the_knees():
    # Flat until 80, linear rise until 200, flat until 300.
    hpwl = [0.0] * 80 + [(t - 80) / 120 for t in range(80, 200)] + [1.0] * 101
    tps = find_transition_points(hpwl)
    start, _, tp2a, _, tp2b, _, end = tps.index
    assert (start, tp2a, tp2b, end) == (0, 80, 200, 300)


def test_first_order_transition_points_lie_inside_their_phases():
    # S-shaped curve: HPWL rises, flattens, then rises again (cf. Fig. 5).
    hpwl = [min(t, 60) + max(t - 200, 0) * 0.8 + 0.02 * t for t in range(301)]
    idx = find_transition_points(hpwl).index
    assert idx == sorted(idx)
    for phase in range(3):
        assert idx[2 * phase] <= idx[2 * phase + 1] <= idx[2 * phase + 2]


def test_cof_max_is_smallest_at_tp2_and_largest_at_tp1():
    # Anchors at iterations 0..6 with HPWL 0,1,2,... and decreasing potential.
    tps = TransitionPoints(list(range(7)))
    hpwl = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    potential = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0]
    ds = DynamicStepSize(tps, hpwl, potential)
    # Start of phase 1 (a phase boundary): minimum of the phase-1 range.
    assert ds.cof_max(0.0, 99.0) == pytest.approx(MIN_COF_MAX[0])
    # Past phase 2's TP1 (potential 70), halfway back toward TP2 at HPWL 4.
    assert ds.cof_max(3.5, 65.0) == pytest.approx(MIN_COF_MAX[1] + 0.5 * COF_RANGE[1])
    # At phase 3's TP1.
    assert ds.cof_max(5.0, 55.0) == pytest.approx(MIN_COF_MAX[2] + COF_RANGE[2])
    # Beyond the trial's end, and the phase never goes back.
    assert ds.cof_max(7.0, 30.0) == pytest.approx(MIN_COF_MAX[2])
    assert ds.cof_max(0.0, 99.0) == pytest.approx(MIN_COF_MAX[2])
