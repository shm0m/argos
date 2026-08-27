from argos.detect import classify


def test_classify_flags_near_zero_projection_as_ablated():
    baseline = {"mean_projection": 4.2}
    candidate = {"mean_projection": 0.05}
    ratio, flagged = classify(baseline, candidate)
    assert flagged
    assert ratio < 0.2


def test_classify_does_not_flag_similar_projection():
    baseline = {"mean_projection": 4.2}
    candidate = {"mean_projection": 3.9}
    _ratio, flagged = classify(baseline, candidate)
    assert not flagged
