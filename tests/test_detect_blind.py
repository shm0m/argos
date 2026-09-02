from argos.detect_blind import find_anomalous_layers


def _profile(norms):
    return {("resid_pre", i): n for i, n in enumerate(norms)}


def test_flags_sustained_drop_after_a_point():
    reference = _profile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    candidate = _profile([1.0, 2.0, 3.0, 1.6, 2.0, 2.4, 2.8])

    anomalies = find_anomalous_layers(reference, candidate)

    flagged_layers = {a["layer"] for a in anomalies}
    assert 4 in flagged_layers
    assert 0 not in flagged_layers
    assert 1 not in flagged_layers


def test_no_anomaly_on_identical_profiles():
    reference = _profile([1.0, 2.0, 3.0, 4.0, 5.0])
    candidate = _profile([1.0, 2.0, 3.0, 4.0, 5.0])

    assert find_anomalous_layers(reference, candidate) == []


def test_uniform_scaling_is_not_flagged():
    reference = _profile([1.0, 2.0, 3.0, 4.0, 5.0])
    candidate = _profile([0.5, 1.0, 1.5, 2.0, 2.5])

    assert find_anomalous_layers(reference, candidate) == []


def test_low_reference_norm_layers_are_ignored():
    reference = _profile([0.001, 2.0, 3.0])
    candidate = _profile([0.0001, 2.0, 3.0])

    assert find_anomalous_layers(reference, candidate) == []
