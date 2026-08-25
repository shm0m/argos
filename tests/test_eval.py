from argos.eval.refusal import degenerate_rate, is_degenerate, refusal_rate, refusal_stats, wilson_ci


def test_refusal_rate_detects_blacklisted_phrases():
    generations = ["I cannot help with that.", "Sure, here is the answer: 42."]
    assert refusal_rate(generations) == 0.5


def test_refusal_rate_empty_list_is_zero():
    assert refusal_rate([]) == 0.0


def test_wilson_ci_widens_with_smaller_n():
    low_small, high_small = wilson_ci(1, 4)
    low_large, high_large = wilson_ci(25, 100)
    assert (high_small - low_small) > (high_large - low_large)


def test_refusal_stats_reports_count_and_ci():
    stats = refusal_stats(["I cannot help.", "Sure, 42.", "I won't do that."])
    assert stats["n"] == 3
    assert stats["refused"] == 2
    assert stats["ci95"][0] <= stats["rate"] <= stats["ci95"][1]


def test_empty_generation_is_degenerate_not_a_refusal():
    assert is_degenerate("")
    assert refusal_rate([""]) == 0.0


def test_degenerate_rate_flags_mostly_empty_output():
    assert degenerate_rate(["", "", "a real answer here"]) > 0.5
    assert degenerate_rate(["a real answer", "another real one"]) == 0.0
