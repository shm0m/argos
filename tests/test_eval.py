from argos.eval.refusal import refusal_rate


def test_refusal_rate_detects_blacklisted_phrases():
    generations = ["I cannot help with that.", "Sure, here is the answer: 42."]
    assert refusal_rate(generations) == 0.5


def test_refusal_rate_empty_list_is_zero():
    assert refusal_rate([]) == 0.0
