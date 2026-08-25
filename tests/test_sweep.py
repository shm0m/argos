from argos.sweep import pick_evenly_spaced


def test_pick_evenly_spaced_covers_full_range():
    items = list(range(26))
    picked = pick_evenly_spaced(items, 8)
    assert picked[0] == 0
    assert picked[-1] == 25
    assert len(picked) == 8
    assert picked == sorted(picked)


def test_pick_evenly_spaced_returns_all_when_n_exceeds_length():
    items = list(range(5))
    assert pick_evenly_spaced(items, 10) == items


def test_pick_evenly_spaced_single():
    items = list(range(10))
    assert pick_evenly_spaced(items, 1) == [0]
