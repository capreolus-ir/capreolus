import capreolus.evaluator as evaluator


def test_interpolate_runs():
    run1 = {1: {"d1": 1, "d2": 2}, 2: {"d1": 1, "d2": 2}}
    run2 = {1: {"d1": 2, "d2": 1}, 2: {"d1": 1, "d2": 2}}

    qids = run1.keys()
    assert evaluator.interpolate_runs(run1, run2, qids, 0.5) == {1: {"d1": 0.5, "d2": 0.5}, 2: {"d1": 0.0, "d2": 1.0}}
    assert evaluator.interpolate_runs(run1, run2, qids, 0.2) == {1: {"d1": 0.8, "d2": 0.2}, 2: {"d1": 0.0, "d2": 1.0}}
