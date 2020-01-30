from capreolus.searcher import Searcher
from capreolus.utils.trec import load_qrels


def test_load_run(tmpdir):
    """ load a TREC searcher file """

    run_txt = """
             q1 Q0 d1 1 1.1 tag
             q1 Q0 d2 2 1.0 tag
             q2 Q0 d5 1 9.0 tag
    """
    run_dict = {"q1": {"d1": 1.1, "d2": 1.0}, "q2": {"d5": 9.0}}

    fn = tmpdir / "searcher"
    with open(fn, "wt", encoding="utf-8") as outf:
        outf.write(run_txt)

    run = Searcher.load_trec_run(fn)
    assert sorted(run.items()) == sorted(run_dict.items())


def test_load_qrels(tmpdir):
    """ load a TREC qrels file """

    qrels_txt = """
               q3 0 d1 1
               q3 0 d2 0
               q3 0 d3 2
               q5 0 d5 1
    """
    qrels_dict = {"q3": {"d1": 1, "d2": 0, "d3": 2}, "q5": {"d5": 1}}

    fn = tmpdir / "qrels"
    with open(fn, "wt", encoding="utf-8") as outf:
        outf.write(qrels_txt)

    qrels = load_qrels(fn)
    assert sorted(qrels.items()) == sorted(qrels_dict.items())


def test_load_norelevant_qrels(tmpdir):
    """ qids with no relevant documents should be removed when loading a qrels file """

    qrels_txt = """
               q3 0 d1 1
               q5 0 d5 0
    """
    qrels_dict = {"q3": {"d1": 1}}

    fn = tmpdir / "qrels"
    with open(fn, "wt", encoding="utf-8") as outf:
        outf.write(qrels_txt)

    qrels = load_qrels(fn)
    assert sorted(qrels.items()) == sorted(qrels_dict.items())


def test_write_run(tmpdir):
    """ write a TREC searcher file """
    fn = tmpdir / "searcher"
    run_dict = {"q1": {"d1": 1.1, "d2": 1.0}, "q2": {"d5": 9.0}}

    Searcher.write_trec_run(run_dict, fn)
    run = Searcher.load_trec_run(fn)
    assert sorted(run.items()) == sorted(run_dict.items())
