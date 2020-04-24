import os
from capreolus.benchmark import CodeSearchNet

def test_csn_benchmark_downloadifmissing():
    for lang in ["python", "java", "javascript", "go", "ruby", "php"]:
        cfg = {"_name": "codesearchnet", "lang": lang}
        benchmark = CodeSearchNet(cfg)
        benchmark.download_if_missing()

        assert os.path.exists(benchmark.docid_map_file)
        assert os.path.exists(benchmark.qid_map_file)

        assert os.path.exists(benchmark.topic_dir / f"{lang}.txt")
        assert os.path.exists(benchmark.qrel_dir / f"{lang}.txt")
        assert os.path.exists(benchmark.fold_dir / f"{lang}.json")

