import pytest

from capreolus import Benchmark, Task, module_registry
from capreolus.tests.common_fixtures import tmpdir_as_cache, dummy_index


tasks = set(module_registry.get_module_names("task"))


@pytest.mark.parametrize("task_name", tasks)
def test_task_creatable(tmpdir_as_cache, dummy_index, task_name):
    provide = {"index": dummy_index, "benchmark": Benchmark.create("dummy"), "collection": dummy_index.collection}
    task = Task.create(task_name, provide=provide)
