import jnius_config
from capreolus.utils.common import Anserini

jnius_config.set_classpath(Anserini.get_fat_jar())

from capreolus.pipeline import Notebook
from capreolus.task.rank import RankTask
from capreolus.task.rerank import RerankTask
