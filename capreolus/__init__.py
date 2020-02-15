import jnius_config
from capreolus.utils.common import Anserini

jnius_config.set_classpath(Anserini.get_fat_jar())
