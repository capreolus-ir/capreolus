from django.apps import AppConfig

from capreolus.demo_app.views import ConfigsView
from capreolus.extractor.embedding import EmbeddingHolder


class DemoAppConfig(AppConfig):
    name = "demo_app"

    def ready(self):
        configs = ConfigsView.get_config_from_results()
        for config in configs:
            EmbeddingHolder.get_instance(config.get("embeddings", "glove6b"))
