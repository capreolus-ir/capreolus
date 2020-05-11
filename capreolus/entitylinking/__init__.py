import csv
import os
from os.path import exists, join

import json
import re

from capreolus.utils.common import get_file_name
from capreolus.utils.loginit import get_logger

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, PACKAGE_PATH
import requests

logger = get_logger(__name__)

class EntityLinking(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "entitylinking"

class AmbiverseNLU(EntityLinking):
    name = 'ambiversenlu'
    server = open(PACKAGE_PATH / "data" / "ambiversenlu" / "server", 'r').read()  # TODO set the ambiverseNLU server here
    yagodescription_dir = '/GW/D5data-11/ghazaleh/search_ranking_data/yago_description_20180120/'
    #PACKAGE_PATH / 'data' / 'yago_descriptions' #TODO set YAGO description path

    dependencies = {
        "benchmark": Dependency(module="benchmark"),
    }

    entity_descriptions = {}

    @staticmethod
    def config():
        extractConcepts = True ## TODO: let's get the pipeline as input (later when I implemented that part).
        descriptions = "YAGO_long_short"

    def get_extracted_entities_cache_path(self):
        return self.get_cache_path() / 'entities'

    def extract_entities(self, textid, text):
        if self['benchmark'].entity_strategy is None:
            return

        benchmark_name = self['benchmark'].name
        benchmark_querytype = self['benchmark'].query_type

        logger.debug("extracting entities from queries(user profiles)")
        outdir = self.get_extracted_entities_cache_path()
        if exists(join(outdir, get_file_name(textid, benchmark_name, benchmark_querytype))):
            for e in self.get_all_entities(textid):
                self.entity_descriptions[e] = ""
            return

        os.makedirs(outdir, exist_ok=True)

        headers = {'accept': 'application/json', 'content-type': 'application/json'}
        data = {"docId": "{}".format(get_file_name(textid, benchmark_name, benchmark_querytype)), "text": "{}".format(text), "extractConcepts": "{}".format(str(self.cfg["extractConcepts"])), "language": "en"}#"annotatedMentions": [{"charLength": 7, "charOffset":5}, {"charLength": 4, "charOffset": 0}]
        r = requests.post(url=self.server, data=json.dumps(data), headers=headers)
        #TODO: later maybe I could use the annotatedMentions to annotate the input??? since in the profile I know what's NE/C mainly????????

        with open(join(outdir, get_file_name(textid, benchmark_name, benchmark_querytype)), 'w') as f:
            f.write(json.dumps(r.json(), sort_keys=True, indent=4))

        if 'entities' in r.json():
            for e in r.json()['entities']:
                self.entity_descriptions[e['name']] = ""

    def load_descriptions(self):
        if self['benchmark'].entity_strategy is None:
            return

        logger.debug("loading entity descriptions")
        if self.cfg["descriptions"] == "YAGO_long_short":
            self.load_YAGOdescriptions()
        else:
            raise NotImplementedError("only have YAGO's long and short descriptions implemented")

    def load_YAGOdescriptions(self):
        with open(join(self.yagodescription_dir, 'wikipediaEntityDescriptions_en.tsv'), "r") as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            next(reader)
            for line in reader:
                if len(line) < 2:
                    continue
                entity = line[1].strip().replace("_", " ")
                entity = entity[1:-1]
                if entity not in self.entity_descriptions.keys():
                    continue
                des = line[3].strip()
                self.entity_descriptions[entity] += des + '\n'

        with open(join(self.yagodescription_dir, 'wikidataEntityDescriptions.tsv'), "r") as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            next(reader)
            for line in reader:
                if len(line) < 2:
                    continue
                entity = line[1].strip().replace("_", " ")
                entity = entity[1:-1]
                if entity not in self.entity_descriptions.keys():
                    continue
                des = line[3].strip()
                lang = des[des.find('@') + 1:len(des)]
                des = des[0:des.find('@')]
                if lang == 'en':
                    self.entity_descriptions[entity] += des + '\n'

    def get_entity_description(self, entity):
        return self.entity_descriptions[entity]

    def get_all_entities(self, textid):
        data = json.load(open(join(self.get_extracted_entities_cache_path(), get_file_name(textid, self['benchmark'].name, self['benchmark'].query_type)), 'r'))
        res = []
        if 'entities' in data:
            for e in data['entities']:
                res.append(e['name'])
        return res

    def get_domain_related_entieis(self, textid):
        domain = self['benchmark'].domain
        #TODO     Implement - a new dependency - component

