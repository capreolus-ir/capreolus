import csv
import os
from os.path import exists, join

import json
import re

from capreolus.utils.loginit import get_logger

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, PACKAGE_PATH
import requests

logger = get_logger(__name__)

class EntityLinking(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "entitylinking"

class AmbiverseNLU(EntityLinking):
    name = 'ambiversenlu'
    server = open(PACKAGE_PATH / "data" / "amvibersenlu" / "server", 'r').read()  # TODO set the ambiverseNLU server here
    yagodescription_dir = '/GW/D5data-11/ghazaleh/search_ranking_data/yago_description_20180120/'
    #PACKAGE_PATH / 'data' / 'yago_descriptions' #TODO set YAGO description path

    dependencies = {"benchmark": Dependency(module="benchmark"),
                    "domainrelatedness": Dependency(module='entityutils', name='relatednesswiki2vec', config_overrides={"strategy": "domain-vector-100"})}

    entity_descriptions = {}

    @staticmethod
    def config():
        extractConcepts = True ## TODO: let's get the pipeline as input (later when I implemented that part).
        descriptions = "YAGO_long_short"

    def get_extracted_path(self):
        return self.get_cache_path() / 'entities'

    def extract_entities(self, textid, text):
        if self['benchmark'].entity_strategy == 'none':
            return

        logger.debug("extracting entities from queries(user profiles)")
        outdir = self.get_extracted_path()
        if exists(join(outdir, self.get_file_name(textid))):
            for e in self.get_all_entities(textid):
                self.entity_descriptions[e] = ""
            return

        os.makedirs(outdir, exist_ok=True)

        headers = {'accept': 'application/json', 'content-type': 'application/json'}
        data = {"docId": "{}".format(self.get_file_name(textid)), "text": "{}".format(text), "extractConcepts": "{}".format(str(self.cfg["extractConcepts"])), "language": "en"}#"annotatedMentions": [{"charLength": 7, "charOffset":5}, {"charLength": 4, "charOffset": 0}]
        r = requests.post(url=self.server, data=json.dumps(data), headers=headers)
        #TODO: later maybe I could use the annotatedMentions to annotate the input??? since in the profile I know what's NE/C mainly????????

        with open(join(outdir, self.get_file_name(textid)), 'w') as f:
            f.write(json.dumps(r.json(), sort_keys=True, indent=4))

        if 'entities' in r.json():
            for e in r.json()['entities']:
                self.entity_descriptions[e['name']] = ""

    def load_descriptions(self):
        if self['benchmark'].entity_strategy == 'none':
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

    def get_entities(self, profile_id):
        entity_strategy = self['benchmark'].entity_strategy
        
        if entity_strategy == 'none':
            return []
        elif entity_strategy == 'all':
            return self.get_all_entities(profile_id)
        elif entity_strategy == 'domain':
            logger.debug(f"GET ENTITIES {entity_strategy}")
            domain = self['benchmark'].domain 
            logger.debug(f"GET ENTITIES {domain}")
            self["domainrelatedness"].initialize(domain)
            return self["domainrelatedness"].get_domain_related_entities(self.get_all_entities(profile_id))
        else:
            raise NotImplementedError("TODO implement other entity strategies (by first implementing measures)")

    def get_all_entities(self, textid):
        data = json.load(open(join(self.get_extracted_path(), self.get_file_name(textid)), 'r'))
        res = []
        if 'entities' in data:
            for e in data['entities']:
                res.append(e['name'])
        return res

    def get_domain_related_entieis(self, textid):
        domain = self['benchmark'].domain
        #TODO     Implement - a new dependency - component

    def get_file_name(self, id):
        benchmark = self['benchmark']
        ### This is written wrt our benchmarks and the ids we have for the queries.
        ### Maybe need to be extended on new benchmarks.
        ## The idea is that, we don't want to have redundency in the extraction and caching

        if benchmark.name in ['pes20', 'kitt']:
            if benchmark.query_type == "query":
                return re.sub(r'(\d+)_(.+)', r'\g<1>', id)
            else:
                return re.sub(r'(\d+)_(.+)', r'\g<2>', id)
        else:
            return id
