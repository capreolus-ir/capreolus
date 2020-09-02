import csv
import os
from os.path import exists, join

import json

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
    server = open(PACKAGE_PATH / "data" / "ambiversenlu" / "server", 'r').read().replace("\n", "")  # TODO set the ambiverseNLU server here
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
        pipeline = "ENTITY_CONCEPT_JOINT_LINKING" #"ENTITY_CONCEPT_SEPARATE_LINKING", "ENTITY_CONCEPT_SALIENCE_STANFORD", "ENTITY_CONCEPT_SALIENCE" "ENTITY_CONCEPT_SPOTTING_SEPARATE_DISAMBIGUATION" "ENTITY_CONCEPT_SPOTTING_JOINT_DISAMBIGUATION"
        typerestriction = False #if true we restrict movies, books, travel, food named entities

    def get_extracted_entities_cache_path(self):
        # logger.debug(f"entities cache path: {self.get_cache_path()}")
        return self.get_cache_path() / 'entities'

    def get_benchmark_domain(self):
        return self['benchmark'].domain

    def get_benchmark_querytype(self):
        return self['benchmark'].query_type

    def get_benchmark_name(self):
        return self['benchmark'].name

    def get_benchmark_cache_dir(self):
        return self['benchmark'].get_cache_path()

    @property
    def pipeline(self):
        return self.cfg["pipeline"]

    def extract_entities(self, textid, text):
        if self.get_benchmark_querytype() == 'entityprofile':
            raise ValueError("wrong usage of incorporate entities. Do not use it with querytype 'entityprofile'")

        outdir = self.get_extracted_entities_cache_path()
        if exists(join(outdir, get_file_name(textid, self.get_benchmark_name(), self.get_benchmark_querytype()))):
            entities = self.get_all_entities(textid)
            for e in entities["NE"]:
                self.entity_descriptions[e] = ""
            for e in entities["C"]:
                self.entity_descriptions[e] = ""
            return

        os.makedirs(outdir, exist_ok=True)


        ## manually strip the annotations grep "\[ " and  grep " \]"
        ## remove the overlapping annotations before here. check_overlapping_annotations -> data_prepration - clean profiles
        annotationsNE = []
        annotationsC = []
        annotationsEither = []
        if self.pipeline in ["ENTITY_CONCEPT_JOINT_LINKING", "ENTITY_CONCEPT_SEPARATE_LINKING"]:
            openbracket = 0
            offset = None
            tag = None
            for i in range(0, len(text)):
                ch = text[i]
                if ch == '[':
                    openbracket+=1
                    if openbracket == 2:
                        tag = "NE"
                    elif openbracket == 3:
                        tag = "C"
                    elif openbracket == 4:
                        tag = "E"
                elif ch == ']':
                    openbracket -= 1
                    if tag == "NE":
#                        logger.debug(f"annotationsNE {textid}: {text[offset:i]}")
                        annotationsNE.append({"charLength": i-offset, "charOffset": offset})
                    elif tag == "C":
#                        logger.debug(f"annotationsC {textid}: {text[offset:i]}")
                        annotationsC.append({"charLength": i - offset, "charOffset": offset})
                    elif tag == "E":
                        logger.debug(f"annotationsEither {textid}: {text[offset:i]}")
                        annotationsEither.append({"charLength": i - offset, "charOffset": offset})
                    offset = None
                    tag = None
                else:
                    if tag is None:
                        continue
                    if offset == None:
                        offset = i
            
#            if len(annotationsC) > 0:
#                logger.debug(f"annotationsC {textid}: {annotationsC}")
#            if len(annotationsNE) > 0:
#                logger.debug(f"annotationsNE {textid}: {annotationsNE}")
#            if len(annotationsEither) > 0:
#                logger.debug(f"annotationsEither {textid}: {annotationsEither}")
            if self.pipeline == "ENTITY_CONCEPT_JOINT_LINKING":
                for e in annotationsEither:
                    annotationsNE.append(e)
                    annotationsC.append(e)
            elif self.pipeline == "ENTITY_CONCEPT_SEPARATE_LINKING":
                for e in annotationsEither:#we will get 2 results (probably). Then we will add both to the entities. if they were different.
                    annotationsNE.append(e)
                    annotationsC.append(e)

        else:
            text = text.replace("[", "")
            text = text.replace("]", "")

        headers = {'accept': 'application/json', 'content-type': 'application/json'}
        data = {"docId": "{}".format(get_file_name(textid, self.get_benchmark_name(), self.get_benchmark_querytype())),
                "text": "{}".format(text),
                "extractConcepts": "{}".format(str(self.cfg["extractConcepts"])),
                "language": "en",
                "pipeline": self.pipeline,
                "annotatedMentionsNE": annotationsNE,
                "annotatedMentionsC": annotationsC
                }

        try:
            r = requests.post(url=self.server, data=json.dumps(data), headers=headers)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(e)

        # TODO: later maybe I could use the annotatedMentions to annotate the input?
        # logger.debug(f"entitylinking id:{textid} {benchmark_name} {benchmark_querytype}  status:{r.status_code}")
        if r.status_code == 200:
            with open(join(outdir, get_file_name(textid, self.get_benchmark_name(), self.get_benchmark_querytype())), 'w') as f:
                f.write(json.dumps(r.json(), sort_keys=True, indent=4))

            if 'entities' in r.json():
                for e in r.json()['entities']:
                    self.entity_descriptions[e['name']] = "" #set of entities
        else:
            raise RuntimeError(f"request status_code is {r.status_code}")

    def load_descriptions(self):
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
        data = json.load(open(join(self.get_extracted_entities_cache_path(), get_file_name(textid, self.get_benchmark_name(), self.get_benchmark_querytype())), 'r'))

        # all_entities = set()
        named_entities = set()
        concepts = set()

        if 'entities' in data:
            # all_entities.update([e['name'] for e in data['entities']])
            named_entities.update([e['name'] for e in data['entities'] if e['type'] != 'CONCEPT'])
            concepts.update([e['name'] for e in data['entities'] if e['type'] == 'CONCEPT'])

        return {"NE": list(named_entities), "C": list(concepts)}