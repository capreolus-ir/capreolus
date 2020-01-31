import json
import os
import sys
import time
import subprocess
from multiprocessing import Manager, Process, Pool

from tqdm import tqdm

# Duplicated util function - we don't want to import from capreolus.utils due to jnius woes
def get_default_cache_dir():
    default_dir = os.path.expanduser("~/.capreolus/cache/")
    if not os.path.exists(default_dir):
        os.makedirs(os.path.dirname(default_dir))
    return default_dir


def crawl():
    """
    Iterates through every document in a collection and looks for the doc ids passed as command line arguments.
    Spawns multiple processes to do this for us. Clueweb12 crawl completes in approximately 42 hours with 8 processes
    See `get_documents_from_disk()` in anserini.py to know how this file is being used
    """
    rootdir = sys.argv[1]
    ctype = sys.argv[2]
    doc_ids = set(input().split(","))
    manager = Manager()
    shared_dict = manager.dict()
    multiprocess_start = time.time()
    print("Start multiprocess")
    args_list = []
    for subdir in os.listdir(rootdir):
        if os.path.isdir(rootdir + "/" + subdir):
            args_list.append({"doc_ids": doc_ids, "rootdir": rootdir + "/" + subdir, "ctype": ctype, "shared_dict": shared_dict})

    pool = Pool(processes=8)
    pool.map(spawn_child_process_to_read_docs, args_list)

    print("Getting all documents from disk took: {0}".format(time.time() - multiprocess_start))
    # TODO: This will fail if multiple crawls are running at the same time
    with open("{0}/disk_crawl_temp_dump.json".format(os.getenv("CAPREOLUS_CACHE", get_default_cache_dir())), "w") as fp:
        json.dump(shared_dict.copy(), fp)
    # return [shared_dict.get(doc_id, []) for doc_id in doc_ids]


def spawn_child_process_to_read_docs(data):
    target_doc_ids = data["doc_ids"]
    path = data["rootdir"]
    ctype = data["ctype"]
    shared_dict = data["shared_dict"]
    local_dict = {}
    start = time.time()
    from pyserini.collection import pycollection
    from pyserini.index import pygenerator

    collection = pycollection.Collection(ctype, path)
    generator = pygenerator.Generator("JsoupGenerator")
    for i, file_segment in enumerate(collection):
        doc_ids, doc_contents = read_file_segment(target_doc_ids, file_segment, generator)
        for i, doc_id in enumerate(doc_ids):
            local_dict[doc_id] = doc_contents[i]
    shared_dict.update(local_dict)
    print("PID: {0}, Done getting documents from disk: {1} for path: {2}".format(os.getpid(), time.time() - start, path))


def read_file_segment(target_doc_ids, file_segment, generator):
    doc_ids = []
    docs = []
    for j, doc in enumerate(file_segment):
        if doc.id is not None:
            parsed = generator.create_document(doc)
            if parsed is None:
                continue
            current_doc_id = parsed.get("id")
            contents = parsed.get("contents")
            if current_doc_id in target_doc_ids:
                doc_ids.append(current_doc_id)
                docs.append(contents)

    return doc_ids, docs


docs_contents = crawl()
