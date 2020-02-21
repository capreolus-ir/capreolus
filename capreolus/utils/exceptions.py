class MissingDocError(Exception):
    def __init__(self, qid, docid):
        self.related_qid = qid
        self.missed_docid = docid
