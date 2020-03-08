import numpy as np
from pathlib import Path


if __name__ == "__main__":
    infile_path = "queries.msmarcopassage.txt"

    outfile = Path("topics.msmarcopassage.txt")
    fout = open(outfile, "wt", encoding="utf-8")

    with open(infile_path, "rt", encoding="utf-8") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            fout.write("<top>\n<num> Number: {0}\n<title> {1}\n<desc> Description:\n{1}\n<narr> Narrative:\n{1}\n</top>\n".format(qid, query))


    fout.close()

