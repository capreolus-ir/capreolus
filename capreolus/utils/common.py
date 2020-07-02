import hashlib
import logging
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import requests
import spacy
from tqdm import tqdm

from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Anserini:
    @classmethod
    def get_fat_jar(cls):
        # Go through sys.path hoping to find the pyserini install dir
        for path in sys.path:
            jar_path = "{0}/pyserini/resources/jars/".format(path)
            if os.path.exists(jar_path):
                fat_jar_path = glob(os.path.join(jar_path, "anserini-0.9*-fatjar.jar"))
                if fat_jar_path:
                    return max(fat_jar_path, key=os.path.getctime)

        raise Exception("could not find anserini fat jar")

    @classmethod
    def filter_and_log_anserini_output(cls, line, logger):
        """ Ignore DEBUG lines and require other lines pass our logging level """
        fields = line.strip().split()

        # is this a log line?
        # at least 5 fields should exist
        # (0) date field should be 10 digits and begin with 20. e.g. 2020-02-14
        # (3) function field should begin with [
        if len(fields) > 5 and len(fields[0]) == 10 and fields[3][0] == "[":
            # skip debug messages
            if fields[2] == "DEBUG":
                msg = None
            else:
                loglevel = logging._nameToLevel.get(fields[2], 40)
                msg = " ".join(fields[3:])
        else:
            loglevel = logging._nameToLevel["WARNING"]
            msg = line.strip()

        if msg:
            logger.log(loglevel, "[AnseriniProcess] %s", msg)


def download_file(url, outfn, expected_hash=None):
    """ Download url to the file outfn. If expected_hash is provided, use it to both verify the file was downloaded
        correctly, and to avoid re-downloading an existing file with a matching hash.
    """

    if expected_hash and os.path.exists(outfn):
        found_hash = hash_file(outfn)

        if found_hash == expected_hash:
            return

    head = requests.head(url)
    size = int(head.headers.get("content-length", 0))

    with open(outfn, "wb") as outf:
        r = requests.get(url, stream=True)
        with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"downloading {url}", miniters=1) as pbar:
            for chunk in r.iter_content(32 * 1024):
                outf.write(chunk)
                pbar.update(len(chunk))

    if not expected_hash:
        return

    found_hash = hash_file(outfn)
    if found_hash != expected_hash:
        raise IOError(f"expected file {outfn} downloaded from {url} to have SHA256 hash {expected_hash} but got {found_hash}")


def hash_file(fn):
    """ Compute a SHA-256 hash for the file fn and return a hexdigest of the hash """
    sha = hashlib.sha256()

    with open(fn, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha.update(data)

    return sha.hexdigest()


def padlist(list_to_pad, padlen, pad_token=0):
    """
    Pads a list with pad_token
    Args:
        list_to_pad: a list of str / int / float, each entry represents the token / token id
        padlen: int, max length of the expected list, list_to_pad would either be truncated or padded to this length
        pad_token: str / int, representing the padding token / token id

    Returns: padded list
    """
    padded_list = list_to_pad[:padlen]
    padded_list = padded_list + [pad_token] * (padlen - len(list_to_pad))
    return padded_list


def plot_loss(history, outfn, interactive=False):
    # epochs, losses = zip(*history)
    losses = history
    epochs = list(range(0, len(history)))
    best_epoch = epochs[np.argmin(losses)]
    fig = plt.figure()
    plt.plot(epochs, losses, "k-.")
    plt.ylabel("Training Loss")
    plt.tick_params("y")
    plt.xlabel("Iteration")
    plt.title("min loss: %d %.3f" % (best_epoch, losses[best_epoch]))
    fig.savefig(outfn, format="pdf")
    if interactive:
        plt.show(block=False)
    plt.close()


def plot_metrics(metrics, outfn, interactive=False, show={"map", "P_20", "ndcg_cut_20"}):
    title = "maxs: "
    fig = plt.figure()
    for metric, xys in metrics.items():
        if metric not in show:
            continue
        # plt.plot(*zip(*xys), label=metric)
        epochs = list(range(1, len(xys) + 1))
        plt.plot(epochs, xys, label=metric)
        # max_iter, max_metric = max(xys, key=lambda x: x[1])
        max_metric = max(xys)
        max_iter = xys.index(max_metric) + 1
        title += f"{metric} {max_metric:0.3f} ({max_iter}) "

    plt.ylabel("Metric")
    plt.tick_params("y")
    plt.xlabel("Iteration")
    plt.title(title)
    plt.legend()
    fig.savefig(outfn, format="pdf")
    if interactive:
        plt.show(block=False)
    plt.close()


def remove_newline(txt):
    return txt.replace("\r", "").replace("\n", "").strip()


def get_udel_query_expander():
    stopwords = {
        "a",
        "about",
        "above",
        "according",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "albeit",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "an",
        "and",
        "another",
        "any",
        "anybody",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "apart",
        "are",
        "around",
        "as",
        "at",
        "av",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "canst",
        "certain",
        "cf",
        "choose",
        "contrariwise",
        "cos",
        "could",
        "cu",
        "day",
        "do",
        "does",
        "doesn",
        "t",
        "doing",
        "dost",
        "doth",
        "double",
        "down",
        "dual",
        "during",
        "each",
        "either",
        "else",
        "elsewhere",
        "enough",
        "et",
        "etc",
        "even",
        "ever",
        "every",
        "everybody",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "excepted",
        "excepting",
        "exception",
        "exclude",
        "excluding",
        "exclusive",
        "far",
        "farther",
        "farthest",
        "few",
        "ff",
        "first",
        "for",
        "formerly",
        "forth",
        "forward",
        "from",
        "front",
        "further",
        "furthermore",
        "furthest",
        "get",
        "go",
        "had",
        "halves",
        "hardly",
        "has",
        "hast",
        "hath",
        "have",
        "he",
        "hence",
        "henceforth",
        "her",
        "here",
        "hereabouts",
        "hereafter",
        "hereby",
        "herein",
        "hereto",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "hindmost",
        "his",
        "hither",
        "hitherto",
        "how",
        "however",
        "howsoever",
        "i",
        "ie",
        "if",
        "in",
        "inasmuch",
        "inc",
        "include",
        "included",
        "including",
        "indeed",
        "indoors",
        "inside",
        "insomuch",
        "instead",
        "into",
        "inward",
        "inwards",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "kg",
        "kind",
        "km",
        "last",
        "latter",
        "latterly",
        "less",
        "lest",
        "let",
        "like",
        "little",
        "ltd",
        "many",
        "may",
        "maybe",
        "me",
        "meantime",
        "meanwhile",
        "might",
        "more",
        "moreover",
        "most",
        "mostly",
        "mr",
        "mrs",
        "ms",
        "much",
        "must",
        "my",
        "myself",
        "namely",
        "need",
        "neither",
        "never",
        "nevertheless",
        "next",
        "no",
        "nobody",
        "none",
        "nonetheless",
        "noone",
        "nope",
        "nor",
        "not",
        "nothing",
        "notwithstanding",
        "now",
        "nowadays",
        "nowhere",
        "of",
        "off",
        "often",
        "ok",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "outside",
        "over",
        "own",
        "per",
        "perhaps",
        "plenty",
        "provide",
        "quite",
        "rather",
        "really",
        "round",
        "said",
        "sake",
        "same",
        "sang",
        "save",
        "saw",
        "see",
        "seeing",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "seen",
        "seldom",
        "selves",
        "sent",
        "several",
        "shalt",
        "she",
        "should",
        "shown",
        "sideways",
        "since",
        "slept",
        "slew",
        "slung",
        "slunk",
        "smote",
        "so",
        "some",
        "somebody",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhat",
        "somewhere",
        "spake",
        "spat",
        "spoke",
        "spoken",
        "sprang",
        "sprung",
        "stave",
        "staves",
        "still",
        "such",
        "supposing",
        "than",
        "that",
        "the",
        "thee",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "thenceforth",
        "there",
        "thereabout",
        "thereabouts",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereof",
        "thereon",
        "thereto",
        "thereupon",
        "these",
        "they",
        "this",
        "those",
        "thou",
        "though",
        "thrice",
        "through",
        "throughout",
        "thru",
        "thus",
        "thy",
        "thyself",
        "till",
        "to",
        "together",
        "too",
        "toward",
        "towards",
        "ugh",
        "unable",
        "under",
        "underneath",
        "unless",
        "unlike",
        "until",
        "up",
        "upon",
        "upward",
        "upwards",
        "us",
        "use",
        "used",
        "using",
        "very",
        "via",
        "vs",
        "want",
        "was",
        "we",
        "week",
        "well",
        "were",
        "what",
        "whatever",
        "whatsoever",
        "when",
        "whence",
        "whenever",
        "whensoever",
        "where",
        "whereabouts",
        "whereafter",
        "whereas",
        "whereat",
        "whereby",
        "wherefore",
        "wherefrom",
        "wherein",
        "whereinto",
        "whereof",
        "whereon",
        "wheresoever",
        "whereto",
        "whereunto",
        "whereupon",
        "wherever",
        "wherewith",
        "whether",
        "whew",
        "which",
        "whichever",
        "whichsoever",
        "while",
        "whilst",
        "whither",
        "who",
        "whoa",
        "whoever",
        "whole",
        "whom",
        "whomever",
        "whomsoever",
        "whose",
        "whosoever",
        "why",
        "will",
        "wilt",
        "with",
        "within",
        "without",
        "worse",
        "worst",
        "would",
        "wow",
        "ye",
        "year",
        "yet",
        "yippee",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
    nlp = spacy.load("en_core_sci_lg")

    def expand_query(txt, rm_sw):
        if rm_sw:
            return " ".join([w.text for w in nlp(txt.strip()) if w.text not in stopwords])

        else:
            return " ".join([w.text for w in nlp(txt.strip()).ents])

    return expand_query
