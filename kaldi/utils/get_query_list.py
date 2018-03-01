# Shane Settle, settle.shane@gmail.com, 2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --

import sys
import re
from argparse import ArgumentParser


def read_format(x):
    """Read line format of words file."""

    segment, word = re.split(" ", x)
    convside, interval = re.split("_", segment)

    return word.lower(), convside, re.split("-", interval)


def get_query_list(words, convsides, is_valid, min_count=1):
    """
    Get the list of queries (derived from words)
    that occur in these conversation sides
    """

    occurrences = dict()

    for word, convside, (start, stop) in map(read_format, words):

        if is_valid(word, start, stop):
            if convside in convsides:
                if word not in occurrences:
                    occurrences[word] = []
                occurrences[word].append("%s_%s_%s-%s" % (word, convside, start, stop))

    query_list = []
    for word, segments in occurrences.items():

        if len(segments) >= min_count:
            query_list += segments

    return sorted(query_list)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--words')  # list of words and corresponding switchboard location
    parser.add_argument('--convsides')  # conversation sides
    parser.add_argument('--min-word-length', type=int)  # minimum number of characters for words to be included
    parser.add_argument('--min-audio-duration', type=int)  # in 10ms frames s.t. 50frames = 50framesx10ms = 0.5s
    parser.add_argument('--min-occurrence-count', type=int)  # min required word occurrences in partition
    parser.add_argument('--query-list-file')  # name of query list file
    args = parser.parse_args()

    with open(args.words, "r") as f:
        words = f.read().splitlines()

    with open(args.convsides, "r") as f:
        convsides = f.read().splitlines()

    invalid_words = [
        "[noise]",
        "[vocalized-noise]",
        "[laughter]",
        "uh-hum",
        "uh-huh",
        "um-hum"
    ]

    sys.stdout.write("Ignoring words:\n" + "\n".join(invalid_words) + "\n")

    def is_valid(word, start, stop):
        if any([c.isdigit() for c in word]):  # includes numeric digits
            return False
        elif word[0] == "-" or word[-1] == "-":  # includes partial word
            return False
        elif word in invalid_words:  # included in set of invalid words
            return False
        elif len(word) < args.min_word_length:  # word is too short
            return False
        elif int(stop) - int(start) + 1 < args.min_audio_duration:  # audio is too short
            return False
        else:
            return True

    sys.stdout.write("Writing query lists...\n")

    with open(args.query_list_file, "w") as f:
        query_list = get_query_list(words, convsides, is_valid, args.min_occurrence_count)
        f.write("\n".join(query_list))

    sys.stdout.write("Done.\n")
