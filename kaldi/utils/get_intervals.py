# Shane Settle, settle.shane@gmail.com, 2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --

from argparse import ArgumentParser
from collections import defaultdict
import sys
import re


def lookup(utterances, query_start, query_stop):
    """
    Looks for the interval in the
    segments given by a set of utterances.
    Will clamp to best interal if not entirely available.
    """

    query_len = query_stop - query_start + 1

    for utt in utterances:

        utt_start, utt_stop = map(int, re.split("-", utt))
        utt_len = utt_stop - utt_start + 1

        overlap_len = min(query_stop, utt_stop) - max(query_start, utt_start) + 1
        overlap_percentage = float(overlap_len) / query_len

        if overlap_percentage > 0.9:
            sys.stdout.write("overlap percentage: %.2f.\n" % overlap_percentage)
            utt_query_range = "%s %s" % (max(0, query_start - utt_start), min(utt_len - 1, query_stop - utt_start))
            return utt, utt_query_range

    return None, None


def get_query_intervals(query_list, scp_list):
    """
    Reads from query list file (query_list_file) and
    a feature pointer file (scp_file). Writes to an intervals
    file which summarizes the necessary information into
    one place for training.
    """

    convsides = defaultdict(list)
    for segment, _ in map(lambda x: re.split(r"\s", x), scp_list):  # split lines on whitespace
        convside, interval = re.split("_", segment)
        convsides[convside].append(interval)

    query_intervals = []
    for query_line in query_list:  # format: <query>_<convside>_<start>-<stop>

        query, convside, query_interval = re.split("_", query_line)

        utterances = convsides[convside]
        query_start, query_stop = map(int, re.split("-", query_interval))

        utt, utt_query_range = lookup(utterances, query_start, query_stop)

        if utt is not None:
            query_id = "%s_%s_%s" % (query, convside, query_interval)
            utt_id = "%s_%s" % (convside, utt)
            query_intervals.append("%s %s %s" % (query_id, utt_id, utt_query_range))

    return sorted(query_intervals)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--query-list-file')  # input list of queries
    parser.add_argument('--scp-file')  # input feature scp file
    parser.add_argument('--intervals-file')  # output intervals file
    args = parser.parse_args()

    sys.stdout.write("Writing intervals...\n")

    with open(args.query_list_file, "r") as f:
        query_list = f.read().splitlines()

    with open(args.scp_file, "r") as f:
        scp_list = f.read().splitlines()

    with open(args.intervals_file, "w") as f:
        intervals = get_query_intervals(query_list, scp_list)
        f.write("\n".join(intervals))
    sys.stdout.write("Done.\n")
