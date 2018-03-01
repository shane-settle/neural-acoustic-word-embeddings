from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
from datetime import datetime
import sys
import re


def make_conv_side_dictionary(scp_fn):
    """
    Make dictionary of segments.
    """

    conv_sides = dict()

    with open(scp_fn, "r") as scp_file:
        lines = scp_file.read().splitlines()

    for segment, ark, pos in map(lambda line: re.split("\s|:", line), lines):
        conv_id, side, start, stop = re.split("_|-", segment)
        ark_details = (ark, pos) # -- ark filename and segment byte position

        cs_id = conv_id + side # -- conversation side id
        interval = (int(start), int(stop)) # -- endpoints of segment for which we have mfccs

        if cs_id in conv_sides:
            conv_sides[cs_id][interval] = ark_details
        else:
            conv_sides[cs_id] = {interval: ark_details}

    return conv_sides


def query_lookup(utts, q_start, q_stop):
    """
    Looks for query interval in set of segments.
    """

    best_overlap_len = 0
    q_ival = set(range(q_start, q_stop))
    q_len = len(q_ival)

    for (utt_start, utt_stop) in utts.keys():

        utt_ival = range(utt_start, utt_stop)
        utt_len = len(utt_ival)
        overlap_len = len(q_ival.intersection(utt_ival))

        if overlap_len > best_overlap_len:
            rev_q_start= max(0, q_start - utt_start)
            rev_q_stop = min(utt_len - 1, q_stop - utt_start)

            rev_q_ival = (rev_q_start, rev_q_stop)
            rev_utt_ival = (utt_start, utt_stop)

            if overlap_len == min(q_len, utt_len):
                break
            else:
                best_overlap_len = overlap_len
        else:
            continue

    return rev_q_ival, rev_utt_ival, utts[rev_utt_ival]


def write_intervals(ival_fn, lst_fn, scp_fn):
    """
    Writes interval file (ival_fn) using a list of
    word boundaries (lst_fn) and available feature
    segments (scp_fn)
    """

    conv_sides = make_conv_side_dictionary(scp_fn)

    with open(ival_fn, "w") as ival_file:
        with open(lst_fn, "r") as lst_file:
            for line in lst_file.read().splitlines():
                """
                Note: a line is of the format: organized_sw02111-A_000280-000367
                """

                query = line[:-24]

                conv_id, side = line[-23:-16], line[-15:-14]
                cs_id = conv_id + side

                start, stop = int(line[-13:-7]), int(line[-6:])

                query_ival, seg_ival, ark_info = query_lookup(conv_sides[cs_id], start, stop)

                ival_file.write(" ".join([
                    "%s_%s-%s_%s-%s" % (query, conv_id, side, start, stop), # query
                    "%s-%s_%06d-%06d" % (conv_id, side, seg_ival[0], seg_ival[1]), # segment
                    "%d-%d" % query_ival, # query endpoints
                    "%s:%s" % ark_info # -- ark file and position for reading
                ]) + "\n")

def main():
    """main function"""

    ival_fn = sys.argv[1]
    lst_fn = sys.argv[2]
    scp_fn = sys.argv[3]

    print("ival_fn: %s" % ival_fn)
    print("lst_fn: %s" % lst_fn)
    print("scp_fn: %s" % scp_fn)

    print("start @", datetime.now())
    write_intervals(ival_fn, lst_fn, scp_fn)
    print("end @", datetime.now())

if __name__ == "__main__":
    main()
