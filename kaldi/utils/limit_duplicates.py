from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from random import uniform

def main():
    """
    python processing script to get rid of short repeat transcriptions
    for example, it removes having an excessive amount of uh-huh's
    """

    max_repeats = float(sys.argv[1])

    counts = dict()

    trns = sys.stdin.read().splitlines()

    num_trns = len(trns)

    for i in range(num_trns):
        _, trn_text = trns[i].split(" ", 1)

        if len(trn_text.split(" ")) < 10:
            if trn_text in counts:
                counts[trn_text] += 1
            else:
                counts[trn_text] = 1

    for i in range(num_trns-1, -1, -1):
        _, trn_text = trns[i].split(" ", 1)

        if trn_text in counts and counts[trn_text] >= max_repeats:
            if uniform(0, 1) < (max_repeats / counts[trn_text]):
                continue
            else:
                del trns[i]

    sys.stdout.write("\n".join(trns) + "\n")


if __name__ == "__main__":
    main()
