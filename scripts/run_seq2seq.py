__requires__ = "autoregkd"
import re
import sys

from autoregkd.training.__main__ import run_seq2seq

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    sys.exit(run_seq2seq())