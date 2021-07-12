__requires__ = "autoregkd"
import re
import sys

from ..src.autoregkd.training.run_qa import main as run_qa

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    sys.exit(run_qa())
