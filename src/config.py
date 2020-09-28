import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

if __name__ == '__main__':
    print(SRC_ROOT)
    print(PRO_ROOT)