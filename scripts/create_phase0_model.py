import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspectnet_cx.release.create_phase0_model import main

if __name__ == "__main__":
    main()
