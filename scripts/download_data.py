"""Download the original ChaosNLI JSONL files without third-party dependencies."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.request import urlopen


REVISION = "eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b"
BASE_URL = "https://huggingface.co/datasets/earino/chaosnli/resolve/" + REVISION + "/raw/"
CHECKSUMS = {
    "README.txt": "b02af8460b221a1a1a629da0d528a87e820a7d76f2772599360608e51de50f1b",
    "chaosNLI_snli.jsonl": "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
    "chaosNLI_mnli_m.jsonl": "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
    "chaosNLI_alphanli.jsonl": "f55817ecc11a02ba3030cd7d2c567d23843456809020723abad53f4fde024c9e",
}


def download_data(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, checksum in CHECKSUMS.items():
        destination = output_dir / filename
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == checksum:
            print("Verified", destination)
            continue
        with urlopen(BASE_URL + filename, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError("Checksum mismatch for " + filename)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
            os.replace(str(temporary_path), str(destination))
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        print("Downloaded", destination)
    print("ChaosNLI ready:", output_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/chaosNLI_v1.0",
        help="Destination directory for the four original files.",
    )
    args = parser.parse_args()
    try:
        download_data(args.output_dir)
    except (OSError, URLError, ValueError) as error:
        parser.exit(1, "Download failed: {}\n".format(error))


if __name__ == "__main__":
    main()
