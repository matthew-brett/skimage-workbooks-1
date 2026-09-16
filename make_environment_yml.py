#!/usr/bin/env python3
"""Generate a conda environment.yml from a pip requirements file.

Requirements conda-forge can satisfy become conda dependencies; direct-URL
specs (name@https://...) and pip-only distributions go under the pip
subsection, so `conda env create` installs everything in one pass.
"""

import argparse
import re
import sys
from pathlib import Path

# conda-forge's opencv-python-headless build is stale (feedstock archived at
# 4.11.0.86, no osx-arm64/aarch64/win-64 builds) and flagged broken; the PyPI
# wheels are current and universal.
PIP_ONLY = {"opencv-python-headless"}

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_requirements(path):
    reqs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


def package_name(req):
    m = _NAME.match(req)
    return m.group(0).lower() if m else req.lower()


def split_conda_pip(reqs):
    conda, pip = [], []
    for req in reqs:
        # Conda cannot express PEP 508 direct references (name@url).
        if "@" in req or package_name(req) in PIP_ONLY:
            pip.append(req)
        else:
            conda.append(req)
    return conda, pip


def write_environment(name, conda, pip, out):
    deps = ["python"] + conda + (["pip"] if pip else [])
    lines = [f"name: {name}", "channels:", "  - conda-forge", "dependencies:"]
    lines += [f"  - {d}" for d in deps]
    if pip:
        lines.append("  - pip:")
        lines += [f'      - "{d}"' for d in pip]
    out.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "requirements",
        nargs="?",
        type=Path,
        default=Path("build_requirements.txt"),
        help="input requirements file (default: %(default)s)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("environment.yml"),
        help="output file (default: %(default)s)",
    )
    ap.add_argument(
        "-n",
        "--name",
        default="skimage-workbooks",
        help="conda environment name (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    if not args.requirements.is_file():
        ap.error(f"{args.requirements}: not a file")

    conda, pip = split_conda_pip(parse_requirements(args.requirements))
    with args.output.open("w") as f:
        write_environment(args.name, conda, pip, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
