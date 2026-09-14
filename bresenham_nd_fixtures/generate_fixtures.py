#!/usr/bin/env python3
"""Regenerate Bresenham N-D comparison fixtures under this directory.

Writes:
  nd_bresenham_references.json  — all references, plus agreement rates
  itk_bresenham_line.json       — ITK port pixels only
  ours_bresenham_nd.json        — `_bresenham_nd` pixels only

Needs the `bresenham-nd` build on ``sys.path`` (see ``--skimage-root``).
External refs (ITK port, Zingl / raster_geometry) are pure Python and need
no extra packages. Optional ``--rust-bin`` re-checks the 3-D Zingl field
against a `line_drawing` Bresenham3d helper.

Example::

    export PYENV_VERSION=bresenham-nd
    python bresenham_nd_fixtures/generate_fixtures.py \\
        --skimage-root ../bresenham-nd/build-install/usr/lib/python3.13/site-packages
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import itk_line, raster_geometry_line, zingl_line  # noqa: E402


def _default_skimage_paths() -> list[Path]:
    """Likely build-install trees next to port-notes."""
    root = HERE.parents[1]  # .../skimages
    candidates = [
        root / "bresenham-nd" / "build-install" / "usr" / "lib",
    ]
    out = []
    for base in candidates:
        if not base.is_dir():
            continue
        out.extend(sorted(base.glob("python*/site-packages")))
    return out


def load_draw(skimage_root: Path | None):
    """Import `_bresenham_nd` and `_line` from a build tree or the env."""
    if skimage_root is not None:
        sys.path.insert(0, str(skimage_root))
    else:
        for path in _default_skimage_paths():
            sys.path.insert(0, str(path))
    try:
        from _skimage2.draw._draw import _bresenham_nd, _line
    except ImportError as exc:
        raise SystemExit(
            "Cannot import _skimage2.draw._draw. Pass --skimage-root to the "
            "bresenham-nd build-install site-packages, or install that tree "
            f"on PYTHONPATH.\n({exc})"
        ) from exc
    return _bresenham_nd, _line


def ours_factory(_bresenham_nd):
    def ours(start, stop):
        coords = _bresenham_nd(
            np.asarray(start, dtype=np.intp), np.asarray(stop, dtype=np.intp)
        )
        return [tuple(int(v) for v in pt) for pt in coords.T]

    return ours


def line2_factory(_line):
    def line2(start, stop):
        rr, cc = _line(start[0], start[1], stop[0], stop[1])
        return list(zip(map(int, rr), map(int, cc)))

    return line2


def rust_bin_line(rust_bin: Path):
    def rust_line3d(p0, p1):
        out = subprocess.check_output(
            [str(rust_bin), *[str(x) for x in p0], *[str(x) for x in p1]],
            text=True,
        )
        return [
            tuple(int(v) for v in line.split(","))
            for line in out.strip().splitlines()
            if line
        ]

    return rust_line3d


def corpora(rng: np.random.Generator):
    """Same starts/stops as the checked-in fixtures (seed 0)."""
    pairs_2d = [
        ((a, b), (c, d))
        for a, b, c, d in itertools.product(range(-3, 4), repeat=4)
    ]
    grid3 = list(itertools.product(range(-2, 3), repeat=3))
    pairs_3d = [
        ((0, 0, 0), (2, 4, 8)),
        ((1, 2, 3), (-2, 5, 0)),
        ((0, 0, 0), (0, 0, 5)),
    ]
    pairs_3d += [
        (grid3[i], grid3[j])
        for i, j in rng.choice(len(grid3), size=(200, 2))
        if grid3[i] != grid3[j]
    ]
    grid4 = list(itertools.product(range(-1, 2), repeat=4))
    pairs_4d = [
        (grid4[i], grid4[j])
        for i, j in rng.choice(len(grid4), size=(100, 2))
        if grid4[i] != grid4[j]
    ]
    return pairs_2d, pairs_3d, pairs_4d


def pack(pairs, refs):
    rows = []
    for start, stop in pairs:
        row = {"start": list(start), "stop": list(stop)}
        for name, fn in refs:
            row[name] = [list(pt) for pt in fn(start, stop)]
        rows.append(row)
    return rows


def agreement(pairs, refs):
    summary = {}
    names = [n for n, _ in refs]
    fns = dict(refs)
    for a in names:
        for b in names:
            if a >= b:
                continue
            ok = sum(fns[a](s, t) == fns[b](s, t) for s, t in pairs)
            summary[f"{a}_vs_{b}"] = {
                "agree": ok,
                "n": len(pairs),
                "rate": round(ok / len(pairs), 4),
            }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE,
        help="Directory for the three JSON files (default: this folder)",
    )
    parser.add_argument(
        "--skimage-root",
        type=Path,
        default=None,
        help="site-packages directory that contains _skimage2",
    )
    parser.add_argument(
        "--rust-bin",
        type=Path,
        default=None,
        help="Optional Bresenham3d CLI; if set, must match zingl_line on dim3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for 3-D / 4-D random pairs (default 0)",
    )
    args = parser.parse_args(argv)

    _bresenham_nd, _line = load_draw(args.skimage_root)
    ours = ours_factory(_bresenham_nd)
    line2 = line2_factory(_line)
    rust_field = zingl_line
    if args.rust_bin is not None:
        rust_field = rust_bin_line(args.rust_bin)

    pairs_2d, pairs_3d, pairs_4d = corpora(np.random.default_rng(args.seed))

    refs2 = [
        ("ours", ours),
        ("skimage_line", line2),
        ("itk", itk_line),
        ("raster_geometry", raster_geometry_line),
    ]
    refs3 = [
        ("ours", ours),
        ("itk", itk_line),
        ("raster_geometry", raster_geometry_line),
        ("rust_line_drawing", rust_field),
    ]
    refs4 = [
        ("ours", ours),
        ("itk", itk_line),
        ("raster_geometry", raster_geometry_line),
    ]

    if args.rust_bin is not None:
        miss = sum(
            zingl_line(a, b) != rust_field(a, b) for a, b in pairs_3d
        )
        if miss:
            raise SystemExit(
                f"--rust-bin disagrees with zingl_line on {miss}/{len(pairs_3d)} "
                "3-D segments"
            )

    data = {
        "meta": {
            "itk_source": (
                "itkBresenhamLine.hxx from ITK v5.4.0 (Python port of BuildLine)"
            ),
            "raster_geometry": (
                "vendored bresenham_line, endpoint=True "
                "(pip package broken on NumPy 2)"
            ),
            "rust": (
                "line_drawing 1.0.1 Bresenham3d rule "
                "(Python zingl_line; optional --rust-bin check)"
            ),
            "ours": "_bresenham_nd in bresenham-nd worktree",
            "note": (
                "External N-D Bresenhams share the Chebyshev length but differ "
                "from skimage/_bresenham_nd on ties. skimage matches _line in "
                "2-D exactly. ITK routes Index→Index through a normalised float "
                "direction."
            ),
            "generator": "bresenham_nd_fixtures/generate_fixtures.py",
            "seed": args.seed,
        },
        "dim2": pack(pairs_2d, refs2),
        "dim3": pack(pairs_3d, refs3),
        "dim4": pack(pairs_4d, refs4),
        "agreement": {
            "dim2": agreement(pairs_2d, refs2),
            "dim3": agreement(pairs_3d, refs3),
            "dim4": agreement(pairs_4d, refs4),
        },
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    refs_path = out / "nd_bresenham_references.json"
    refs_path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {refs_path} ({refs_path.stat().st_size} bytes)")
    print(json.dumps(data["agreement"], indent=2))

    itk_only = {
        "meta": {k: data["meta"][k] for k in data["meta"] if k != "generator"},
        "dim2": [
            {"start": r["start"], "stop": r["stop"], "pixels": r["itk"]}
            for r in data["dim2"]
        ],
        "dim3": [
            {"start": r["start"], "stop": r["stop"], "pixels": r["itk"]}
            for r in data["dim3"]
        ],
        "dim4": [
            {"start": r["start"], "stop": r["stop"], "pixels": r["itk"]}
            for r in data["dim4"]
        ],
    }
    itk_path = out / "itk_bresenham_line.json"
    itk_path.write_text(json.dumps(itk_only, separators=(",", ":")))

    line_ok = all(
        ours(a, b) == line2(a, b) for a, b in pairs_2d
    )
    ours_only = {
        "meta": {
            "generator": "_bresenham_nd",
            "matches_skimage_line_2d": line_ok,
            "script": "bresenham_nd_fixtures/generate_fixtures.py",
            "seed": args.seed,
        },
        "dim2": [
            {"start": r["start"], "stop": r["stop"], "pixels": r["ours"]}
            for r in data["dim2"]
        ],
        "dim3": [
            {"start": r["start"], "stop": r["stop"], "pixels": r["ours"]}
            for r in data["dim3"]
        ],
        "dim4": [
            {"start": r["start"], "stop": r["stop"], "pixels": r["ours"]}
            for r in data["dim4"]
        ],
    }
    ours_path = out / "ours_bresenham_nd.json"
    ours_path.write_text(json.dumps(ours_only, separators=(",", ":")))
    print(f"wrote {itk_path.name} and {ours_path.name}")
    if not line_ok:
        print("warning: ours != skimage _line on some 2-D pairs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
