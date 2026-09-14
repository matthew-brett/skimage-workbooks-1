"""Reference Bresenham walkers used to regenerate fixture JSON.

These are pure Python ports. They do not need ITK wheels or a Rust toolchain.
`zingl_line` matches `raster_geometry.bresenham_line` and Rust
`line_drawing::Bresenham3d` on the checked-in 3-D corpus. `itk_line` follows
`itkBresenhamLine.hxx` from ITK v5.4.0 (Index→Index via float Normalize).
"""

from __future__ import annotations

import numpy as np


def zingl_line(coord_a, coord_b, endpoint: bool = True):
    """Zingl / raster_geometry / line_drawing style (late step on ties).

    Faithful to ``raster_geometry.bresenham_line`` (including ``steps = -1``
    when an axis has zero delta; those axes never fire because their error
    never drops).
    """
    diffs = [abs(b - a) for a, b in zip(coord_a, coord_b)]
    steps = [1 if a < b else -1 for a, b in zip(coord_a, coord_b)]
    max_diff = max(diffs) if diffs else 0
    if max_diff == 0:
        return [tuple(coord_a)]
    updates = [max_diff / 2] * len(coord_a)
    coord = list(coord_a)
    out = []
    for _ in range(max_diff):
        # ``u`` is the error *before* this step's decrement (matches upstream).
        for j, (d, s, u) in enumerate(zip(diffs, steps, updates)):
            updates[j] -= d
            if u < 0:
                coord[j] += s
                updates[j] += max_diff
        out.append(tuple(coord))
    if endpoint:
        out.append(tuple(coord_b))
    return out


def raster_geometry_line(coord_a, coord_b, endpoint: bool = True):
    """Alias: same algorithm as the vendored raster_geometry helper."""
    return zingl_line(coord_a, coord_b, endpoint=endpoint)


def itk_build_line_direction(direction, length: int):
    """ITK `BresenhamLine::BuildLine` after `LastIndex` is known."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return [tuple(0 for _ in direction)]
    direction = direction / norm
    last = np.array(
        [int(length * direction[i]) for i in range(len(direction))],
        dtype=np.intp,
    )
    distances = np.abs(last)
    max_distance = int(distances.max())
    if max_distance == 0:
        return [tuple(0 for _ in direction)]
    main = int(np.argmax(distances))
    increment_error = 2 * distances
    overflow_inc = np.where(last < 0, -1, 1).astype(np.intp)
    maximal_error = np.full(len(direction), max_distance, dtype=np.intp)
    reduce_after = np.full(len(direction), 2 * max_distance, dtype=np.intp)
    accum = np.zeros(len(direction), dtype=np.intp)
    current = np.zeros(len(direction), dtype=np.intp)
    result = [tuple(int(x) for x in current)]
    steps = 1
    while steps < length:
        for i in range(len(direction)):
            if i == main:
                current[i] += overflow_inc[i]
            else:
                accum[i] += increment_error[i]
                if accum[i] >= maximal_error[i]:
                    current[i] += overflow_inc[i]
                    accum[i] -= reduce_after[i]
        result.append(tuple(int(x) for x in current))
        steps += 1
    return result


def itk_line(p0, p1):
    """ITK Index→Index Bresenham (float-normalised LastIndex)."""
    p0 = tuple(int(x) for x in p0)
    p1 = tuple(int(x) for x in p1)
    max_distance = max(abs(p0[i] - p1[i]) + 1 for i in range(len(p0)))
    direction = [float(p1[i] - p0[i]) for i in range(len(p0))]
    offsets = itk_build_line_direction(direction, max_distance + 1)
    indices = []
    for off in offsets:
        pt = tuple(p0[i] + off[i] for i in range(len(p0)))
        indices.append(pt)
        if pt == p1:
            break
    return indices
