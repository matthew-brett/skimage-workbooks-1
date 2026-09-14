#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
"""Local copy of early-step Bresenham for the diagnosis notebook.

Standalone: no ``_skimage2`` imports. ``_line`` matches scikit-image's
2-D integer Bresenham; ``_bresenham_nd`` is the N-D generalisation that
matches ``_line`` bit for bit when ``ndim == 2``.
"""
import numpy as np

cimport numpy as cnp

cnp.import_array()


def _line(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1):
    """Generate 2-D line pixel coordinates (early-step Bresenham)."""
    cdef char steep = 0
    cdef Py_ssize_t r = r0
    cdef Py_ssize_t c = c0
    cdef Py_ssize_t dr = abs(r1 - r0)
    cdef Py_ssize_t dc = abs(c1 - c0)
    cdef Py_ssize_t sr, sc, d, i

    cdef Py_ssize_t[::1] rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    cdef Py_ssize_t[::1] cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)

    with nogil:
        if (c1 - c) > 0:
            sc = 1
        else:
            sc = -1
        if (r1 - r) > 0:
            sr = 1
        else:
            sr = -1
        if dr > dc:
            steep = 1
            c, r = r, c
            dc, dr = dr, dc
            sc, sr = sr, sc
        d = (2 * dr) - dc

        for i in range(dc):
            if steep:
                rr[i] = c
                cc[i] = r
            else:
                rr[i] = r
                cc[i] = c
            while d >= 0:
                r = r + sr
                d = d - (2 * dc)
            c = c + sc
            d = d + (2 * dr)

        rr[dc] = r1
        cc[dc] = c1

    return np.asarray(rr), np.asarray(cc)


def _bresenham_nd(const cnp.intp_t[::1] start, const cnp.intp_t[::1] stop):
    r"""Generate N-D Bresenham line coordinates.

    Find line distance traveled in each axis direction.

    Call these deltas.

    Select the axis with the largest absolute value for delta; label this as
    the "major" axis.

    Without loss of generality, say this is the first axis.

    Without loss of generality, describe for 3D.

    Without loss of generality, say the line starts at coordinate $(0, 0, 0)$
    (we can always subtract the start coordinate to make this so, and add it
    back when calculating the coordinates on the line).

    Without loss of generality, declare all the deltas to be positive; we can
    deal with negative deltas by suitable changes to the step sizes.

    Step along the major axis in steps of $1$.  Because this has the longest
    distance traveled, by definition, all the other axes will move at step
    sizes $\leq 1$.

    Let us say we are at position on the line $m, n, p$ where $m$ is the
    position on the major axis.  $m$ is the number of steps we have taken on
    the major axis.

    Call the gradients of the line in each axis $g$.  $g_0$ is the gradient of
    the first (here, major) axis, so is $1$.

    $g_1, g_2$ are given by $\delta_1 / \delta_0$, $\delta_2 / \delta_0$.
    All $g$ are $\leq 1$.

    With $(0, 0, 0)$ origin, and first (major) axis of $m$, then (in continuous
    coordinates) $n = m g_1$ and $p = m g_2$.

    Let us say that the we have stored the previous position on the line as
    $pc_0, pc_1, pc_2$.  $pc_0 = m - 1$.  The question to answer at this step
    is whether we should update $pc_1$ by $1$, and whether should update $pc_2$
    by $1$.

    Consider the second axis (first non-major axis). The continuous coordinate
    is $n$.  The integer pixel coordinate we last used is $pc_1$.  The distance
    (residual) along the second axis, from $pc_1$ to $n$ is
    $n - pc_1 = m g_1 - pc_1$.  Call coordinates $< pc_1$: above $pc_1$.
    Negative residuals mean that $n$ is above $pc_1$.  A residual of $0.5$ or
    greater means that $pc_1 + 1$ is as close, or closer to $n$, than $pc_1$.

    We decide to update the $pc_1$ coordinate (to $pc_1 + 1$) when the residual
    $m g_1 - pc_1 \geq 0.5$; equivalently, when the residual less $0.5$ is
    non-negative, $m g_1 - pc_1 - 0.5 \geq 0$.

    We want to keep this score as an integer to save floating point
    calculations, with their associated errors.  Remembering
    $g_1 = \delta_1 / \delta_0$, we can multiply through by $2 \delta_0$ to
    get what we call the `switch_score` $s$:

    $$
    s = 2 m \delta_1 - 2 \delta_0 pc_1 - \delta_0 \geq 0.
    $$

    We start (see above) at $(0, 0, 0)$, with $m = 0$.  $pc_1$ and $pc_2$ are
    $0$. Now take one step along the major axis so $m = 1$.  The switch score
    $s$ for the second axis is $2 \delta_1 - \delta_0$.  This will only
    trigger a switch to $pc_1 = 1$ if $\delta_1 \geq \delta_0 / 2$ or,
    equivalently $g_1 \geq 0.5$.  Subsequently, at each step we
    unconditionally add $2 \delta_1$ (for the increment of $m$), and subtract
    $2 \delta_0$ if we switch (add one to the upcoming $pc_1$), to accumulate
    the results of the switch cost formula above.

    In what follows, read `step_i + 1` as $m$, `previous - start` as $pc$,
    `major_delta` as $\delta_0$.
    """
    cdef Py_ssize_t ndim = start.shape[0]
    if stop.shape[0] != ndim:
        raise ValueError("start and stop must have the same length")
    if ndim < 1:
        raise ValueError("start and stop must have length at least 1")
    if ndim > cnp.NPY_MAXDIMS:
        raise ValueError(
            f"ndim={ndim} exceeds limit of {cnp.NPY_MAXDIMS}"
        )

    cdef Py_ssize_t i, axis, major = 0
    cdef Py_ssize_t major_delta = 0
    cdef Py_ssize_t step_i
    cdef cnp.intp_t d
    cdef cnp.intp_t delta[cnp.NPY_MAXDIMS]
    cdef cnp.intp_t step[cnp.NPY_MAXDIMS]
    cdef cnp.intp_t switch_score[cnp.NPY_MAXDIMS]
    cdef cnp.intp_t previous[cnp.NPY_MAXDIMS]

    for i in range(ndim):
        d = stop[i] - start[i]
        if d > 0:
            delta[i] = d
            step[i] = 1
        elif d < 0:
            delta[i] = -d
            step[i] = -1
        else:
            delta[i] = 0
            step[i] = 0
        if delta[i] > major_delta:
            major_delta = delta[i]
            major = i

    # Final coordinate adds one to steps that must be stored.
    cdef cnp.intp_t[:, ::1] coords = np.empty((ndim, major_delta + 1),
                                              dtype=np.intp)

    for i in range(ndim):
        previous[i] = start[i]
        # Unused on major axis; kept so all axes share one update form.
        switch_score[i] = 2 * delta[i] - major_delta

    with nogil:
        for step_i in range(major_delta):
            for i in range(ndim):
                coords[i, step_i] = previous[i]
            for axis in range(ndim):
                if axis == major:
                    continue
                if switch_score[axis] >= 0:
                    previous[axis] += step[axis]
                    switch_score[axis] -= 2 * major_delta
                switch_score[axis] += 2 * delta[axis]
            previous[major] += step[major]

        # Add end coordinate.
        for i in range(ndim):
            coords[i, major_delta] = stop[i]

    return np.asarray(coords)
