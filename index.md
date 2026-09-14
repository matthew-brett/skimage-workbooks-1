# Port notes

Working notes on the move from the scikit-image 1 API to the scikit-image 2
API, with an emphasis on coordinate conventions.

## Diagnostic notebooks

* {doc}`on_lines` — how `draw.line` and `draw.line_nd` rasterise a segment,
  why they disagree, and how they compare with Pillow and OpenCV.
* {doc}`hessian_review` — review of the `hessian_matrix` algorithm, why the
  current implementation gives excess error, and how it can be fixed.  More
  detail below.
* {doc}`on_hessian` — more details and exploration of `hessian_matrix`.  In
  particular, an analysis of why `hessian_matrix` returns a different mixed
  partial depending on the `order` argument, and what it costs at the image
  border.
* {doc}`on_blob_dog` — how the three blob detectors disagree on radius, and
  which differences are defects.
* {doc}`on_meijering` — problems with the scale-space search of the Meijering
  algorithm.
* {doc}`on_frangi` — various potential defects in the Frangi implementation.
* {doc}`on_warp_maps` — what each kind of argument to `warp`'s `inverse_map`
  means, and where the kinds disagree with each other.
* {doc}`rotate_description` — the direction and centre conventions of
  `transform.rotate`.
* {doc}`bresenham_nd_cython` - implementation of Bresenham's algorithm in N-D.

```{note}
Most of this material is AI-generated. See the repository `README.md` for what
that means for re-use.
```
