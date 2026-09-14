# Port notes

Working notes on the move from the scikit-image 1 API to the scikit-image 2
API, with an emphasis on coordinate conventions.

## Refactoring plans

* {doc}`coordinate_review` — what coordinate convention we want in SK2-API.
* {doc}`coordinate_port_plan` — staged plan to move `_skimage2` to array order.
* {doc}`hessian_refactor_plan` — correctness fix for `hessian_matrix` at the
  image border, and removal of `order` from `structure_tensor`.
* {doc}`warp_refactor` — what `warp` should accept as `inverse_map`.

## Diagnostic notebooks

* {doc}`on_lines` — how `draw.line` and `draw.line_nd` rasterise a segment,
  why they disagree, and how they compare with Pillow and OpenCV.
* {doc}`on_hessian` — why `hessian_matrix` returns a different mixed partial
  depending on the `order` argument, and what it costs at the image border.
* {doc}`on_blob_dog` — how the three blob detectors disagree on radius, and
  which differences are defects.
* {doc}`on_warp_maps` — what each kind of argument to `warp`'s `inverse_map`
  means, and where the kinds disagree with each other.
* {doc}`rotate_description` — the direction and centre conventions of
  `transform.rotate`.

```{note}
Most of this material is AI-generated. See the repository `README.md` for what
that means for re-use.
```
