# sandbox

Scratch space and early work that is not part of the nyangrad library itself.

- `warmup/` - a from-scratch softmax regression / two layer network in plain NumPy, plus a C++ and
  pybind11 version of the training step. This is where I started before building the autograd engine, so
  none of it depends on the `nyangrad` package. Build the C++ extension with `make` inside `warmup/`.

There is also a `course-materials/` folder locally (git-ignored) where I keep the CMU 10-714 notebooks
and notes I worked through.
