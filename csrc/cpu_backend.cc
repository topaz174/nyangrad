#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace nyangrad {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Walk the strided array a and write its elements into out in compact order.
   * shape is shared; strides and offset describe a, since out is compact.
   */
  size_t ndim = shape.size();
  std::vector<int32_t> position(ndim, 0);

  for (size_t i = 0; i < out->size; i++) {
    size_t aIdx = offset;
    size_t pIdx = ndim - 1;

    // freshly calculate index
    for (size_t j = 0; j < ndim; j++) {
      aIdx += position[j] * strides[j];
    }

    out->ptr[i] = a.ptr[aIdx];

    // increment the vector cleanly
    while (pIdx > 0 && position[pIdx] >= shape[pIdx] - 1) {
      position[pIdx] = 0;
      pIdx--;
    }

    position[pIdx]++;
  }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * The inverse of Compact: read a in compact order and scatter it into the
   * strided view of out described by shape, strides and offset.
   */
  size_t ndim = shape.size();
  std::vector<int32_t> position(ndim, 0);

  for (size_t i = 0; i < a.size; i++) {
    size_t oIdx = offset;
    size_t pIdx = ndim - 1;

    // freshly calculate index
    for (size_t j = 0; j < ndim; j++) {
      oIdx += position[j] * strides[j];
    }

    out->ptr[oIdx] = a.ptr[i];

    // increment the vector cleanly
    while (pIdx > 0 && position[pIdx] >= shape[pIdx] - 1) {
      position[pIdx] = 0;
      pIdx--;
    }

    position[pIdx]++;
  }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Write val into the strided view of out. size is the number of elements in
   * the view, which is smaller than out.size whenever the view is a subset.
   */

  size_t ndim = shape.size();
  std::vector<int32_t> position(ndim, 0);

  for (size_t i = 0; i < size; i++) {
    size_t oIdx = offset;
    size_t pIdx = ndim - 1;

    // freshly calculate index
    for (size_t j = 0; j < ndim; j++) {
      oIdx += position[j] * strides[j];
    }

    out->ptr[oIdx] = val;

    // increment the vector cleanly
    while (pIdx > 0 && position[pIdx] >= shape[pIdx] - 1) {
      position[pIdx] = 0;
      pIdx--;
    }

    position[pIdx]++;
  }
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * The remaining elementwise and scalar kernels all share the same loop, so they
 * are generated from these macros rather than written out one by one.
 */

#define DEFINE_EWISE_OP(func_name, op) \
void func_name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = a.ptr[i] op b.ptr[i]; \
  } \
}

#define DEFINE_SCALAR_OP(func_name, op) \
void func_name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = a.ptr[i] op val; \
  } \
}

#define DEFINE_UNARY_FUNC(func_name, func) \
void func_name(const AlignedArray& a, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i]); \
  } \
}

#define DEFINE_EWISE_FUNC(func_name, func) \
void func_name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i], b.ptr[i]); \
  } \
}

#define DEFINE_SCALAR_FUNC(func_name, func) \
void func_name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
  for (size_t i = 0; i < a.size; i++) { \
    out->ptr[i] = func(a.ptr[i], val); \
  } \
}

DEFINE_EWISE_OP(EwiseMul, *)
DEFINE_EWISE_OP(EwiseDiv, /)
DEFINE_EWISE_OP(EwiseEq, ==)
DEFINE_EWISE_OP(EwiseGe, >=)

DEFINE_SCALAR_OP(ScalarMul, *)
DEFINE_SCALAR_OP(ScalarDiv, /)
DEFINE_SCALAR_OP(ScalarEq, ==)
DEFINE_SCALAR_OP(ScalarGe, >=)

DEFINE_UNARY_FUNC(EwiseLog, std::log)
DEFINE_UNARY_FUNC(EwiseExp, std::exp)
DEFINE_UNARY_FUNC(EwiseTanh, std::tanh)

DEFINE_EWISE_FUNC(EwiseMaximum, std::max)

DEFINE_SCALAR_FUNC(ScalarPower, std::pow)
DEFINE_SCALAR_FUNC(ScalarMaximum, std::max)


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Naive three-loop matmul of compact (m x n) by (n x p) into compact (m x p).
   */

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      out->ptr[i * p + j] = 0;

      for (size_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply two TILE x TILE matrices and accumulate into out, which is not
   * zeroed here. __restrict__ and the alignment hints are what let the compiler
   * vectorise this inner kernel.
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (int i = 0; i < TILE; i++) {
    for (int k = 0; k < TILE; k++) {
      float a_val = a[i * TILE + k];

      for (int j = 0; j < TILE; j++) {
        out[i * TILE + j] += a_val * b[j + k * TILE];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Tile-by-tile matmul over 4D tiled layouts, e.g. a[m/TILE][n/TILE][TILE][TILE],
   * accumulating each tile product through AlignedDot. Only called when m, n and
   * p all divide evenly by TILE.
   */
  std::memset(out->ptr, 0, m * p * sizeof(float));

  for (int i = 0; i < m / TILE; i++) {
    for (int j = 0; j < p / TILE; j++) {
      for (int k = 0; k < n / TILE; k++){
        AlignedDot(a.ptr + i * (n * TILE) + k * (TILE * TILE), 
                   b.ptr + j * (TILE * TILE) + k * (p * TILE), 
                   out->ptr + i * (p * TILE) + j * (TILE * TILE));
      }
    }
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Max over each contiguous block of reduce_size elements.
   */

  for (size_t i = 0; i < out->size; i++) {
    scalar_t curr = a.ptr[i * reduce_size];

    for (size_t j = 0; j < reduce_size; j++) {
      curr = std::max(curr, a.ptr[i * reduce_size + j]);
    }

    out->ptr[i] = curr;
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Sum over each contiguous block of reduce_size elements.
   */

  for (size_t i = 0; i < out->size; i++) {
    scalar_t curr = 0.0;

    for (size_t j = 0; j < reduce_size; j++) {
      curr += a.ptr[i * reduce_size + j];
    }

    out->ptr[i] = curr;
  }
}

}  // namespace cpu
}  // namespace nyangrad

PYBIND11_MODULE(_cpu_backend, m) {
  namespace py = pybind11;
  using namespace nyangrad;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
