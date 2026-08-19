#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace nyangrad {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4

#define V TILE
#define L 64   // (sqrt(256) * 4)
#define S 4    // Or whatever reduction slice size you choose

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}


// Utility function to convert contiguous index i to memory location from strides

__device__ size_t getStridedIdx(const size_t gid, CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * Helper function to get strided index from given gid
   */

  size_t ans = offset;
  int32_t idx = static_cast<int32_t>(shape.size) - 1;
  size_t curr = gid;

  while (curr != 0 && idx >= 0) {
    ans += (curr % shape.data[idx]) * strides.data[idx];
    curr /= shape.data[idx];
    idx--;
  }
   
  return ans;
}


////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * One thread per output element: map gid in the compact output back to the
   * strided location it comes from in a.
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t stridedIdx = getStridedIdx(gid, shape, strides, offset);
    out[gid] = a[stridedIdx];
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact a strided array into a dense one; the work happens in the kernel.
   */
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  /**
   * Kernel for EwiseSetitem
   */
  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t stridedIdx = getStridedIdx(gid, shape, strides, offset);
    out[stridedIdx] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Scatter the compact array a into the strided view of out.
   */
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), 
                                              VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  /**
   * Kernel for ScalarSetitem
   */
  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t stridedIdx = getStridedIdx(gid, shape, strides, offset);
    out[stridedIdx] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Write val into the strided view of out; size is the size of the view.
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.block, dim.grid>>>(val, out->ptr, size, VecToCuda(shape), 
                                              VecToCuda(strides), offset);
}


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * Each remaining elementwise and scalar op is a kernel plus a launcher with the
 * same shape, so the macros below stamp out both from the operation itself.
 */


#define DEFINE_EWISE_OP(func_name, op) \
__global__ void func_name ## Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op b[gid]; \
} \
\
void func_name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(a.size); \
  func_name ## Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define DEFINE_SCALAR_OP(func_name, op) \
__global__ void func_name ## Kernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op val; \
} \
\
void func_name(const CudaArray& a, const scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(a.size); \
  func_name ## Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define DEFINE_UNARY_FUNC(func_name, func) \
__global__ void func_name ## Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid]); \
} \
\
void func_name(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(a.size); \
  func_name ## Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

#define DEFINE_EWISE_FUNC(func_name, func) \
__global__ void func_name ## Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid], b[gid]); \
} \
\
void func_name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(a.size); \
  func_name ## Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define DEFINE_SCALAR_FUNC(func_name, func) \
__global__ void func_name ## Kernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid], val); \
} \
\
void func_name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(a.size); \
  func_name ## Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}


DEFINE_EWISE_OP(EwiseMul, *)
DEFINE_EWISE_OP(EwiseDiv, /)
DEFINE_EWISE_OP(EwiseEq, ==)
DEFINE_EWISE_OP(EwiseGe, >=)

DEFINE_SCALAR_OP(ScalarMul, *)
DEFINE_SCALAR_OP(ScalarDiv, /)
DEFINE_SCALAR_OP(ScalarEq, ==)
DEFINE_SCALAR_OP(ScalarGe, >=)

DEFINE_UNARY_FUNC(EwiseLog, log)
DEFINE_UNARY_FUNC(EwiseExp, exp)
DEFINE_UNARY_FUNC(EwiseTanh, tanh)

DEFINE_EWISE_FUNC(EwiseMaximum, max)

DEFINE_SCALAR_FUNC(ScalarPower, pow)
DEFINE_SCALAR_FUNC(ScalarMaximum, max)


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const float* a, const float* b, float* out, uint32_t M, uint32_t N, 
                             uint32_t P) {

  /**
   * Kernel for matmul
   */
  __shared__ float sA[L][S];
  __shared__ float sB[S][L];

  float c[V][V] = {0};
  float reg_a[V];
  float reg_b[V];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int nthreads = blockDim.y * blockDim.x;

  for (int ko = 0; ko < N; ko += S) {
    __syncthreads();

    // cooperatively fetch tile into sA
    for (int j = 0; j < (L * S) / nthreads; ++j) {
      int index = j * nthreads + tid;
      int y = index / S;
      int x = index % S;
      int global_y = blockIdx.y * L + y;
      int global_x = ko + x;

      if (global_y < M && global_x < N) {
          sA[y][x] = a[global_y * N + global_x];
      } else {
          sA[y][x] = 0.0f;
      }
    }

    // cooperatively fetch tile into sB
    for (int j = 0; j < (L * S) / nthreads; ++j) {
      int index = j * nthreads + tid;
      int y = index / L;
      int x = index % L;
      int global_y = ko + y;
      int global_x = blockIdx.x * L + x;

      if (global_y < N && global_x < P) {
          sB[y][x] = b[global_y * P + global_x];
      } else {
          sB[y][x] = 0.0f;
      }
    }

    __syncthreads();

    for (int ki = 0; ki < S; ki++) {
      for (int y = 0; y < V; ++y) {
        reg_a[y] = sA[threadIdx.y * V + y][ki];
      }
      for (int x = 0; x < V; ++x) {
        reg_b[x] = sB[ki][threadIdx.x * V + x];
      }

      for (int y = 0; y < V; y++) {
        for (int x = 0; x < V; x++) {
          c[y][x] += reg_a[y] * reg_b[x];
        }
      }
    }
  }

  int ybase = blockIdx.y * blockDim.y + threadIdx.y;
  int xbase = blockIdx.x * blockDim.x + threadIdx.x;

  for (int y = 0; y < V; ++y) {
    for (int x = 0; x < V; ++x) {
      int out_y = ybase * V + y;
      int out_x = xbase * V + x;

      if (out_y < M && out_x < P) {
          out[out_y * P + out_x] = c[y][x];
      }
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Matmul of compact (M x N) by (N x P). The kernel tiles through shared memory
   * with register-level accumulation, and handles sizes that are not multiples
   * of the tile size, so there is no separate aligned path like on the CPU.
   */

  dim3 grid((P + L - 1) / L, (M + L - 1) / L);
  dim3 block(L / V, L / V);

  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const float* a, float* out, size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    scalar_t curr = a[gid * reduce_size];

    for (size_t i = 0; i < reduce_size; i++) {
      curr = max(curr, a[gid * reduce_size + i]);
    }

    out[gid] = curr;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Max over each contiguous block of reduce_size elements, one thread per block.
   */
  CudaDims dim = CudaOneDim(out->size);

  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}


__global__ void ReduceSumKernel(const float* a, float* out, size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    scalar_t curr = 0;

    for (size_t i = 0; i < reduce_size; i++) {
      curr += a[gid * reduce_size + i];
    }

    out[gid] = curr;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Sum over each contiguous block of reduce_size elements, one thread per block.
   */
  CudaDims dim = CudaOneDim(out->size);

  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

}  // namespace cuda
}  // namespace nyangrad

PYBIND11_MODULE(_cuda_backend, m) {
  namespace py = pybind11;
  using namespace nyangrad;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}

