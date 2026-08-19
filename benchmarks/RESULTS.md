# nyangrad benchmark results

Generated from `benchmarks/results.json` (2026-08-20T01:57:13+0900), 165 measurements in 335s of wall time.

## Machine and build


-------------  -------------------------------------------------------------
cpu            AMD Ryzen 5 5600X 6-Core Processor (6 cores / 12 threads)
gpu            NVIDIA GeForce RTX 3060 (28 SMs, sm_86, 12.88 GB, 2.36 MB L2)
numpy          2.2.6 on scipy-openblas 0.3.29
python         3.10.12
driver / cuda  610.47 / 12.6
torch          2.7.0+cu126


### Measured ceilings on this machine

Every 'percent of peak' below is against these, not against a spec sheet.


ceiling                     value          how it was measured
--------------------------  -------------  ----------------------------------------
gpu fp32 gemm               7130 GFLOP/s   torch.matmul fp32 4096^3 (tf32 disabled)
gpu bandwidth               297 GB/s       torch device-to-device copy of 67M fp32
cpu fp32 gemm, 1 thread     121 GFLOP/s    torch.matmul fp32 2048^3
cpu fp32 gemm, 6 threads    422 GFLOP/s    torch.matmul fp32 2048^3
cpu bandwidth               49 GB/s        numpy copyto of 34M fp32
gpu fp32 spec at max boost  15270 GFLOP/s  SMs x 128 lanes x 2 x max boost clock


The GPU was held under load for 5s before these were
taken, which brought the SM clock from 210 to 1657 MHz. Measuring a ceiling on an
idle card understates it, and would flatter everything compared against it.


### Clock stability


This is a throttling check rather than a per-measurement record of the clock:
sampling runs once a second, which is far coarser than most measurements here,
and much of the run is CPU-only with the GPU idling. Across the whole run the
card stayed between 41 and 63 C and never drew more than 170 W, well inside
its limits, so nothing here was measured on a throttled card. Under load the SM
clock ran at a median of 1695 MHz and peaked at 1965 MHz. Clocks were left on the
default governor rather than pinned, so these are the frequencies the hardware
delivers in ordinary use rather than a tuned best case.


## Correctness before performance

Timings from a backend that computes the wrong answer are worthless, so every
backend is checked against a numpy reference first.


backend         check                                   max abs err  max rel err
--------------  --------------------------------------  -----------  -----------  ----
nyangrad_cpu    gemm 256^3 vs numpy                        5.34e-05     7.73e-07  pass
nyangrad_cuda   gemm 256^3 vs numpy                        4.58e-05     6.63e-07  pass
nyangrad_cpu    nhwc_to_nchw compaction                    0.00e+00            -  pass
nyangrad_cuda   nhwc_to_nchw compaction                    0.00e+00            -  pass
nyangrad_numpy  2-layer mlp forward vs numpy reference     0.00e+00     0.00e+00  pass
nyangrad_cpu    2-layer mlp forward vs numpy reference     1.79e-06     5.49e-07  pass
nyangrad_cuda   2-layer mlp forward vs numpy reference     1.67e-06     5.12e-07  pass


### Square GEMM throughput summary

GFLOP/s                N=256   N=512  N=1024  N=2048  N=4096
--------------------  ------  ------  ------  ------  ------
nyangrad cpu, naive      2.0     1.1     0.4       -       -
nyangrad cpu, tiled      5.0     6.5     7.3     7.7     7.9
nyangrad cuda          312.8  1362.2  1493.9  2804.8  3271.3
numpy (OpenBLAS)       293.0   336.7   386.6   467.2   460.7
torch cpu, 1 thread    123.9   127.4   119.0   123.8   122.5
torch cpu, 6 threads   281.1   339.8   366.4   409.3   421.6
torch cuda (cuBLAS)   1963.5  5288.8  6599.1  7000.0  7218.9


### Optimisation progression

Each of my kernels against the previous one, then against the libraries.

size    naive  tiled  tiled/naive    cuda  cuda/tiled  cuda/1t BLAS  cuBLAS/cuda
------  -----  -----  -----------  ------  ----------  ------------  -----------
N=256     2.0    5.0        2.55x   312.8       62.3x          2.5x         6.3x
N=512     1.1    6.5        5.81x  1362.2      208.6x         10.7x         3.9x
N=1024    0.4    7.3       19.18x  1493.9      204.0x         12.6x         4.4x
N=2048      -    7.7            -  2804.8      364.0x         22.7x         2.5x
N=4096      -    7.9            -  3271.3      413.6x         26.7x         2.2x


## GEMM scaling

Square matmul, C = A @ B, fp32. Latency is the median of repeated batched
launches; GFLOP/s is 2mnk over that latency. tf32 is disabled for the torch
CUDA rows so every backend is doing the same fp32 arithmetic.



### Square


N = 256   (34 MFLOP)


backend                   ms  GFLOP/s  of peak  vs best   IQR
--------------------  ------  -------  -------  -------  ----
nyangrad cpu, naive   17.065      2.0       2%   998.6x    2%
nyangrad cpu, tiled    6.683      5.0       4%   391.1x    8%
nyangrad cuda          0.107    312.8       4%     6.3x    5%
numpy (OpenBLAS)       0.115    293.0      69%     6.7x  45%*
torch cpu, 1 thread    0.271    123.9     102%    15.8x    5%
torch cpu, 6 threads   0.119    281.1      67%     7.0x   19%
torch cuda (cuBLAS)    0.017   1963.5      28%     1.0x   24%


N = 512   (268 MFLOP)


backend                    ms  GFLOP/s  of peak  vs best   IQR
--------------------  -------  -------  -------  -------  ----
nyangrad cpu, naive   238.902      1.1       1%  4706.9x    1%
nyangrad cpu, tiled    41.098      6.5       5%   809.7x    2%
nyangrad cuda           0.197   1362.2      19%     3.9x    6%
numpy (OpenBLAS)        0.797    336.7      80%    15.7x  26%*
torch cpu, 1 thread     2.108    127.4     105%    41.5x    5%
torch cpu, 6 threads    0.790    339.8      81%    15.6x    8%
torch cuda (cuBLAS)     0.051   5288.8      74%     1.0x   10%


N = 1024   (2.1 GFLOP)


backend                     ms  GFLOP/s  of peak   vs best     IQR
--------------------  --------  -------  -------  --------  ------
nyangrad cpu, naive   5623.092      0.4       0%  17279.6x  1 shot
nyangrad cpu, tiled    293.212      7.3       6%    901.0x      1%
nyangrad cuda            1.437   1493.9      21%      4.4x     13%
numpy (OpenBLAS)         5.555    386.6      92%     17.1x     23%
torch cpu, 1 thread     18.043    119.0      98%     55.4x      5%
torch cpu, 6 threads     5.862    366.4      87%     18.0x     10%
torch cuda (cuBLAS)      0.325   6599.1      93%      1.0x     15%


N = 2048   (17.2 GFLOP)


backend                     ms  GFLOP/s  of peak  vs best     IQR
--------------------  --------  -------  -------  -------  ------
nyangrad cpu, tiled   2229.726      7.7       6%   908.5x  1 shot
nyangrad cuda            6.125   2804.8      39%     2.5x      4%
numpy (OpenBLAS)        36.770    467.2     111%    15.0x      8%
torch cpu, 1 thread    138.799    123.8     102%    56.6x      2%
torch cpu, 6 threads    41.974    409.3      97%    17.1x      5%
torch cuda (cuBLAS)      2.454   7000.0      98%     1.0x      4%


N = 4096   (137.4 GFLOP)


backend                      ms  GFLOP/s  of peak  vs best     IQR
--------------------  ---------  -------  -------  -------  ------
nyangrad cpu, tiled   17378.070      7.9       7%   912.8x  1 shot
nyangrad cuda            42.013   3271.3      46%     2.2x      4%
numpy (OpenBLAS)        298.349    460.7     109%    15.7x      8%
torch cpu, 1 thread    1121.971    122.5     101%    58.9x      1%
torch cpu, 6 threads    326.016    421.6     100%    17.1x      6%
torch cuda (cuBLAS)      19.039   7218.9     101%     1.0x      1%


### Rectangular


(8192x256) @ (256x256)   (1.1 GFLOP)


backend                  ms  GFLOP/s  of peak  vs best   IQR
--------------------  -----  -------  -------  -------  ----
nyangrad cuda         1.171    916.7      13%     7.2x    4%
numpy (OpenBLAS)      2.876    373.3      89%    17.7x  30%*
torch cpu, 1 thread   9.736    110.3      91%    59.8x    7%
torch cpu, 6 threads  3.718    288.8      68%    22.8x   13%
torch cuda (cuBLAS)   0.163   6591.6      92%     1.0x   13%


(512x2048) @ (2048x2048)   (4.3 GFLOP)


backend                    ms  GFLOP/s  of peak  vs best  IQR
--------------------  -------  -------  -------  -------  ---
nyangrad cpu, tiled   595.303      7.2       6%   932.8x   1%
nyangrad cuda           2.203   1949.3      27%     3.5x   7%
numpy (OpenBLAS)       11.561    371.5      88%    18.1x  19%
torch cpu, 1 thread    34.665    123.9     102%    54.3x   6%
torch cpu, 6 threads   10.646    403.4      96%    16.7x  10%
torch cuda (cuBLAS)     0.638   6730.2      94%     1.0x   8%


(2048x2048) @ (2048x512)   (4.3 GFLOP)


backend                    ms  GFLOP/s  of peak  vs best  IQR
--------------------  -------  -------  -------  -------  ---
nyangrad cpu, tiled   587.804      7.3       6%   895.1x   1%
nyangrad cuda           2.270   1891.9      27%     3.5x  10%
numpy (OpenBLAS)       15.126    284.0      67%    23.0x  22%
torch cpu, 1 thread    34.407    124.8     103%    52.4x   2%
torch cpu, 6 threads   10.559    406.8      96%    16.1x  13%
torch cuda (cuBLAS)     0.657   6540.5      92%     1.0x   8%


IQR is the interquartile range as a fraction of the median, i.e. run to run
spread. A star marks a measurement still jittery after being retaken; those are
threaded BLAS calls whose thread scheduling varies, not timing errors.

A few CPU rows read above 100% of peak. The ceiling was measured at N=2048, where
the operands do not fit in cache, so a smaller GEMM that stays resident in L3 can
legitimately beat it. It is a property of the cache, not a broken measurement.


## Striding and compaction
Views are metadata only, so nothing above is paid for until a kernel needs
contiguous input. This is that bill. Bandwidth counts the minimum required
traffic (read the source, write the destination), so a backend that moves more
than the minimum shows up here as low effective bandwidth.


### Latency (ms)

case                nyangrad cpu  nyangrad cpu, kernel only  nyangrad cuda  nyangrad cuda, kernel only  numpy (OpenBLAS)  torch cpu  torch cuda (cuBLAS)
------------------  ------------  -------------------------  -------------  --------------------------  ----------------  ---------  -------------------
transpose_2d_1k            5.307                      4.168          2.028                       0.318             4.663      0.802                0.045
transpose_2d_2k           16.669                     16.450          2.619                       1.308            21.235      6.677                0.198
transpose_2d_4k           72.352                     67.859          8.158                       5.312            94.436     21.202                1.043
nhwc_to_nchw             127.835                    122.098          5.980                       3.717            18.829     10.946                0.466
broadcast_row             43.769                     39.894         10.211                       4.280             9.531      8.539                0.216
strided_slice             13.027                     12.297          3.069                       1.248             3.585      2.552                0.160
elementwise_stream        11.918                      7.939          3.933                       0.450            12.758     11.274                0.450

### Effective bandwidth (GB/s)

case                nyangrad cpu  nyangrad cpu, kernel only  nyangrad cuda  nyangrad cuda, kernel only  numpy (OpenBLAS)  torch cpu  torch cuda (cuBLAS)
------------------  ------------  -------------------------  -------------  --------------------------  ----------------  ---------  -------------------
transpose_2d_1k              1.6                        2.0            4.1                        26.3               1.8       10.5                186.3
transpose_2d_2k              2.0                        2.0           12.8                        25.7               1.6        5.0                169.4
transpose_2d_4k              1.9                        2.0           16.5                        25.3               1.4        6.3                128.7
nhwc_to_nchw                 0.8                        0.8           17.2                        27.6               5.5        9.4                220.4
broadcast_row                1.5                        1.7            6.6                        15.7               7.0        7.9                311.2
strided_slice                2.6                        2.7           10.9                        26.9               9.4       13.2                209.1
elementwise_stream          11.3                       16.9           34.1                       298.5              10.5       11.9                298.4

The 'kernel only' rows reuse an output buffer instead of allocating one, which
separates the kernel from the allocation every op normally pays for. A kernel
reading at or just over the measured bandwidth ceiling is at the roof; the ceiling
and the kernel were timed minutes apart, so agreement to a percent is agreement.

  transpose_2d_1k      2D transpose, stride-1 reads become strided
  transpose_2d_2k      2D transpose at 16 MB
  transpose_2d_4k      2D transpose at 64 MB, far past L2
  nhwc_to_nchw         4D permute, the layout change a conv needs
  broadcast_row        stride-0 read expanded to 4096x4096
  strided_slice        every other element on both axes
  elementwise_stream   one streaming pass over compact memory


## Allocation cost per operation (microseconds)

Every nyangrad op allocates its output buffer straight from the device, while
torch serves the same request from a caching allocator. On a large buffer that
difference is bigger than any kernel difference.

buffer   nyangrad cpu  nyangrad cuda  torch cpu  torch cuda (cuBLAS)
-------  ------------  -------------  ---------  -------------------
0.26 MB           1.9            3.8        1.2                  2.1
4 MB              1.9          553.5        1.2                  2.0
67 MB            11.1         3499.8        9.9                  2.1


## Anatomy of one elementwise op (microseconds)

The parts do not add up, and that is the finding. Allocating costs a few
microseconds and launching costs a few more, but an op that does both costs
several times their sum, because freeing the output while the kernel writing it
is still in flight makes cudaFree wait for the device to drain. Keeping the
output alive removes the free and most of the cost with it.

step               nyangrad cpu  nyangrad cuda  what it isolates
-----------------  ------------  -------------  ------------------------------------------------------
launch_only                 0.5           11.1  output reused, nothing allocated or freed
alloc_free_only             1.8            4.5  allocate and free, no kernel
alloc_launch_free           2.4           66.7  what every op actually does
alloc_launch_kept           3.0           12.4  same, but the output is kept alive so nothing is freed


## End-to-end MLP

784 -> h -> h -> h -> 10 with ReLU and softmax cross entropy. The small
config is dominated by per-op dispatch, the large one by arithmetic, and the
ranking changes between them.

### Forward pass, latency in ms

config             nyangrad numpy  nyangrad cpu  nyangrad cuda  torch cpu, 1 thread  torch cpu, 6 threads  torch cuda (cuBLAS)
-----------------  --------------  ------------  -------------  -------------------  --------------------  -------------------
small_b64_h256               0.59         10.54           4.54                 0.47                  0.27                 0.26
medium_b256_h1024            8.18        230.21           4.63                12.69                  4.98                 0.32
large_b512_h2048            34.85       1441.62          11.47                89.16                 28.36                 1.67

### Forward pass, effective GFLOP/s

config             nyangrad numpy  nyangrad cpu  nyangrad cuda  torch cpu, 1 thread  torch cpu, 6 threads  torch cuda (cuBLAS)
-----------------  --------------  ------------  -------------  -------------------  --------------------  -------------------
small_b64_h256                 72             4              9                   91                   157                  167
medium_b256_h1024             182             6            322                  117                   299                 4622
large_b512_h2048              294             7            894                  115                   362                 6124

### Forward + backward, latency in ms

config             nyangrad numpy  nyangrad cpu  nyangrad cuda  torch cpu, 1 thread  torch cpu, 6 threads  torch cuda (cuBLAS)
-----------------  --------------  ------------  -------------  -------------------  --------------------  -------------------
small_b64_h256               2.69         51.90           6.42                 1.20                  0.89                 1.00
medium_b256_h1024           40.19       1171.03          29.20                36.10                 14.67                 1.10
large_b512_h2048           168.56       7288.30          65.00               248.62                 82.69                 4.77

### Forward + backward, effective GFLOP/s

config             nyangrad numpy  nyangrad cpu  nyangrad cuda  torch cpu, 1 thread  torch cpu, 6 threads  torch cuda (cuBLAS)
-----------------  --------------  ------------  -------------  -------------------  --------------------  -------------------
small_b64_h256                 48             2             20                  107                   144                  128
medium_b256_h1024             111             4            153                  124                   305                 4079
large_b512_h2048              183             4            473                  124                   372                 6446


## Per-op dispatch cost on a 1-element tensor (microseconds)

With one element there is no arithmetic left to measure, so this is pure
overhead. The ndarray row skips the autograd layer entirely, which separates
graph bookkeeping from kernel launch and allocation.

op                   nyangrad numpy  nyangrad cpu  nyangrad cuda  torch cpu  torch cuda (cuBLAS)
-------------------  --------------  ------------  -------------  ---------  -------------------
tensor_add_1elem                2.3           7.9           67.4        1.1                 12.0
tensor_matmul_1elem             2.8           9.2           85.4          -                    -
ndarray_add_1elem                 -           5.2           78.4          -                    -


## What one step actually asks of the backend

Counted by wrapping the kernel module in a proxy, so these are real call
counts rather than estimates.

config                   backend  graph nodes  launches fwd  launches fwd+bwd  allocations  h<->d copies
-----------------  -------------  -----------  ------------  ----------------  -----------  ------------
small_b64_h256      nyangrad cpu           30            42               157          158             2
small_b64_h256     nyangrad cuda           30            33               112          113             2
medium_b256_h1024   nyangrad cpu           30            42               157          158             2
medium_b256_h1024  nyangrad cuda           30            33               112          113             2
large_b512_h2048    nyangrad cpu           30            42               157          158             2
large_b512_h2048   nyangrad cuda           30            33               112          113             2

Most-called kernels in one nyangrad cpu step: Array x158, compact x77, matmul_tiled x15, ewise_mul x12, ewise_add x8, scalar_add x8

Most-called kernels in one nyangrad cuda step: Array x113, compact x32, matmul x20, ewise_mul x12, ewise_add x8, scalar_add x8
