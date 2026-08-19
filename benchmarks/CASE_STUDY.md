# Benchmarking nyangrad: what I learned making my own kernels go fast

This is a write-up of the benchmark results for nyangrad, the deep learning framework I
wrote from scratch. It has its own autodiff engine, its own strided array library, and its
own CPU and GPU kernels underneath, and the question I wanted to answer was simply: how
close to the real thing did I actually get?

The headline, on a Ryzen 5 5600X and an RTX 3060, all fp32:

- My CUDA matmul kernel reaches **3271 GFLOP/s** at 4096x4096, which is **46% of what cuBLAS
  gets on the same card** with tf32 disabled, and **27x faster than a single-threaded BLAS**
  on the CPU.
- Tiling the CPU matmul was worth **19x** over the naive three-loop version at N=1024, and
  the reason it is 19x and not 3x turned out to be more interesting than the tiling itself.
- My CUDA elementwise kernel runs at **298 GB/s**, which is the memory bandwidth ceiling of
  this card as I measured it (297 GB/s). That kernel is done. There is nothing left to win.
- Despite that, **89% of the wall time of an elementwise op is not the kernel**. It is
  `cudaFree`. Finding out why was the most useful thing this exercise produced.
- End to end, a 512x2048 MLP training step runs in **65 ms** on my CUDA backend versus
  **169 ms** on NumPy and **83 ms** on 6-thread PyTorch CPU. PyTorch on the same GPU does it
  in 4.8 ms, and the gap is almost entirely one fixable thing.

Everything below is reproducible with `python benchmarks/run_all.py`. Raw numbers are in
`benchmarks/results.json`, full tables in `benchmarks/RESULTS.md`, charts in `assets/`.

## How I measured it, and why that took a few tries

I care about this section because my first attempt produced numbers that were wrong in
ways I would not have caught by eyeballing them.

GPU kernel launches are asynchronous, so a host timer wrapped around one launch measures
how long it took to queue the work, not to do it. The harness enqueues a batch of
repetitions and synchronises the device once at the end, sizing the batch so each sample
lands near 5 ms. It reports the median of many samples rather than the mean, because these
distributions have a long right tail, and it records p10, p90 and the interquartile range
so the spread is visible instead of hidden.

There were 3 main bugs I had to fix along the way, each of which had produced a believable-looking but
wrong result:

1. **My calibration run was the cold run.** I used the first timed repetition both to size
   the batch and as the warmup. That first call pays for cuBLAS picking an algorithm, kernel
   modules loading, and first-touch page faults on a fresh buffer. It reported the RTX 3060
   as a 61 GFLOP/s card and host memory bandwidth as 0.2 GB/s. Now an untimed call comes
   first, and the batch size is derived from a second probe taken after warmup.
2. **A slow probe bought a short warmup.** Warmup length was derived from the probe, so when
   a threaded BLAS was still spinning up its thread pool the probe was slow, which bought a
   short warmup, which produced more slow samples. That is how numpy at N=512 once reported
   31 GFLOP/s when it really does 337. There is now a floor of 10 warmup repetitions
   regardless, and any sample set whose IQR exceeds a quarter of its median is taken again,
   keeping the most stable attempt and flagging anything still jittery.
3. **I measured the ceilings on an idle GPU.** The ceilings run first, when the card is
   sitting at its lowest clock, and a GPU needs a few seconds of sustained load to settle at
   its working frequency. That understated the bandwidth ceiling by 30% and made my own
   kernels look better than they are. The suite now holds the GPU under load for five
   seconds before measuring anything.

The last one matters most, because every "percent of peak" here is quoted against a ceiling
measured on this machine rather than off a spec sheet. The RTX 3060's spec fp32 number is
15.3 TFLOP/s at max boost. cuBLAS gets 7.1 TFLOP/s of that in practice. Comparing my kernel
to the spec sheet would let me claim a smaller gap to something nothing achieves, so I
compare against cuBLAS and note the spec figure separately.

Every backend is also checked against a NumPy reference before it is timed, because a
timing from a kernel that computes the wrong answer is worse than no timing. All seven
checks pass, with a maximum relative error of 8e-7 on GEMM and bit-exact compaction.

## GEMM: three kernels, two orders of magnitude

Square fp32 matmul, GFLOP/s, higher is better:

| N | naive C++ | tiled C++ | my CUDA | numpy (OpenBLAS, 6t) | torch CPU 1t | torch CPU 6t | cuBLAS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 2.0 | 5.0 | 313 | 293 | 124 | 281 | 1964 |
| 512 | 1.1 | 6.5 | 1362 | 337 | 127 | 340 | 5289 |
| 1024 | 0.4 | 7.3 | 1494 | 387 | 119 | 366 | 6599 |
| 2048 | - | 7.7 | 2805 | 467 | 124 | 409 | 7000 |
| 4096 | - | 7.9 | 3271 | 461 | 123 | 422 | 7219 |

The naive kernel is not just slow, it gets *worse* as the matrices grow: 2.0, then 1.1, then
0.4 GFLOP/s. That is the interesting part. Absolute speed being low is expected from three
nested loops. Speed dropping five-fold while the problem grows is a cache story. The inner
loop walks down a column of B, so consecutive accesses are N*4 bytes apart and each one
touches a different cache line. At N=256 the whole of B is 256 KB and lives comfortably in
L2, so the strided access is cheap. At N=512, B is 1 MB and no longer fits in this core's
512 KB L2, so every inner iteration goes out to L3. By N=1024, B is 4 MB and the access
pattern is also thrashing the TLB. Same code, same instruction count per element, five
times slower, purely because of where the data had to come from.

That is why the tiled kernel wins by more as the problem grows: 2.5x at N=256 but 19x at
N=1024. It copies 8x8 blocks into small aligned buffers and works out of those, so the
strided walk happens once per block instead of once per output element, and the inner
product becomes something the compiler can vectorise. The speedup is not really "tiling made
the arithmetic faster", it is "tiling stopped the memory system being the bottleneck", and
the effect scales with how badly the naive version was missing.

Still, the tiled kernel plateaus at 7.9 GFLOP/s, which is **6.5% of the 121 GFLOP/s a
single-threaded MKL reaches** on this chip. I am not going to dress that up. What MKL has
that I do not: hand-written AVX2 microkernels with software pipelining, a multi-level
blocking scheme that packs for L1, L2 and L3 rather than only for registers, prefetch
instructions placed by someone who has measured where they help, and per-architecture
tuning. Getting from 7.9 to even 40 would mean writing intrinsics rather than hoping the
compiler vectorises a nested loop, and that is the honest next step for this kernel.

The CUDA kernel is the one I am pleased with. It reaches 46% of cuBLAS at N=4096, and the
gap *narrows* with size, from 6.3x at N=256 to 2.2x at N=4096. That shape is a
launch-and-setup story: at small N the fixed per-op cost is a large fraction of the total,
and as the matrix grows the kernel gets to spend its time doing arithmetic instead. On the
roofline it sits far to the right of the ridge point at every size measured, meaning it is
compute-bound rather than bandwidth-bound, so the remaining 2.2x is genuine arithmetic
efficiency: cuBLAS uses larger tiles, vectorised 128-bit loads, and double-buffered
async copies that overlap the next tile's load with the current tile's math. Mine loads a
64x64 tile into shared memory, has each thread accumulate a 4x4 block in registers, and
does not overlap anything.

Being 27x faster than a single-threaded BLAS and 7.1x faster than a 6-core OpenBLAS is the
result I actually wanted from writing a GPU kernel by hand.

## Views are free. Making them contiguous is not.

An `NDArray` in nyangrad is shape, strides, an offset and a handle to flat memory, so
reshape, permute, broadcast and slicing all just rewrite metadata and hand back an array
pointing at the same buffer. Broadcasting is a stride of zero. None of it copies anything.

The bill arrives when a kernel needs contiguous input and `compact()` has to walk the
strided layout and write the elements out densely. That operation does no arithmetic at all,
so it is pure memory traffic, which makes it the cleanest possible look at how the backends
behave when bandwidth is the only thing that matters.

Effective bandwidth, counting the minimum traffic the operation requires, against a measured
ceiling of 297 GB/s on the GPU and 48.6 GB/s on the CPU:

| case | my CUDA kernel | % of roof | torch CUDA | my CPU kernel | % of roof |
| --- | --- | --- | --- | --- | --- |
| 4096² transpose | 25.3 GB/s | 9% | 129 GB/s | 2.0 GB/s | 4% |
| NHWC to NCHW permute | 27.6 GB/s | 9% | 220 GB/s | 0.8 GB/s | 2% |
| broadcast a row to 4096² | 15.7 GB/s | 5% | 311 GB/s | 1.7 GB/s | 3% |
| every other element, both axes | 26.9 GB/s | 9% | 209 GB/s | 2.7 GB/s | 6% |
| one streaming pass, already compact | **298 GB/s** | **~100%** | 298 GB/s | 16.9 GB/s | 35% |

The last row is the control, and it is the one that tells me where the problem is not. When
the data is already contiguous, my CUDA kernel moves it at 298 GB/s, which is the ceiling,
identical to what PyTorch manages. So the kernel can saturate this card. The elementwise
machinery, the launch configuration, the memory access pattern for the easy case: all fine.

Which means the 9% on the compaction cases is specifically about strided access. Two things
cause it. First, coalescing: when 32 consecutive threads in a warp read locations 16 KB
apart, the hardware cannot merge them into one transaction, so a warp that should cost one
memory transaction costs 32, and most of every 128-byte line fetched is discarded. Second,
every single element pays for its own index arithmetic, converting a flat thread id into a
strided offset with a division and a modulo per dimension inside the kernel. PyTorch avoids
the first by having its copy kernels reason about the layout and tile the transpose through
shared memory so that both reads and writes come out coalesced, and avoids the second by
specialising on dimensionality ahead of the launch. Neither is exotic. Both are on my list.

The CPU compaction numbers are worse in relative terms, 2 to 6% of the achievable
bandwidth, and for the same second reason: a scalar loop recomputing a strided offset per
element, with no vectorisation and no blocking. The 35% on the compact streaming case is a
fair reflection of a single-threaded scalar loop against a ceiling that assumes all six
cores.

## The bottleneck was not a kernel at all

Here is the result that changed how I think about the framework.

I noticed that a single elementwise op on a **one-element** tensor takes 67 microseconds on
my CUDA backend, against 12 for PyTorch. With one element there is no arithmetic to speak
of, so that number is pure overhead. My first assumption was that the autograd layer was
expensive: building a graph node, recording inputs, the Python bookkeeping. That assumption
was wrong. Doing the same operation at the bare array level, skipping autograd entirely,
costs the same. So the cost is underneath.

So I took one elementwise op apart. Microseconds, CUDA:

| step | µs | what it isolates |
| --- | --- | --- |
| launch the kernel into a reused buffer | 11.1 | the launch on its own |
| allocate and free a buffer, no kernel | 4.5 | the allocation on its own |
| **allocate, launch, free — what every op does** | **66.7** | the real thing |
| allocate, launch, keep the output alive | 12.4 | the same, minus the free |

The parts do not add up, and that is the whole finding. Launching costs 11 µs. Allocating
costs 4.5 µs. Doing both costs 67 µs, which is four times their sum. And if I keep the
output alive so that nothing is freed, the same work costs 12.4 µs — a **5.4x difference
from deleting nothing but the free**.

The reason is that `cudaFree` is a synchronising call. When an op allocates a fresh output
buffer, launches a kernel that writes to it, and then drops the last reference, the free
happens while the kernel is potentially still in flight, so the driver has to wait for the
device to drain before it can release the memory. Every op therefore ends with a full device
synchronisation. Launches can never overlap, the queue is never deep, and the GPU spends
most of its life idle waiting for the host to catch up. A framework whose every operation
allocates and frees its own output is, without meaning to, running fully serialised.

The same effect scales brutally with buffer size, because `cudaMalloc` itself is not free
either:

| buffer | my CUDA backend | torch CUDA | my CPU backend | torch CPU |
| --- | --- | --- | --- | --- |
| 0.26 MB | 3.8 µs | 2.1 µs | 1.9 µs | 1.2 µs |
| 4 MB | 554 µs | 2.0 µs | 1.9 µs | 1.2 µs |
| 67 MB | **3500 µs** | **2.1 µs** | 11.1 µs | 9.9 µs |

Allocating and freeing 67 MB on the device costs 3.5 milliseconds through the CUDA API and
2.1 microseconds through PyTorch's caching allocator, which is about 1700x. PyTorch's
allocator is not clever about arithmetic; it simply never returns memory to the driver, it
keeps a pool of blocks and hands them back out. That one design decision is worth more on
this workload than any kernel I could write. On the CPU side my allocation is at parity with
torch, which confirms the effect is specific to the device allocator rather than to my code
being slow in general.

This is why 89% of the wall time of my CUDA elementwise pass is not the kernel. The kernel
is at the hardware roof. The op around it spends nine tenths of its life allocating and
freeing.

## End to end: where it all lands

A 784 -> h -> h -> h -> 10 MLP with ReLU and softmax cross entropy, forward and backward,
median latency in milliseconds:

| config | nyangrad numpy | nyangrad CPU | nyangrad CUDA | torch CPU 6t | torch CUDA |
| --- | --- | --- | --- | --- | --- |
| batch 64, hidden 256 | 2.7 | 51.9 | 6.4 | 0.89 | 1.0 |
| batch 256, hidden 1024 | 40.2 | 1171 | 29.2 | 14.7 | 1.1 |
| batch 512, hidden 2048 | 169 | 7288 | **65.0** | 82.7 | 4.8 |

The ranking flips as the model grows, which is the most instructive thing in the table. At
batch 64 my GPU backend (6.4 ms) is *slower than my NumPy backend* (2.7 ms). At batch 512 it
is 2.6x faster than NumPy and beats 6-thread PyTorch on the CPU, at 473 GFLOP/s effective
throughput including the backward pass.

The small case is explained entirely by the previous section. I instrumented the backend by
wrapping the kernel module in a counting proxy, so these are real call counts rather than
estimates: one forward and backward pass through this model issues **112 kernel launches and
113 device allocations**. At the ~67 µs of serialised overhead an op costs in isolation, 112
of them comes to roughly 7.5 ms, against a measured step of 6.4 ms. The estimate slightly
overshoots, which only means the isolated microbenchmark is a little pessimistic compared to
a real step; the point stands that dispatch overhead alone accounts for essentially the
entire step. There is no room left for the arithmetic to matter. The GPU is idle almost the
whole time, waiting on a synchronisation caused by freeing a buffer it just finished writing.

Two other things the instrumentation turned up that I would not have guessed:

- **`compact` is the most-called kernel in a training step**, 32 times on CUDA and 77 times
  on the CPU backend, more than matmul and every elementwise op. The op layer inserts a
  compaction whenever it needs contiguous input for a reshape or to line up shapes for
  broadcasting. So a third to a half of all kernel launches in a training step are not
  computing anything; they are fixing layout. This is exactly the work a fusing compiler
  removes, and it is also why the 9%-of-roof compaction kernel matters more than its name
  suggests.
- **Every step does two host-to-device round trips.** My softmax loss builds its one-hot
  targets by copying labels to the host and the result back. On a GPU that is not just
  bandwidth, it is another forced synchronisation on the critical path of every single step.

The CPU backend at large sizes (7.3 seconds a step) is exactly what a single-threaded
7.9 GFLOP/s matmul predicts, and NumPy beating it by 43x is OpenBLAS being multithreaded and
vectorised. That comparison is fair and I would rather show it than not.

## What I would fix, in order

The measurements rank the work for me, which is the point of measuring.

1. **A caching allocator.** Reuse device buffers instead of calling `cudaMalloc` and
   `cudaFree` per op. The dispatch breakdown says this is worth about 5x on small ops
   immediately, by removing the implicit synchronisation and letting launches pipeline, and
   the allocator table says it is worth up to 1700x on large buffers. It is also the least
   clever item on this list, which is a lesson in itself.
2. **Coalesced, specialised compaction.** Tile the transpose through shared memory so reads
   and writes are both coalesced, and hoist the index arithmetic out of the per-element path
   by specialising on rank. There is a 10x sitting in the gap between my 27 GB/s and
   PyTorch's 220 on the same operation.
3. **Stop compacting so often.** A third of the launches in a step exist to fix layout.
   Teaching the op layer to let kernels read strided input directly, or fusing elementwise
   chains so intermediates are never materialised, removes the work rather than making it
   faster.
4. **Better GEMM microkernels.** For CUDA, wider vectorised loads and double buffering to
   overlap the next tile's load with the current tile's math; that is most of the remaining
   2.2x to cuBLAS. For the CPU, AVX2 intrinsics and cache-level blocking instead of
   register blocking alone, plus threading.

## What I took away from it

The thing I did not expect was how sharply the measurements disagreed with my intuition
about my own code. I assumed my CUDA elementwise kernel was mediocre and the autograd layer
was heavy. In fact the kernel is at the hardware bandwidth roof, the autograd layer is
nearly free, and the bottleneck was a memory-management decision I had never thought of as a
performance decision at all. I only found it because I measured the parts separately and
noticed they did not add up to the whole.

The other takeaway is how much of benchmarking is not the timing loop. Three of my early
results were confidently wrong: a 61 GFLOP/s GPU, a 0.2 GB/s memory bus, a 31 GFLOP/s
OpenBLAS. Every one of them came from measuring something cold and reporting it as steady
state, and every one looked plausible enough that I could have published it. Warmup,
synchronisation, ceilings measured on the machine rather than read off a spec sheet, and a
correctness check before every timing are not pedantry, they are the difference between a
benchmark and a number.

---

## Shorter version, for a post

I spent a while benchmarking nyangrad, the deep learning framework I built from scratch with
its own autodiff, strided array library, and hand-written C++ and CUDA kernels. Some
findings, on an RTX 3060 and a Ryzen 5 5600X, all fp32:

My CUDA matmul hits 3.3 TFLOP/s at 4096², which is 46% of cuBLAS on the same card and 27x a
single-threaded BLAS. Tiling the CPU version was worth 18x over three naive loops at
N=1024 — and the reason it was 18x rather than 3x is that the naive kernel gets *slower* as
matrices grow, from 2.0 down to 0.4 GFLOP/s, because its column walk falls out of L2. Tiling
did not speed up the arithmetic, it stopped the memory system being the bottleneck.

The result I did not expect: a one-element elementwise op costs 67 µs on my GPU backend
versus 12 for PyTorch. I assumed my autograd layer was heavy. It is not — the bare array op
costs the same. So I took the op apart. Launching the kernel costs 11 µs. Allocating the
output costs 4.5 µs. Doing both costs 67 µs. Keep the output alive so nothing gets freed and
it costs 12.4 µs.

`cudaFree` synchronises. Every op was allocating a fresh output, launching a kernel into it,
then freeing it while that kernel was still in flight, which made the driver wait for the
device to drain. Every operation ended in a full device sync, so launches never overlapped
and the GPU sat idle waiting on the host. A training step of my MLP issues 112 launches and
113 allocations; at 67 µs of serialised overhead each, that is the entire 6.4 ms step.
Allocating and freeing 67 MB costs 3.5 ms through the CUDA API and 2.1 µs through PyTorch's
caching allocator.

Meanwhile my elementwise kernel on contiguous data runs at 298 GB/s, exactly the bandwidth
ceiling I measured on the card and identical to PyTorch. The kernel was already done. The
bottleneck was a memory management decision I had never thought of as a performance
decision.

Two more things I only found by instrumenting the backend to count real kernel calls:
`compact` is the most-called kernel in a training step, more than matmul, because a third of
all launches exist only to fix memory layout rather than compute anything. And my softmax
loss was doing two host-to-device round trips per step, each one another forced sync on the
critical path.

Also worth saying: three of my first results were confidently wrong. I measured a 61
GFLOP/s GPU, a 0.2 GB/s memory bus, and a 31 GFLOP/s OpenBLAS, all because I timed cold
runs and reported them as steady state. Warmup, device synchronisation, ceilings measured on
the actual machine instead of a spec sheet, and a correctness check before every timing are
the difference between a benchmark and a number.
