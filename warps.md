## Warps

Each thread block is further subdivided into 32-thread warps. For example, if a grid is of size 16x16, there are a total of `16*16 / 32 = 8` warps in the block. Warps are the scheduling unit in SMs, i.e. they execute in SIMD manner (single instruction, multiple data). To organize the block into warps, the block is first linearized into a 1D row-major format and then consecutive chunks of 32 are partitioned into consecutive warps.

Again, all threads in a warp execute the same instruction at any point in time. This means that we usually want threads in a warp to do the same thing to avoid divergence.

Occurs when threads in a warp take different control paths. The execution of threads taking different paths are serialized in current GPUs.

* example of CD: `if (threadIdx.x > 2)`
* example of no CD: `if (blockIdx.x > 2)`

## Is Warp Execution Serial or Parallel?

Different blocks can be mapped to different SM's and hence executed in parallel. But, internally, blocks consist of warps which are scheduled for execution on an SM one at a time (on 1.x devices). However, the graphics hardware can switch between different warps with 0 overhead (owing to static register allocation). Therefore usually instructions from different warps (and possibly from different blocks) exist in the SM's pipeline at different stages.

Active warps are those that are ready to execute, i.e. not waiting on a barrier, memory access and do not have register dependencies (like read-after-write). I am not sure how the hardware chooses the next warp to execute. Propabably warps are prioritized by "age" (waiting time) and other factors to prevent starvation.

"zero overhead" just means that context switch between warps in the SM is almost for free (does not induce additional cost). This is because each thread has its own copy of physical registers (static allocation). Hence, when the hardware needs to switch from one warp to another, there is no need to save/restore a register set from the memory as opposed to other architectures like x86 where there is just one set of registers used by all threads.
