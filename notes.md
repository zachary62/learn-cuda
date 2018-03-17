## Is Warp Execution Serial or Parallel?

Different blocks can be mapped to different SM's and hence executed in parallel. But, internally, blocks consist of warps which are scheduled for execution on an SM one at a time (on 1.x devices). However, the graphics hardware can switch between different warps with 0 overhead (owing to static register allocation). Therefore usually instructions from different warps (and possibly from different blocks) exist in the SM's pipeline at different stages.

Active warps are those that are ready to execute, i.e. not waiting on a barrier, memory access and do not have register dependencies (like read-after-write). I am not sure how the hardware chooses the next warp to execute. Propabably warps are prioritized by "age" (waiting time) and other factors to prevent starvation.

"zero overhead" just means that context switch between warps in the SM is almost for free (does not induce additional cost). This is because each thread has its own copy of physical registers (static allocation). Hence, when the hardware needs to switch from one warp to another, there is no need to save/restore a register set from the memory as opposed to other architectures like x86 where there is just one set of registers used by all threads
