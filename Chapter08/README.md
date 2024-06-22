<p align="center">
 <h1 align="center"></h1>
</p>

<p align="center">
  <img src="" />
</p>

<p align="center">
 <h1 align="center">Occupancy Part 2</h1>
</p>

In the first part, I introduced occupancy; in this second part, I'll delve deeper into how to improve achieved occupancy.

Before we dive into the lesson, let me explain two important concepts for this article:

- Tail Effect: If the total number of threads isn't divisible by a warp (32), the tail effect occurs. The tail effect is the remaining threads that run last in the warp. The fewer the remaining threads, the more significant the tail effect, leading to slower program execution.
    - For example, if we have data with N = 80 and 40 threads, we would need two warps to assign 40 threads, meaning the second warp would only use 8 threads, wasting resources. Given N = 80, we would need 4 warps when only 3 warps should suffice.

- Waves: Another term for warp.



<p align="center">
 <h1 align="center">Achieved Occupancy</h1>
</p>


**Theoretical occupancy** gives us the upper bound of active warps per SM. However, in practice, threads in blocks may execute at different speeds and complete their executions at different times. Thus, the actual number of active warps per SM fluctuates over time, depending on how the threads in the blocks execute.

This brings us to a new concept - **achieved occupancy**, which addresses this issue: **achieved occupancy** looks at warp schedulers and uses hardware performance counters to determine the number of active warps per clock cycle.

You can refer to the [warp schedulers](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter06) article for a better understanding



<p align="center">
 <h1 align="center">Causes of Low Achieved Occupancy</h1>
</p>

Achieved occupancy cannot exceed theoretical occupancy, so the first step toward increasing occupancy should be to increase theoretical occupancy by adjusting the limiting factors, for example using cudaOccupancyMaxPotentialBlockSize to gain 100% Theoretical occupancy. The next step is to check if the achieved value is close to the theoretical value. The achieved value will be lower than the theoretical value when the theoretical number of active warps is not maintained for the full time the SM is active. This occurs in the following situations


<p align="center">
 <h1 align="center">Unbalanced workload within blocks</h1>
</p>

If the warps within a block do not execute simultaneously, we encounter an Unbalanced issue within the block. This can be understood as having too many threads in one block, leading to some warps being stalled because each thread requires a certain number of registers. When we use many threads in one block, it results in less utilization of the last warps, leading to the appearance of the tail effect.

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/de4b8ac8-009c-48eb-955d-bb9961a935a9" />
</p>


Instead of using the maximum number of threads in one block (1024 threads), consider and choose a number that is appropriate for the data.

<p align="center">
 <h1 align="center">Unbalanced workload across blocks</h1>
</p>

If the blocks do not execute simultaneously within each SM, we also encounter an Unbalanced issue within the grid. This can be understood as having too many blocks per SM, leading to stalls. We can address this by adjusting the number of blocks in the grid or using streams to run the kernels concurrently.

You can review the [streaming](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter11) section to understand this better. To answer the question of how many streams are appropriate, divide in a way that minimizes the tail effect of each thread in the block.


<p align="center">
 <h1 align="center">Too few blocks launched</h1>
</p>

We are not utilizing the maximum capacity of the SMs (the number of blocks used is less than the number of blocks that can run simultaneously in an SM). The phenomenon of full waves - full warps occurs when the total number of SMs multiplied by the total number of active warps per SM is achieved.

For example, if we have 15 SMs and 100% theoretical occupancy with 4 blocks per SM, the full waves would be 60 blocks. If we only use 45 blocks, we would only achieve 75% achieved occupancy.




































