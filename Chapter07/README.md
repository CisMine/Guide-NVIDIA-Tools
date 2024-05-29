


In lesson 6, I discussed the issue of **how to select the suitable number of threads.** In this article, I will share a quite common method to determine this. Many of you might wonder **why we don't just simplify the problem by running multiple cases with different thread counts to determine the appropriate number.** This approach is only suitable if your code is simple because, if it is complex, each run will take a long time. Therefore, running multiple cases to choose the appropriate number of threads is not a good choice.

<p align="center">
 <h1 align="center">Occupancy </h1>
</p>

Before diving into the article, let me give an example to help you understand what occupancy is and its utility.

For instance, we have 6 workers and 6 tasks. The simplest way to distribute the work is to assign each worker one task. However, if each worker has the capability to handle 3 tasks simultaneously, then we only need 2 workers for the 6 tasks. **This results in hiring fewer workers, which costs less money, and the number of tasks is always greater than the number of workers, so optimizing workers for the tasks is necessary.**

Here, the workers are threads and the tasks are data. The question arises: how do we determine how many tasks each worker can handle (how much data a thread can process)? This is where NVIDIA's Occupancy metric comes into play.

**Occupancy is used to determine the optimal number of threads to be used in a kernel to achieve the highest performance.**


<p align="center">
 <h1 align="center"> Code </h1>
</p>

With N = 1000000

```
__global__ void square(int *array, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        array[idx] *= array[idx];
    }
}
```

In normolly, we'll use like this

```
    blockSize = 1024;
    gridSize = (N + blockSize - 1) / blockSize;
```
However, whether the number 1024 is optimal or not, we need to profile to check the occupancy.

```
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/526ba6e8-e398-4af1-ad8e-f3ea427a1157" />
</p>


It can be seen that using 1024 threads is a waste of resources (because we are using the maximum number of threads per block, but the occupancy is only 53.73%) ==> This indicates that one thread can handle more than one task.

NVIDIA has created a function to determine the appropriate occupancy.

```
template<class T>
__inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int    *minGridSize,
    int    *blockSize,
    T       func,
    size_t  dynamicSMemSize = 0,
    int     blockSizeLimit = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMem(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit);
}

minGridSize     = Suggested min grid size to achieve a full machine launch.
blockSize       = Suggested block size to achieve maximum occupancy.
func            = Kernel function.
dynamicSMemSize = Size of dynamically allocated shared memory. Of course, it is known at runtime before any kernel launch. The size of the statically allocated shared memory is not needed as it is inferred by the properties of func.
blockSizeLimit  = Maximum size for each block. In the case of 1D kernels, it can coincide with the number of input elements.
```

And here are the results when using Occupancy to determine the number of threads

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/dc650b57-f359-4007-8da6-1ae6a2c2902a" />
</p>



<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/4650f8ea-88f2-40e2-aad3-57df0146501f" />
</p>

**One small note is that each computer has different configurations, leading to different numbers of threads needed to achieve 100% occupancy.**

Here, you might wonder why there are two different occupancy values: one is 74.89% and the other is 100%, even though the same code is being used.

This introduces a new concept called **Theoretical Occupancy vs. Achieved Occupancy.** You can understand these simply as the theoretical value expected vs. the actual value when the code runs.

The reason why Theoretical Occupancy and Achieved Occupancy yield different results is that when the code runs, threads are influenced by many other factors.


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/9683f65c-6f76-41b4-93a0-92294cb116e4" />
</p>


**We should focus on Achieved Occupancy rather than worrying too much about Theoretical Occupancy.**
Example:
- Theoretical 100% and Achieved 50%
- Theoretical 80% and Achieved 70%

In this case, we should choose the second scenario.

In the upcoming lessons, I will guide you through methods to increase Achieved Occupancy.
















































