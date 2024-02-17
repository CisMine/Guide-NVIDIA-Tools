

In this article, I will continue discussing how to use the NVIDIA Compute Sanitizer. Please read these articles: [NVIDIA Compute Sanitize Part 1](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter03), [Data Hazard](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter12) before reading this one.

<p align="center">
 <h1 align="center">NVIDIA Compute Sanitizer </h1>
</p>

Following up on Part 1, Part 2 will cover the remaining two tools:

- **Racecheck**, a shared memory data access hazard detection tool

- **Synccheck** for thread synchronization hazard detection

<p align="center">
 <h1 align="center">Racecheck </h1>
</p>

As NVIDIA has mentioned about the [NVIDIA Compute Sanitizer](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/), Racecheck is used to check for hazards when using shared memory. So, if you test on global memory, it will not yield any results.

<p align="center">
 <h1 align="center">Code </h1>
</p>

```
__global__ void sumWithSharedMemory(int* input) {
    __shared__ int sharedMemory[4];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedMemory[tid] = input[i];

    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        // __syncthreads(); -----> barrier

        if (tid % (2 * stride) == 0) {
            sharedMemory[tid] += sharedMemory[tid + stride];
        }
    }

    printf("blockIdx.x=%d --> %d\n", blockIdx.x, sharedMemory[tid]);

}
```
This code snippet is identical to the one in the [Data Hazard](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter12) article, and here is how it works.

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/baec13cf-42a8-4fa4-b139-86ca8e710094" />
</p>

The only difference is that instead of using global memory, here we use shared memory.

Shared memory is a topic I will discuss separately since it is a very important concept when talking about CUDA. So, in this article, you only need to understand that instead of using global memory to perform the addition 1+2+3+4 in parallel, we use shared memory.

And now we use the NVIDIA Compute Sanitizer to check for data hazards with a command line.

```
compute-sanitizer --tool racecheck --show-backtrace no  ./a.out 
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/d004f4a3-1926-4ee0-abcc-95bba036f342" />
</p>


Here, you will find a surprising fact that the result is still correct even though there has been a data hazard. As I mentioned in the [previous article](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter12) about the phenomenon of **"undefined behavior,"** this is exactly it. We cannot determine whether it will cause an error or not. It may be that my machine produces the correct result, BUT yours might differ ==> thus, the phenomenon of **"undefined behavior"** can be quite troublesome.

At this point, if we use __syncthreads(), it will solve this problem.

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/8ab78146-26d5-4040-b58e-1f70b19e923d" />
</p>


Regarding Synccheck, from my tests and searches through NVIDIA's blog, I noticed they didn't mention anything about the code, so I cannot provide an illustration for you. I will skip this part, but I will add it later if I find any relevant information (if you find something, please comment below).


<p align="center">
 <h1 align="center">Exercise </h1>
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/38132a72-6522-420c-8cd5-4612fb5c78c2" />
</p>

The actual error in our case is just 4 data hazards (N = 4), but why does the image above show us having 2 data hazard errors (4 and 8)?

Hint: 1 data hazard = 1 read or 1 write.

Is 4 data hazards comprised of 4 reads or 4 writes?

And is 8 data hazards comprised of 8 reads, 8 writes, or 4 reads and 4 writes?
