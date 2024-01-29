<p align="center">
 <h1 align="center"> NVIDIA Compute Sanitizer Part 1 </h1>
</p>

In this article, I will guide you on how to use the NVIDIA Compute Sanitizer, a fantastic tool to support those who are new to CUDA.

For those who are already very familiar with CUDA, NVIDIA Compute Sanitizer may not be of much help, but it's still better to know about it than not.

NVIDIA Compute Sanitizer helps us check for four important errors that CUDA beginners often encounter:
- **Memcheck** for memory access error and leak detection
- **Racecheck**, a shared memory data access hazard detection tool
- **Initcheck**, an uninitialized device global memory access detection tool
- **Synccheck** for thread synchronization hazard detection

This is a simple code snippet (adding two vectors) to analyze four cases.

```
#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
        c[tid] = a[tid] + b[tid];
}

int main() {
    int n = 10;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  
    vectorAdd<<<1, ?>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    free(a);
    free(b);
    free(c);

    return 0;
}
```

When you start coding in CUDA, one common mistake is using too few or too many threads compared to the data, and this can lead to a **"undefined behavior"** bug. It may work without giving any error messages. In a larger program, this can result in logic issues and have a significant impact on memory allocations.

<p align="center">
 <h1 align="center"> Initcheck </h1>
</p>

Here, I will provide a specific example.

```
vectorAdd<<<1, ?>>>(d_a, d_b, d_c, n)--> vectorAdd<<<1, 9>>>(d_a, d_b, d_c, n)
```

As you can see, we have N = 10 but are using only 9 threads for processing, and this leads to memory leakage. We will use this command to profile.

```
compute-sanitizer --tool initcheck --track-unused-memory yes --show-backtrace no
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/dc04a3ce-10d9-4d14-83ae-6ca897295106" />
</p>

We use N = 10 (int), so the total bytes are 40 bytes, and we use 9 threads, leaving 10% of memory unused.

And here are the results when we fix it to use 10 threads.

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/f65f9225-b67b-4871-b6d1-7549e0f6072e" />
</p>


<p align="center">
 <h1 align="center"> Memcheck </h1>
</p>


Above, we discussed the case of using fewer threads. Now, let's consider the case of using more threads.

```
vectorAdd<<<1, ?>>>(d_a, d_b, d_c, n)--> vectorAdd<<<1, 11>>>(d_a, d_b, d_c, n)
```
As you can see, we only initialize enough for 10 threads to operate, which leads to the 11th thread suffering from **out-of-bounds array access**. We will use this command to profile.

```
compute-sanitizer --tool memcheck --show-backtrace no
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/a02a9d7b-24b8-49d6-9754-1ea0780052fe" />
</p>


To explain it simply, when we use cudaMemcpy to copy to the GPU, the 11th element will fail because we only initialize enough for 10 elements. This failure happens at thread(10, 0, 0), which means the 11th thread is accessing data beyond the boundary, leading to **"undefined behavior."**

The solution is to either adjust to use 10 threads or **add boundary checks.**

```
                                   if (tid < n) {
  c[tid] = a[tid] + b[tid] --->        c[tid] = a[tid] + b[tid];
                                               }
```

Additionally, if you notice, in this code snippet, we are missing the **cudaFree**, which will also lead to **memory leaks**. We will use this command to profile.

```
compute-sanitizer --tool memcheck --leak-check=full --show-backtrace no 
```
<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/1d703c57-f071-49d2-b04c-219e875db7ee" />
</p>


The remaining two errors, Synccheck and Racecheck, will be discussed later after we cover atomic functions and data hazards.
