<p align="center">
 <h1 align="center"></h1>
</p>

<p align="center">
  <img src="" />
</p>


In this article, I will introduce three critical concepts in profiling: Bandwidth, Throughput, and Latency.

<p align="center">
 <h1 align="center">Bandwidth - Throughput - Latency</h1>
</p>

When evaluating a piece of code or a program, three important concepts need to be considered: **bandwidth, throughput, and latency**. However, it is easy to get confused when only one of these pieces of information is provided without the others, leading to an inaccurate assessment of performance. Since different computers can have varying latency or bandwidth, providing only one piece of information will not reflect the true performance of the code

<p align="center">
 <h1 align="center">Latency</h1>
</p>

**Latency (s):** is the time taken to complete a task. An extremely important note is that profiling (e.g., using **cudaEvent_t start, stop**) can affect performance, so profiling code should be removed during the final run of the program.

Instead of using cudaEvent_t start, stop to check latency, we can use Nsight System with the command:

```
nsys profile -o timeline --trace cuda,nvtx,osrt,openacc ./a.out
```

This way, everything running on the GPU will be measured in detail.



<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/ba59a142-c63b-478a-b01c-16323de9a159" />
</p>

Based on the article [Introduction to Nsight Systems - Nsight Compute](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter01) we can identify that our code needs improvement in data allocation for the GPU (cudaMalloc).

<p align="center">
 <h1 align="center">Bandwidth</h1>
</p>

**Bandwidth (GB/s):** represents the data transfer speed.

When discussing bandwidth, we refer to two concepts:

- **Theoretical Peak Bandwidth:** The theoretical ideal speed for data transfer.
  
<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/8a158e65-3b89-4a70-9b58-99e6dbb10840" />
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/108be482-7587-4e55-8877-5a5a017f711b" />
</p>

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/468f077c-0214-4f25-989b-ade4e927362f" />
</p>

```
Divide by 8 to convert from bits to bytes.

Divide by 10^9 to convert to GB/s.

DDR (Double Data Rate): multiply by 2.

SDR (Single Data Rate): multiply by 1.
```

To determine DDR or SDR, use the command:

```
sudo lshw -c memory
```

Since my machine is DDR:

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/82a6601c-6691-4ea7-b701-66db304c51be" />
</p>


- **Effective Bandwidth:** The actual data transfer speed of the kernel.


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/58a2074b-3507-4a71-9dd4-76a6667381b1" />
</p>


- R(B): number of bytes read by each kernel

- W(B): number of bytes written by each kernel

- t(s): latency



<p align="center">
 <h1 align="center">Code</h1>
</p>


We'll implement this: y[i] = a*x[i] + y[i] with N = 20 * (1 << 20)


```
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
```

```
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("time: %f\n", milliseconds);
  printf("Effective Bandwidth (GB/s): %f", N*4*3/milliseconds/1e6);
}
```

 using the formula R + W: N * 3 ( read a + read y + write y) & N * 4 (1 float = 4 bytes )

 and this is the output

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/c72e666d-bd45-4030-8901-eb54de1af329" />
</p>

But if we profile using Nsight compute with this command:

```
ncu -o profile --set full ./a.out
```

We'll see that

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/d1525faa-8393-46cd-b1e5-889967e1404c" />
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/efd02ed4-74e0-441a-88fa-dde36061a8dd" />
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/b8329288-1a06-42f0-8a22-c582ff4b7861" />
</p>

From here we can calculate that

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/5ccd6f6f-f6c4-4eff-8943-bc94d4fb9591" />
</p>


As I mentioned above, if we profile using cudaEvent_t start, stop, it will affect the code's performance from **91 GB/s to 87.8 GB/s.**

We can also reverse calculate to determine the Theoretical.

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/c25524b6-2435-4ba6-97ac-b16c2e340695" />
</p>


The slight deviation from the formula above (**96 GB/s to 95.8 GB/s**) is due to factors such as memory and kernel impact.

From this, we can conclude that our code is **very efficient in terms of data transfer**, as there is **no significant gap between Theoretical and Effective bandwidth.**

You can determine the effective bandwidth more quickly using the command:

```
Load: ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second ./a.out
Store: ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/133b404a-416a-4c20-bce3-0646487af5a6" />
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/91e0651e-6c52-4642-a49b-3f328fcd3352" />
</p>

We can see that 60.68 + 30.40 = 91.08 GB/s, which is close to 96 GB/s, indicating that the code is efficient.


<p align="center">
 <h1 align="center">Computational Throughput</h1>
</p>

**Throughput(GFLOP/s):** refers to the number of Floating Point Operations (FLOPs) a kernel can perform in one second.

`A FLOP is a floating point operation, which includes basic arithmetic operations such as addition, subtraction, multiplication, division, as well as more complex operations like square roots, sine, cosine, etc.`

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/d6a4456e-cafe-46e4-a2fd-2cc475e7b434" />
</p>

The question posed is whether to improve **bandwidth or throughput**, as improving one might affect the other. How can we determine the optimal value?

`Increasing bandwidth ==> more data is read/written ==> the compute increases ==> throughput decrease and opposite`

To answer this question, we can use a technique called the **roofline chart**, which I will discuss in future articles.

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/bbd32c83-0954-4f29-891a-1c74917f24e0" />
</p>






























