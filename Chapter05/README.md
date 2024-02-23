

Global memory is the largest memory BUT also the slowest on the GPU, so in this article we will analyze what factors lead to **"low performance"** as well as how to fix them. Before diving into this, it is recommended to review articles on [GPU memories](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter05) and [their utilization](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter06) to better understand the context.

<p align="center">
 <h1 align="center"> Global Memory Coalescing </h1>
</p>


Before we dive into the lesson, let's start with an example:

Imagine you have a task to distribute candies and cakes to children, each with different preferences. Instead of waiting for their turn to come up and asking what they like, which can be time-consuming (in terms of both asking and fetching the respective item), you decide to organize them by preference from the start: those who choose cakes on the left and candies on the right. This way, the distribution process is optimized.

When discussing global memory access, three key concepts often come up:

- **Coalescing:** This is the process by which **threads within the same warp** access memory simultaneously, optimizing memory access by reducing the number of necessary accesses and speeding up data transfer (similar to the candy and cake distribution, where instead of asking each time, it's already known what to give out, leading to cache hits).
- **Alignment:** This relates to organizing data in memory optimally to ensure memory accesses are as efficient as possible, minimizing unnecessary data reads and enhancing processing performance (like organizing children by their preference for cakes or candies on different sides to avoid confusion during distribution).
- **Sector:** Refers to the basic unit of memory that can be accessed simultaneously in a single access, clarifying the scope and method by which data is retrieved or written to memory.
Though these are three distinct concepts, they share a common goal: optimizing access to a large memory space.

**In summary, coalescing is about accessing memory in the most optimal way possible (the fewer accesses, the better), alignment involves arranging data optimally, and a sector is the unit of each access.**

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/0d1f3e00-36a7-4614-8804-dca5d7683aaf " />
</p>

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/130b4712-2763-4ed6-8b30-925c565550c2" />
</p>


<p align="center">
 <h1 align="center"> Code </h1>
</p>


I will demonstrate a simple piece of code using 32 blocks ( 32 threads / block ) and elements (number of elements) = 1024.

<p align="center">
 <h1 align="center"> Coalescing </h1>
</p>


```
__global__ void testCoalesced(int* in, int* out, int elements)
{
int id = blockDim.x * blockIdx.x +threadIdx.x;
out[id] = in[id];
}
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/3cbf36db-1203-4a99-987b-6c0fab588b18" />
</p>


And we will profile the above code:

- **global load transactions per request: the smaller, the better (this is about copying chunks --> checking coalescing)**

```
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/8d3dfa95-cf32-4e07-a141-4a3e52f4b1a0" />
</p>

- **global store transactions per request: the smaller, the better (this is about copying chunks --> checking coalescing)**

```
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/88146a08-3b35-4257-a8d7-326e48585e76" />
</p>



- **global load transactions: (compare to see which kernel has coalescing || the smaller, the better).**

```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/9539f644-c68e-49c2-9098-12545ebc2ff0" />
</p>



- **global store transactions:(compare to see which kernel has coalescing || the smaller, the better).**

```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/6ea4d22a-9564-4b48-acd7-d16ce014b173" />
</p>


`The reason "the smaller, the better" applies is because it's akin to distributing candies; the fewer times we need to exchange cookies for candies, the quicker the distribution process. Here, sector/request means that for each request, we only use 4 sectors, totaling just 256 sectors (load and store).`

`It's important to note that "sector" here does not refer to the number of elements processed per request but to the number of simultaneous data storage access operations the computer performs to process a request. The fewer the accesses, the faster it is (hit cache).`


<p align="center">
 <h1 align="center"> Mix but in cache line </h1>
</p>


```
__global__ void testMixed(int* in, int* out, int elements)
{
int id = ((blockDim.x * blockIdx.x +threadIdx.x* 7) % elements) %elements;
out[id] = in[id];
}
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/fdd4a53d-e925-4bb1-aa4c-273277f9e754" />
</p>



Here, we profile the same:

- **global load transactions per request: the smaller, the better (this is about copying chunks --> checking coalescing)**

```
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/adb7634f-2ffa-42ec-b8d2-d743cb690240" />
</p>


- **global store transactions per request: the smaller, the better (this is about copying chunks --> checking coalescing)**

```
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/fbf7ecdc-0ada-4bbe-ace7-097648acb6f8" />
</p>



- **global load transactions: (compare to see which kernel has coalescing || the smaller, the better).**

```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/45dc3e18-dedc-48c4-a27f-e92196c0e15a" />
</p>



**global store transactions:(compare to see which kernel has coalescing || the smaller, the better).**

```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./a.out
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/bc908490-2a0c-48ae-b79f-b24fabc7d119" />
</p>


As I mentioned, even though it still resides within the cache line (meaning the threads do not exceed the array space), because it is not coalesced (not in order, such as cookies first then candies, or vice versa), it results in more sectors/request, leading to slower performance.


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/ff57dc22-781b-498c-b587-12324b7a4c0e" />
</p>

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/6aa74fe7-bae7-4346-8f39-7e8023c45800" />
</p>


BUT IF YOU PROFILE FULLY (meaning to output to a .ncu-rep file for use with Nsight Compute, here is the command line)

`One note is that I will not delve too deeply into analyzing Nsight Compute but will leave it for a later article.`

```
ncu --set full -o <tên file> ./a.out
```

**And you will notice a somewhat strange point:**

<p align="center">
 <h1 align="center">Coalescing </h1>
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/b5c54329-0059-454f-b55e-654523ea6205" />
</p>

<p align="center">
 <h1 align="center">Mix </h1>
</p>


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/238d0766-057e-423d-883d-7a341b7952df" />
</p>

Why is the Coalescing throughput (GB/s) lower than Mix and the L2 cache hit rate lower, but the total time is faster?

Here (as I speculate), the computer optimizes for us: meaning for a certain amount of bytes, it will optimize what the transfer speed needs to be. It's not always the case that higher is better because if it's too high, it can lead to:

- When the data transfer rate is too high, it may cause congestion, reducing data transfer efficiency.
- A high data transfer rate may also consume more energy.
- In some cases, a high data transfer rate does not significantly benefit, for example, when transferring small files.
  
`It's like shopping; the most expensive option isn't always the best, and sometimes it depends on our needs.`

**Therefore, using more GB/s leads to a higher hit rate.**

**In summary: In this article, you have learned how to analyze and optimize when using global memory (and from what I've researched, 4 sectors/request is best ==> meaning we achieve coalescing when sector/request = 4).**


<p align="center">
 <h1 align="center">Exercise </h1>
</p>

- Try to code a case with an offset and profiling it

  <p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/8a74800d-29b0-4fe8-866e-8b29ca7976c2" />
</p>

In the picture above, the offset is 2, and having an offset leads to going out of the cache line (meaning instead of using 1024 * 4 bytes (since it's an int) for an array, here we use 1024 * 2 * 4 bytes).

- An interesting question: **(WE STILL USE GLOBAL MEMORY)** Although it is coalescing, we can still improve, so before improving, what is the reason for its slowness?
  
Hint:
  - memory bound (not yet fully utilizing the computer's capabilities)

  <p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/20e2848a-6902-47ec-843d-cecab7d4e49d" />
</p>

  - datatype between int ( 4 bytes ) and int4 ( 16 bytes )






