

<p align="center">
 <h1 align="center">Compute Bound & Memory Bound</h1>
</p>


In any program, we need to do 2 things:
- Bring data from memory
- Perform computation on the data

Or we can said that: When discussing performance in a code segment, we consider two main concepts: **memory and compute**

<p align="center">
 <h4 align="center">What are memory and compute, and why are they so important?</h4>
</p>


<p align="center">
 <h3 align="center">Memory - Compute</h3>
</p>


- Compute: Refers to computational power, often measured using a popular metric called the **FLOPS rate (floating point operations per second)**. It quantifies a computer's performance in executing floating-point calculations in one second.

<p align="center">
  <img src="https://github.com/user-attachments/assets/621f8bf5-f2fa-4207-9c38-e4dd5a2bbd86" />
</p>


- Memory: This doesn’t refer to the total memory used but rather the memory bandwidth (GB/s), which is the rate at which data can be loaded or stored between memory and processing components.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4153b6c-4c6e-47b8-9de3-71a589ee40d4" />
</p>


<p align="center">
 <h4 align="center">How do we determine a good FLOPS rate or Memory Bandwidth?</h4>
</p>


<p align="center">
 <h3 align="center">Desired Compute to Memory Ratio (OP/B)</h3>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/f2c81d4a-7d2c-4c39-bbe4-aa8372b6494e" />
</p>

This is a critical metric for balancing a computer’s processing power and its load/store capabilities in memory.

```
Why balance them? - Balancing ensures efficient hardware resource utilization, avoiding performance bottlenecks.
```

- If OP/B is low: The system is handling heavy computational tasks, but the computational power is limit. This leads to a **compute-bound scenario.**
- If OP/B is high: The system cannot supply enough data for processing, leading to data starvation or a **memory-bound scenario.**


<p align="center">
 <h4 align="center">What are Compute/Memory Bound, and how can we identify and resolve them? </h4>
</p>

- Compute-bound: Occurs when a computer’s performance is limited by its computational capacity. This is common when executing complex calculations.
- Memory-bound: Occurs when performance is limited by the ability to access data from memory. This usually happens when handling large amounts of data load/store operations.


<p align="center">
 <h4 align="center">Identifying Compute/Memory Bound in Your Code </h4>
</p>



<p align="center">
 <h3 align="center">Speed Of Light Throughput (SoL) </h3>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/707de8ea-7c43-4a1c-96a5-9a664d502015" />
</p>

In **Nsight Compute** can help us to identify whether **Compute/Memory bound** by using **SoL**

**SoL**: Achieved % of utilization with respect to maximum, represents the level of activity of the computer hardware, not the performance of the code.

Our goal is to ensure that compute and memory resources are utilized evenly without a significant imbalance.

- Balanced utilization prevents bottlenecks, where one resource (compute or memory) becomes a limiting factor for overall performance.
- Uniform usage also helps maximize the efficiency of the hardware, achieving a closer approximation to the theoretical peak performance


<p align="center">
  <img src="https://github.com/user-attachments/assets/78516478-822f-4e7a-b606-43d7f62c846a" />
</p>


- Latency (M & SM < 60):
  
As explained earlier, SoL reflects the level of activity of the computer. However, in this case, we can see that both M (memory) and SM (compute) are not being used at their full potential. This suggests that the system isn’t fully utilizing the available resources, which could mean that the workload is not heavy enough to stress the hardware.

- Compute Bound (SM > 60 || M < 60):

This situation suggests that while the system has enough computational power to handle the data, it’s being overused in complex calculations, leading to a compute-bound scenario.


- Memory Bound (SM < 60 || M > 60):

This leads to data starvation, where the system cannot provide enough data to the computation resources in time, even though the calculations themselves may be simple.

- Compute/Memory Bound (SM & M > 60):

In this case, the system is operating at a high capacity for both compute and memory. It’s important to monitor performance carefully to avoid potential bottlenecks.



**In summary, each of these situations reflects a different type of resource imbalance:**

- Latency: Low usage of both resources.
- Compute Bound: Excessive compute usage with underused memory.
- Memory Bound: High memory usage with underused compute.
- Balanced Usage: Optimized usage of both compute and memory or encountering performance limits with both resources.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fc66d5e0-8ec9-4b6b-824d-8e68ea1a9dd3" />
</p>

- SM: Inst Executed Pipe Lsu(%) : if % high is SM instead of compute then the load/store unit will be expensive in time
- SM: Pipe Fma/Alu Cycles Active (%): %SM compute


<p align="center">
 <h3 align="center">Roofline chart </h3>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6b04564-aa62-41dd-a730-acbd146c19ca" />
</p>


Before explaining in depth about the roofline chart, I will go over the definitions you need to know

**Arithmetic Intensity** is a metric that represents the efficiency of utilizing both computation resources (FLOP - Floating Point Operations) and memory bandwidth (bytes of data transferred).

<p align="center">
  <img src="https://github.com/user-attachments/assets/95febfe8-7646-436e-be51-4b1bf66aa863" />
</p>


As we know, math is much faster than memory, so we need to balance the ratio appropriately to avoid memory/compute bound cases. Each computer will have a different ratio, and to determine this, you can click on the square as shown in the image.


<p align="center">
  <img src="https://github.com/user-attachments/assets/0d53a2e5-2e28-41ad-8900-463df642629a" />
</p>

We will analyze this diagram in more detail.


<p align="center">
  <img src="https://github.com/user-attachments/assets/e95a21a3-4e55-4bdf-a4a6-2f4f37b08e15" />
</p>


- Peak FLOP/s: The maximum computational speed that a computer can achieve.

- Bandwidth GB/s: The rate at which the computer can load/store memory, reaching its peak at the intersection of the red and blue lines. This point is called the key point or knee point.

**Key point (knee point):** The point where there is a transition between two stages:

- Memory bound stage.
- Compute bound stage.

In theory: If we achieve the key point ratio (as shown in the image, AI = 0.55), it means our code is nearly perfect (balanced between math and memory).

In practice: Reaching a point along the diagonal line, as shown in the image, is already a very good outcome.

Bottleneck situation:


<p align="center">
  <img src="https://github.com/user-attachments/assets/074756f5-34b4-4282-972d-fc6ef0c676bd" />
</p>

```

P (FLOP/s): Represents the speed at which a task can be executed.

P (peak): The maximum computational speed that the computer can theoretically achieve.

I . b (FLOP/byte * byte/s): The actual speed required to run a specific piece of code.

```

When we use the min function to determine whether the system is compute-bound or memory-bound:

If **P(peak) min ==> compute-bound:** This means the actual speed to run the code is higher, but our computer is limited to the peak computational speed and cannot go beyond that. Solution: Use a larger unit size.

If **I . b min ==> memory-bound:** This indicates that we haven't fully utilized the computer's capacity. Solution: Use coarsening.

Thus, through these two aspects, we can determine whether our code has issues with computation or with load/store data. In the following sessions, I will guide you on how to specifically address each case.













