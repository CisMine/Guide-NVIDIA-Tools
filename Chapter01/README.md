<p align="center">
 <h1 align="center">Introduction to Nsight Systems - Nsight Compute </h1>
</p>

In this article, I will provide a brief introduction to Nsight Systems and Nsight Compute, giving you an overview of which tool to use for your specific needs.

Please note that this article serves as a high-level introduction to these two tools and does not delve into every detail. Therefore, the content below provides an overview of what to pay attention to, while in-depth explanations, debugging, and optimization will be covered in future articles.

<p align="center">
 <h1 align="center">Nsight Systems - Nsight Compute </h1>
</p>

Before we go through these two tools, let me give you an example to make it easier for you to understand. When you go to the doctor for a regular check-up, you will first have a general check-up. If everything is fine, then you can leave. But if there is a problem (for example, with your heart or lungs), then you will need to have a more detailed examination of the parts that are not working properly. In this case, **the performance of our code** is similar to our health. First, we will use **Nsight Systems to check our code overall** to see if there are any problems (for example, with the **functions or the copy data**). If there are, then we will use **Nsight Compute to identify the problem in the function/copy data** so that we can **optimize and debug it.**

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/20e03fb6-f1da-4149-abef-5e4630c4ad28" />
</p>

`As you can see in the figure, we will start with Nsight Systems (general check-up) and then move on to Nsight Compute (detailed analysis of the kernels, also known as functions on the GPU). It is important to note that I will not be covering Nsight Graphics because it is for the graphics and gaming industry. However, you should not be disappointed because the metrics are very similar to those of Nsight Compute.`

**One thing to keep in mind is that these two tools, Compute and Systems, are ONLY for programs that use GPUs to run. That is why in this series, I will only be showing how to use them for parallel programming or Deep Learning models.** 

<p align="center">
 <h1 align="center">Nsight Systems</h1>
</p>

As you can see in the figure, Nsight Systems is first used to analyze the program. So, what specifically do we analyze here?

## 1. Time/speed/size when transferring data from the host to the device and vice versa

 ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/662fb9fd-032b-4d69-aaf6-5704e6282694)

 ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/ec6d18f0-4bc5-4d90-9260-34b59b543748)

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/a417645e-446a-45e5-b6c2-c9a2c3c3bbc9" />
</p>

Based on the three images above, we can see that we can **improve the copy from the host to the device.**

## 2. Next, we can look at an overview of our kernel (kernel name: mygemm)

The metrics that we will need to focus on for analysis are: **Grid/block/Theoretical occupancy**

**Summary: After the general check-up, we see that we can improve the code in two areas: copy data and kernel.**

<p align="center">
 <h1 align="center">Nsight Compute </h1>
</p>

After confirming that the two problems to be addressed are data copy and kernel, we will use Nsight Compute to analyze in more detail what the problem is.

### 1. First, the "summary" will show us where we are having problems and how to solve them (I will not go into too much detail here, but I will provide a brief explanation).

  ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/6bec7cab-c45e-4067-9234-3f5a63945cc5)

As you can see in the figure, we can improve **three things, including two** that I have already analyzed above:

- **Theoretical warps speedup 33.33%:** You will notice that in the kernel overview figure, the Theoretical occupancy is 66.66%, which means that we can improve it further (in theory, it can reach 100%).
  
- **DRAM Execessive Read Sectors:** This means that our memory allocation and organization is not optimized, which leads to problems with read/write during data transfer.
Source

### 2. Next, the "Source" will show us the line of code that is performing the heaviest work (consuming a lot of time/memory).

  ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/51881ee3-ba12-4875-96d4-8b6b8bd288e4)

### 3. The "Detail" is also the most difficult section and contains the most information that needs to be analyzed. The Detail contains a lot of information, but we will focus on the following:

- **GPU Speed of Light Throughput**
  
  ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/48d60176-6cb8-4d9a-889d-1f70bbe81686)
  

  ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/8441d8de-31b4-484e-96eb-62f6ce0aa02d)


- **Memory Workload Analysis**

   ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/d60e8d78-0409-43e6-a5d6-b831f3e3033b)
  

   ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/7f0afaf4-dec6-4d2a-bf4c-ad523033a16b)

- **Scheduler Statistics**

    ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/739ae9d8-1788-43d6-bf05-cffa8c7a53ce)

  
- **Occupancy**
  
   ![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/637c70ca-06ad-4296-8cae-2da0346551d4)

<p align="center">
 <h1 align="center">Summary </h1>
</p>

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/49a90063-0fb5-4fb8-b401-d32b01054290" />
</p>

After reading this article, you should have a good idea of the usefulness of Nsight Systems and Nsight Compute. In the following articles, I will go into more detail.

I hope this translation is accurate and helpful. Let me know if you have any other questions.
