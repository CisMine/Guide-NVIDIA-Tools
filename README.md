<p align="center">
 <h1 align="center">NVIDIA Tools Usage Guide </h1>
</p>

This repository contains documentation and examples on how to use NVIDIA tools for profiling, analyzing, and optimizing GPU-accelerated applications for beginners with a starting point. Currently, it covers NVIDIA Nsight Systems and NVIDIA Nsight Compute, and it may include guidance on using NVIDIA Nsight Deep Learning Designer in the future as well.


<p align="center">
 <h1 align="center">Introduction to NVIDIA Tools </h1>
</p>

NVIDIA provides a suite of powerful profiling and analysis tools to help developers optimize, debug, and accelerate GPU applications. This repository aims to provide comprehensive guidance on using these tools effectively in your GPU development workflow.

- ### NVIDIA Nsight Systems:
  This tool allows you to profile CPU and GPU activities, view timeline traces, and analyze system-wide performance bottlenecks. It helps you gain insights into how your application is utilizing GPU resources.

- ### NVIDIA Nsight Compute:
  Nsight Compute is a GPU profiler that provides detailed insights into the performance of individual GPU kernels. It helps you identify performance bottlenecks at the kernel level and optimize your GPU code accordingly.

- ### NVIDIA Nsight Deep Learning Designer (Future):
  Nsight Deep Learning Designer is designed for deep learning model optimization and debugging. While it may not be covered in this repository yet, future updates may include guidance on using this tool for your deep learning projects.

<p align="center">
 <h1 align="center">Getting Started </h1>
</p>

### Download Nsight systems
- Follow this [link for downloading](https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-3)

- You can use this command to verify its existence: 

  $nsys -v


### Download Nsight compute
- Nsight Compute is bundled within the CUDA Toolkit. If you've already installed the CUDA Toolkit, there's no need to download Nsight Compute separately. If you wish to switch to a different version, you can do so by using the [provided link](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)

- You can use this command to verify its existence:
 
  $ncu -v

![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/6d0bb179-42a1-4bce-b1ed-3f5682a988b4)

- If you haven't installed CUDA Toolkit yet, please follow these steps:
   - If your computer has GPU, follow these steps in NIVIDA to install [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
  
     - If you are using Linux, I advise you to watch [this video](https://www.youtube.com/watch?v=wxNQQP9U1Bc)
     
     - If you are using Windows, this is [your video](https://www.youtube.com/watch?v=cuCWbztXk4Y&t=49s)


  - If your computer doesn't have GPU
    
    - Don't worry; I'll demonstrate how to set up and use Google Colab to code [in here](https://medium.com/@giahuy04/the-easiest-way-to-run-cuda-c-in-google-colab-831efbc33d7a)


<p align="center">
 <h1 align="center">Prerequisites </h1>
</p>

- Basic knowledge of C/C++ programming.
- Understanding of parallel programming concepts.
- Familiarity with the CUDA programming model.
- Access to a CUDA-capable GPU.

  ### If you are unfamiliar with these concepts, please refer to this [series parallel computing](https://github.com/CisMine/Parallel-Computing-Cuda-C)


 
<p align="center">
 <h1 align="center">Table of Contents </h1>
</p>

[Chapter01: Introduction to Nsight Systems - Nsight Compute](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter01)

[Fix-Bug](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Fix-Bug)


<p align="center">
 <h1 align="center">Resources </h1>
</p>

In addition to the code examples, this repository provides a curated list of resources, including books, tutorials, online courses, and research papers, to further enhance your understanding of using NVIDIA Tools. These resources will help you delve deeper into the subject and explore advanced topics and techniques.

- [Nsight Systems v2023.3.1 Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute v2023.2.1 Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
