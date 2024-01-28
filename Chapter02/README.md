
Before using Nvidia's profiling tools, it's essential to have a basic understanding of how CUDA works. In this article, I'll briefly explain two commonly mentioned concepts in CUDA: CUDA Toolkit and CUDA Driver.

I will provide a simple explanation without diving too deep into the details, so don't worry.

<p align="center">
 <h1 align="center">Cuda toolkit - Cuda driver </h1>
</p>

![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/720652d1-dab8-44cd-8ad4-4048f6f3dafb)


Before explaining these two terms, let's start with an analogy to help you understand better: Imagine you're playing a video game, and your character is at level 10, equipped with a level 5 weapon. In this scenario, your total combat power is 100. You have two ways to increase your character's combat power:

- The Easy Way: Find a level 10 weapon that matches your character's level.
- The Hard Way: Increase your character's level.
  
A small note: You cannot equip a weapon with a higher level than your character's level.

In the context of CUDA, it's similar. If you want to optimize a CUDA program (excluding code-related factors), you have two options: increase the level of the CUDA Toolkit or increase the level of the CUDA Driver.

- CUDA Driver: This represents the capability of your computer (similar to your character's level). The more powerful your computer is, the faster it can run, and each computer will have a certain level of capability.


![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/ec0c590f-a7f6-436e-bed0-fdda364f64a1)


- CUDA Toolkit: This represents the version of CUDA you are using (similar to the level of your weapon). A higher version **can POTENTIALLY run faster** than an older version (because newer versions are usually more optimized and may have more advanced functions compared to older versions).

**In summary, CUDA Driver is physical, representing the maximum capability your CUDA program can run at, while CUDA Toolkit is logical, representing the level of CUDA utilization. A higher version of the CUDA Toolkit indicates a more advanced level of utilization.**


<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/9c8d7c62-51e0-4a57-b0e8-d90bdd677f94" />
</p>


When coding, we have two perspectives: the coder view (logical view) and the hardware view (physical view). This means that when we optimize our code, it's optimized at the logical level, and that code is then compiled into binary code for the hardware to execute and further optimize. In the case of CUDA, the CUDA Toolkit and Driver operate similarly. We use the CUDA Toolkit to optimize our CUDA code, and the CUDA Driver optimizes the hardware for us.

**The question then arises: How do we determine which CUDA Driver and Toolkit versions to use?**

It's quite simple. We use the [NVIDIA driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) to determine the level of driver our computer is using. Here's an example from my computer:

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/6c6e9fea-6f91-4ba0-926b-125d7b938e29" />
</p>

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/50e650b6-4222-407e-afac-39d8e9e76c47" />
</p>


Here you will see that the suitable driver version for me is version 535. After that, you can go to the driver version 535 and, once downloaded, open a terminal and run this command to check the compatible CUDA Toolkit version:
```
$nvidia-smi
```

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/75d32b80-9f44-4a29-8181-c88eff9185dd" />
</p>
