<p align="center">
 <h1 align="center"> Common errors when using
  
   Nsight Systems - Nsight Compute </h1>
</p>


Even though the download was successful and checked with the commands nsys -v or ncu -v, there will still be some errors so I will show you how to fix them.

![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/349f86a1-d566-4c4f-b227-36ff70816c33)

```
#include <stdio.h>


__global__ void kernel()
{

    printf("hello world");
}

int main()
{
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

This is a simple piece of code for us to test the two tools we just downloaded

<p align="center">
 <h1 align="center"> Nsight Systems  </h1>
</p>

run these commands ( I'll explain in others chapter)

$nvcc test.cu

$./a.out

$nsys profile ./a.out

![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/2fdad835-220d-48a0-a18d-4e91c60df6ef)

Open Nsight system and  open that file (.nsys-rep)

Click the warmings and you'll see the warning Daemon

![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/7c22fd95-baa8-4091-938f-c705496c6755)

![image](https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/5cef9e18-2fb6-4c78-92a4-ed1a2bf6bfc3)


Then run this command to fix it. PLEASE NOTE THAT EACH COMPUTER WILL HAVE A DIFFERENT LEVEL, SO PLEASE PAY ATTENTION.

$cat /proc/sys/kernel/perf_event_paranoid

$sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid' ( change the number 1 into your computer's number )

$sudo sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'

run this to check:

$cat /proc/sys/kernel/perf_event_paranoid


<p align="center">
 <h1 align="center"> Nsight Compute  </h1>
</p>

Instead of running the command line $nsys profile ./a.out  run this

$ncu --set full -o test ./a.out


![image](https://github.com/user-attachments/assets/b6441013-116f-4056-91ac-d70d9f33fcb7)

If it creates the .ncu-rep file, it is successful BUT if you encounter the problem of **nsight compute permission deny**, then run these commands:

$sudo nano /etc/modprobe.d/nvidia.conf

$options nvidia NVreg_RestrictProfilingToAdminUsers=0

then do ctrl + o then ctrl + x ( save and exit )
