


In the article on [Synchronization - Asynchronization](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter08), we mentioned the concept of **latency hiding**, a very common term when talking about CUDA. When discussing **latency hiding**, it often involves the idea of **always keeping threads busy**. Therefore, in this article, I will explain this concept in more detail as well as its operating mechanism - it can be said to be indispensable because it greatly helps us in optimizing code.

`The purpose of this article is to help you better understand the operation mechanism of CUDA, so it will be quite important in the NVIDIA Tools series. However, if you are only interested in coding with CUDA at a basic level, you may skip this article.`

<p align="center">
 <h1 align="center">Warp Scheduler </h1>
</p>


Before diving into the lesson, let's use an example to make it easier to visualize:

Imagine there are 100 people coming to the post office to send parcels, and there's only one worker available. To successfully send a parcel, you must complete two steps: fill in the parcel information form (which takes a lot of time) - the staff confirms the form and proceeds with the parcel sending procedure (which is quite fast). Here, instead of the staff waiting for each person to finish filling out the form to proceed with the procedure, as soon as someone's turn comes, they are given a form and go somewhere else to fill it out, and once completed, they return to the queue ==> this is much faster compared to waiting for each person to fill out the form one by one.

Similarly, with computers, let's assume we have a problem: y[i] += x[i] * 3. The computer also has to perform two steps:

- **Memory instruction:** the time between the load/store operation being issued and the data arriving at its destination.
- **Arithmetic instruction:** the time an arithmetic operation starts to its output.

Going back to the example y[i] += x[i] * 3, instead of the computer having to wait to load/store x[0] and y[0], the computer will move on to load/store x[1] and y[1], and continue doing so until x[0] and y[0] are loaded/stored before returning to compute.


<p align="center">
 <h1 align="center">1st method </h1>
</p>

<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/990d2874-7647-4b5e-b7ae-272f13fbb46f" />
</p>




<p align="center">
 <h1 align="center">2nd method </h1>
</p>



<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/3757745f-6d9b-4435-b02e-c67d136f3c27" />
</p>


**To summarize, the Warp Scheduler performs the action of swapping busy warps to save time, hence it's often referred to as latency hiding or always keeping threads busy (depending on the machine, a Warp Scheduler can control a certain number of warps).**


If you find the above example similar to the candy distribution example in the article on [how computers work](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter02), then you are correct.


When discussing warps, we typically encounter three states of a warp:
- **Stalled:** The warp is busy executing something.
- **Eligible:** The warp is idle and ready to participate.
- **Selected:** The warp is chosen for execution.
  
The idea is that after a warp is **selected**, it will execute a Memory instruction and, during the wait time, it will be swapped for another warp. Here, there can be two scenarios: the subsequent warp is either **stalled or eligible**. If it's **eligible**, that's great; if it's **stalled**, it will be swapped again until an **eligible** warp is found.

**The question arises: If so, can we just create many warps so the number of eligibles increases?**

**If you think so, that's a mistake**. Creating more warps means the warp scheduler has to do more work, and creating many warps (i.e., many threads) leads to a decrease in the number of registers available per thread ==> causing the SM (Streaming Multiprocessors) to run slower ==> we have to consider how many threads are appropriate to use.

**If you think that if 128 people come to mail letters, we should use 128 workers, that's incorrect. Similarly, if we need to process an array of 128 elements using 128 threads (4 warps), that's a mistake.**

Reason: It wastes resources and, given today's computers are very powerful, it means one worker can handle two people at once, but if we only have them handle one person at a time, it's somewhat wasteful ==> one thread handles two elements ===> reduces the number of threads initiated ==> increases registers for each thread + reduces the workload for the warp scheduler.

`For the same reason, when you profile OpenCV CUDA code with Nsight Systems, you will see very few threads being used. Here is the example using opencv cuda to add 2 images`

```
#include "opencv2/opencv.hpp"
#include <opencv2/cudaarithm.hpp>

cv::Mat opencv_add(const cv::Mat &img1, const cv::Mat &img2)
{
   cv::cuda::GpuMat d_img1, d_img2, d_result;

   d_img1.upload(img1);
   d_img2.upload(img2);

   cv::cuda::add(d_img1, d_img2, d_result);

   cv::Mat result;
   d_result.download(result);

   return result;
}
int main()
{
   cv::Mat img1 = cv::imread("circles.png");
   cv::Mat img2 = cv::imread("cameraman.png");

   cv::Mat result = opencv_add(img1, img2);

   cv::imshow("Result", result);

   cv::waitKey();

   return 0;
}
```

`Then profile by using Nsight system to see the kernel by this command:`

```
nsys profile -o test ./a.out
```
<p align="center">
  <img src="https://github.com/CisMine/Guide-NVIDIA-Tools/assets/122800932/6397679d-44ff-4fe8-b287-7a99e678d791" />
</p>



**The question arises: So how many threads should we use?**

==> It depends on the configuration of each computer as well as the reasons causing warps to be stalled. In the next article, I will analyze the reasons causing warp stalls as well as how to determine the appropriate number of threads.




