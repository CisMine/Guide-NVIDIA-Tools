#include <iostream>

const int manualBlockSize = 1024;

////////////////////////////////////////////////////////////////////////////////
// Test kernel
//
// This kernel squares each array element. Each thread addresses
// himself with threadIdx and blockIdx, so that it can handle any
// execution configuration, including anything the launch configurator
// API suggests.
////////////////////////////////////////////////////////////////////////////////
__global__ void square(int *array, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        array[idx] *= array[idx];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Potential occupancy calculator
//
// The potential occupancy is calculated according to the kernel and
// execution configuration the user desires. Occupancy is defined in
// terms of active blocks per multiprocessor, and the user can convert
// it to other metrics.
//
// This wrapper routine computes the occupancy of kernel, and reports
// it in terms of active warps / maximum warps per SM.
////////////////////////////////////////////////////////////////////////////////
static double reportPotentialOccupancy(void *kernel, int blockSize, size_t dynamicSMem)
{
    int device;
    cudaDeviceProp prop;

    int numBlocks;
    int activeWarps;
    int maxWarps;

    double occupancy;

    (cudaGetDevice(&device));
    (cudaGetDeviceProperties(&prop, device));

    (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        kernel,
        blockSize,
        dynamicSMem));

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    occupancy = (double)activeWarps / maxWarps;

    return occupancy;
}

////////////////////////////////////////////////////////////////////////////////
// Occupancy-based launch configurator
//
// The launch configurator, cudaOccupancyMaxPotentialBlockSize and
// cudaOccupancyMaxPotentialBlockSizeVariableSMem, suggests a block
// size that achieves the best theoretical occupancy. It also returns
// the minimum number of blocks needed to achieve the occupancy on the
// whole device.
//
// This launch configurator is purely occupancy-based. It doesn't
// translate directly to performance, but the suggestion should
// nevertheless be a good starting point for further optimizations.
//
// This function configures the launch based on the "automatic"
// argument, records the runtime, and reports occupancy and runtime.
////////////////////////////////////////////////////////////////////////////////
static int launchConfig(int *array, int arrayCount, bool automatic)
{
    int blockSize;
    int minGridSize;
    int gridSize;
    size_t dynamicSMemUsage = 0;

    cudaEvent_t start;
    cudaEvent_t end;

    float elapsedTime;

    double potentialOccupancy;

    (cudaEventCreate(&start));
    (cudaEventCreate(&end));

    if (automatic)
    {
        (cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            (void *)square,
            dynamicSMemUsage,
            arrayCount));

        std::cout << "Suggested block size: " << blockSize << std::endl
                  << "Minimum grid size for maximum occupancy: " << minGridSize << std::endl;
    }
    else
    {
        // This block size is too small. Given limited number of
        // active blocks per multiprocessor, the number of active
        // threads will be limited, and thus unable to achieve maximum
        // occupancy.
        //
        blockSize = manualBlockSize;
    }

    // Round up
    //
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    // Launch and profile
    //
    (cudaEventRecord(start));
    square<<<gridSize, blockSize, dynamicSMemUsage>>>(array, arrayCount);
    (cudaEventRecord(end));

    (cudaDeviceSynchronize());

    // Calculate occupancy
    //
    potentialOccupancy = reportPotentialOccupancy((void *)square, blockSize, dynamicSMemUsage);

    std::cout << "Potential occupancy: " << potentialOccupancy * 100 << "%" << std::endl;

    // Report elapsed time
    //
    (cudaEventElapsedTime(&elapsedTime, start, end));
    std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// The test
//
// The test generates an array and squares it with a CUDA kernel, then
// verifies the result.
////////////////////////////////////////////////////////////////////////////////
static int test(bool automaticLaunchConfig, const int count = 1000000)
{
    int *array;
    int *dArray;
    int size = count * sizeof(int);

    array = new int[count];

    for (int i = 0; i < count; i += 1)
    {
        array[i] = i;
    }

    (cudaMalloc(&dArray, size));
    (cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice));

    for (int i = 0; i < count; i += 1)
    {
        array[i] = 0;
    }

    launchConfig(dArray, count, automaticLaunchConfig);

    (cudaMemcpy(array, dArray, size, cudaMemcpyDeviceToHost));
    (cudaFree(dArray));

    // Verify the return data
    //
    for (int i = 0; i < count; i += 1)
    {
        if (array[i] != i * i)
        {
            std::cout << "element " << i << " expected " << i * i << " actual " << array[i] << std::endl;
            return 1;
        }
    }

    (cudaDeviceReset());

    delete[] array;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Sample Main
//
// The sample runs the test with manually configured launch and
// automatically configured launch, and reports the occupancy and
// performance.
////////////////////////////////////////////////////////////////////////////////
int main()
{
    int status;

    std::cout << "starting Simple Occupancy" << std::endl
              << std::endl;

    std::cout << "[ Manual configuration with " << manualBlockSize
              << " threads per block ]" << std::endl;

    status = test(false);
    if (status)
    {
        std::cerr << "Test failed\n"
                  << std::endl;
        return -1;
    }

    std::cout << std::endl;

    std::cout << "[ Automatic, occupancy-based configuration ]" << std::endl;
    status = test(true);
    if (status)
    {
        std::cerr << "Test failed\n"
                  << std::endl;
        return -1;
    }

    std::cout << std::endl;
    std::cout << "Test PASSED\n"
              << std::endl;

    return 0;
}
