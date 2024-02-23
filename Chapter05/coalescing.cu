#include <cuda_runtime.h>
#include <iostream>

__global__ void testCoalesced(int* in, int* out, int elements)
{
int id = (blockDim.x * blockIdx.x +threadIdx.x) % elements;
out[id] = in[id];
}

__global__ void testMixed(int* in, int* out, int elements)
{
int id = ((blockDim.x * blockIdx.x +threadIdx.x* 7) % elements) %elements;
out[id] = in[id];
}



int main() {
    int elements = 1024;
    size_t size = elements * sizeof(int);

    int *in, *out;
    int *d_in, *d_out;

    in = (int*)malloc(size);
    out = (int*)malloc(size);

    for (int i = 0; i < elements; i++) {
        in[i] = i;
    }

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = 32;

  

    testCoalesced<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, elements);
    testMixed<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, elements);
  


    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // for(int i = 0; i < elements; i++) {
    //     std::cout << out[i] << " ";
    // }



    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
    free(out);

    return 0;
}
