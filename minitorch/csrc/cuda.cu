#include "tensor.h"
#include <stdio.h>

// device move codes
__host__ void cpu_to_cuda(Tensor *tensor){
    // tensor->dataをGPUに送ったデータへ変換
    float *gpu_data;
    cudaMalloc((void **)&gpu_data, sizeof(float) * tensor->size);
    cudaMemcpy(gpu_data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyHostToDevice);
    tensor->data = gpu_data;

    const char* device_str = "cuda";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 
    printf("Successfully sent tensor to: %s\n", tensor->device);
}
__host__ void cuda_to_cpu(Tensor *tensor){
    float *cpu_data = (float *)malloc(sizeof(float) * tensor->size);
    cudaMemcpy(cpu_data, tensor->data, tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tensor->data);

    tensor->data = cpu_data;
    const char* device_str = "cpu";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 
    printf("Successfully sent tensor to: %s\n", tensor->device);
}

// operation codes
#define THREADS_PER_BLOCK 128

__global__ void add_tensor_cuda_kernel(float *data1, float *data2, float *result_data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    result_data[i] = data1[i] + data2[i];
}

__host__ void add_tensor_cuda(Tensor *tensor1, Tensor *tensor2, float *result_data){
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        norch_error("CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
}

__global__ void sub_tensor_cuda_kernel(float *data1, float *data2, float *result_data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    result_data[i] = data1[i] - data2[i];
}

__host__ void sub_tensor_cuda(Tensor *tensor1, Tensor *tensor2, float *result_data){
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        norch_error("CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
}
