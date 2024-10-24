// Torchは内部でC/C++コードが動いている。
// このソースコードはTensorの定義とTensorの処理を書く
#include "tensor.h"
#include "cpu.h"
#include <cuda_runtime_api.h> // これをいれないと.cuhのコンパイルで躓くことになる
#include "cuda.cuh"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void norch_error(const char *err, ...){
    // fprintfのように任意引数を受取errorを吐く
    va_list args;
    va_start(args, err);
    vfprintf(stderr, err, args);
    va_end(args);
    exit(1);
}

Tensor *create_tensor(float *data, int *shape, int ndim, char *device){
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if (tensor == NULL){
        norch_error("Memory allocation failed\n");
    }

    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;
    tensor->device = device;

    tensor->size = 1;
    for (int i = 0; i < ndim; i++){
        tensor->size *= shape[i];
    }

    tensor->strides = (int *)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL){
        norch_error("Memory allocation failed\n");
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--){
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    return tensor;
}

float get_item(Tensor *tensor, int *indices){
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++){
        index += tensor->strides[i] * indices[i];
    }
    float result;
    result = tensor->data[index];

    return result;
}

Tensor *add_tensor(Tensor *tensor1, Tensor *tensor2){
    if (tensor1->ndim != tensor2->ndim){
        norch_error("Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
    }
    if (strcmp(tensor1->device, tensor2->device) != 0){
        norch_error("Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
    }
    // add した結果の計算結果を置くデバイスに関する情報も保持する
    char* device = (char*)malloc(strlen(tensor1->device) + 1);
    if (device != NULL) {
        strcpy(device, tensor1->device);
    } else {
        norch_error("Memory allocation failed\n");
    }

    int ndim = tensor1->ndim;
    int *shape = (int *)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        norch_error("Memory allocation failed\n");
    }
    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            norch_error("Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
        }
        shape[i] = tensor1->shape[i];
    }

    if (strcmp("cuda", tensor1->device) == 0){
        // on cuda
        float *result_data;
        cudaMalloc((void **)&result_data, sizeof(float) * tensor1->size);
        add_tensor_cuda(tensor1, tensor2, result_data);
        return create_tensor(result_data, shape, ndim, device);
    } else {
        float *result_data = (float *)malloc(tensor1->size * sizeof(float));
        if (result_data == NULL) {
            norch_error("Memory allocation failed\n");
        }
        add_tensor_cpu(tensor1, tensor2, result_data);
        return create_tensor(result_data, shape, ndim, device);
    }
}

Tensor *sub_tensor(Tensor *tensor1, Tensor *tensor2){
    if (tensor1->ndim != tensor2->ndim){
        norch_error("Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
    }
    if (strcmp(tensor1->device, tensor2->device) != 0){
        norch_error("Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
    }
    // sub した結果の計算結果を置くデバイスに関する情報も保持する
    char* device = (char*)malloc(strlen(tensor1->device) + 1);
    if (device != NULL) {
        strcpy(device, tensor1->device);
    } else {
        norch_error("Memory allocation failed\n");
    }

    int ndim = tensor1->ndim;
    int *shape = (int *)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        norch_error("Memory allocation failed\n");
    }
    for (int i = 0; i < ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            norch_error("Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
        }
        shape[i] = tensor1->shape[i];
    }

    if (strcmp("cuda", tensor1->device) == 0){
        // on cuda
        float *result_data;
        cudaMalloc((void **)&result_data, sizeof(float) * tensor1->size);
        sub_tensor_cuda(tensor1, tensor2, result_data);
        return create_tensor(result_data, shape, ndim, device);
    } else {
        float *result_data = (float *)malloc(tensor1->size * sizeof(float));
        if (result_data == NULL) {
            norch_error("Memory allocation failed\n");
        }
        sub_tensor_cpu(tensor1, tensor2, result_data);
        return create_tensor(result_data, shape, ndim, device);
    }
}

Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim){

    int ndim = new_ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        norch_error("Memory allocation failed\n");
    }

    for (int i = 0; i < ndim; i++) {
        shape[i] = new_shape[i];
    }

    // Calculate the total number of elements in the new shape
    int size = 1;
    for (int i = 0; i < new_ndim; i++) {
        size *= shape[i];
    }

    // Check if the total number of elements matches the current tensor's size
    if (size != tensor->size) {
        norch_error("Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor.\n");
    }

    float* result_data = (float*)malloc(tensor->size * sizeof(float));
    if (result_data == NULL) {
        norch_error("Memory allocation failed\n");
    }

    assign_tensor_cpu(tensor, result_data);
    char* device = (char*)malloc(strlen(tensor->device) + 1);
    if (device != NULL) {
        strcpy(device, tensor->device);
    } else {
        norch_error("Memory allocation failed\n");
    }

    return create_tensor(result_data, shape, ndim, device); // 一旦cpuのまま
}

void to_device(Tensor *tensor, char *target_device){
    if ((strcmp(target_device, "cuda") == 0) && (strcmp(tensor->device, "cuda") != 0)){
        cpu_to_cuda(tensor);
    } else if ((strcmp(target_device, "cpu") == 0) && (strcmp(tensor->device, "cpu") != 0)){
        cuda_to_cpu(tensor);
    }
}
