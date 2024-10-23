// Torchは内部でC/C++コードが動いている。
// このソースコードはTensorの定義とTensorの処理を書く
#include "tensor.h"
#include "cpu.h"
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

Tensor *create_tensor(float *data, int *shape, int ndim){
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if (tensor == NULL){
        norch_error("Memory allocation failed\n");
    }

    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

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
    int size = tensor1->size;
    float *result_data = (float *)malloc(size * sizeof(float));
    if (result_data == NULL) {
        norch_error("Memory allocation failed\n");
    }
    add_tensor_cpu(tensor1, tensor2, result_data);

    return create_tensor(result_data, shape, ndim);
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
    return create_tensor(result_data, shape, ndim);
}
