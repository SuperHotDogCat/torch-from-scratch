// cpuでの演算をするコード
#include "tensor.h"

void add_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data){
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

void sub_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data){
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

void elementwise_mul_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data){
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] * tensor2->data[i];
    }
}

void matmul_tensor_cpu(Tensor *tensor1, Tensor *tensor2, float *result_data){
    // 一旦は2次元配列を想定してmatmulを書く
    int row = tensor1->shape[0]; // result row
    int col = tensor2->shape[1]; // result col
    int num_iterate_line = tensor1->shape[1];
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            float tmp = 0.0; // tmp value for i, j
            for (int k = 0; k < num_iterate_line; k++){
                tmp += tensor1->data[i*num_iterate_line + k] * tensor2->data[k*col + j];
            }
            result_data[i*col + j] = tmp;
        }
    }
}

void assign_tensor_cpu(Tensor *tensor, float *result_data){
    for (int i = 0; i < tensor->size; i++){
        result_data[i] = tensor->data[i];
    }
}
