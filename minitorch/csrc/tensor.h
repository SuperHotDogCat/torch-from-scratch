#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data; // 32 bit data
    int *strides;
    int *shape;
    int ndim; // 次元数
    int size; // 全サイズ
    char *device;
} Tensor;

/*
C/C++では、関数を共有ライブラリからエクスポートするには適切にエクスポート指定を行う必要がある
通常、以下のように extern "C" などを使用して、関数を正しくエクスポートする必要がある

これをやらないとundefined symbol: create_tensorなどになる
*/

extern "C" {
    void minitorch_error(const char *err, ...);
    Tensor *create_tensor(float *data, int *shape, int ndim, char *device);
    float get_item(Tensor *tensor, int *indices);
    Tensor *add_tensor(Tensor *tensor1, Tensor *tensor2);
    Tensor *sub_tensor(Tensor *tensor1, Tensor *tensor2);
    Tensor *elementwise_mul_tensor(Tensor *tensor1, Tensor *tensor2);
    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    void to_device(Tensor *tensor, char *target_device);
}

#endif
