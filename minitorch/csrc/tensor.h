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
    Tensor *create_tensor(float *data, int *shape, int ndim);
    float get_item(Tensor *tensor, int *indices);
    Tensor *add_tensor(Tensor *tensor1, Tensor *tensor2);
    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
}

#endif
