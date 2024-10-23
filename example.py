import minitorch

tensor1 = minitorch.Tensor([[1, 2, 3], [3, 2, 1]])
tensor2 = minitorch.Tensor([[3, 2, 1], [1, 2, 3]])

result = tensor1 + tensor2
print(result[1,1])
# 4
