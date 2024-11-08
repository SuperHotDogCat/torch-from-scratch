import minitorch
print("Initialization Start")
tensor1 = minitorch.Tensor([[1, 2, 3], [3, 2, 1]])
tensor2 = minitorch.Tensor([[3, 2, 1], [1, 2, 3]])
print("Initialization OK")

print("Addition Start")
result = tensor1 + tensor2
print(result[1,1])
assert result[1,1] == 4
print("Addition OK")
# 4
print("Subtraction Start")
result = tensor1 - tensor2
print(result[1,1])
assert result[1,1] == 0
print("Subtraction OK")
# 0

print("Elementwise mul Start")
result = tensor1 * tensor2
assert result[0,0] == 3
print("Elementwise mul OK")

print("Device on Start")
tensor3 = tensor1.to("cuda")
print("Device on OK")

print("Addition on device Start")
tensor4 = tensor2.to("cuda")
result = tensor3 + tensor4
result = result.to("cpu")
assert result[1,1] == 4
print("Addition on device OK")

print("Subtraction on device Start")
result = tensor3 - tensor4
result = result.to("cpu")
assert result[1,1] == 0
print("Subtraction on device OK")

print("Elementwise mul on device Start")
result = tensor1 * tensor2
result = result.to("cpu")
assert result[0,0] == 3
print("Elementwise mul on device OK")
