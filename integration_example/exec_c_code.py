import ctypes

lib = ctypes.CDLL("./add_floats.so")
# Define the argument and return types for the function
lib.add_floats.argtypes = [ctypes.c_float, ctypes.c_float]
lib.add_floats.restype = ctypes.c_float
# Convert python float to c_float type 
a = ctypes.c_float(2.5)
b = ctypes.c_float(3.2)
# Call the C function
print(lib.add_floats(a, b))

#　配列（浮動小数点数リスト）のような複雑な型には、
#  ポインターを使うことができる。
data = [1.0, 2.0, 3.0]
data_ctype = (ctypes.c_float * len(data))(*data)
print(*data)
