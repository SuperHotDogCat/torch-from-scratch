# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -fPIC
NVCCFLAGS = -Xcompiler -fPIC -arch sm_89

# CUDA and system include directories
CUDA_PATH = /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include

# Source files
CPU_SRCS = cpu.cpp tensor.cpp
CUDA_SRCS = cuda.cu

# Object files
CPU_OBJS = $(CPU_SRCS:.cpp=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Target shared library
LIBRARY = ../libtensor.so

# Rule for building CPU object files
%.o: %.cpp
	$(CXX) -shared $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for building CUDA object files
%.o: %.cu
	$(NVCC) -shared $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Link object files into a shared library
$(LIBRARY): $(CPU_OBJS) $(CUDA_OBJS)
	$(CXX) -shared $(CPU_OBJS) $(CUDA_OBJS) -L$(CUDA_PATH)/lib64 -lcudart -o $(LIBRARY)

# Clean up object files and the shared library
clean:
	rm -f $(CPU_OBJS) $(CUDA_OBJS) $(LIBRARY)
