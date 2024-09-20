# Specify the compiler for CUDA files
NVCC = nvcc

# Specify the C++ compiler
CXX = g++

# Specify compiler flags
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include
NVCCFLAGS = -arch=sm_60

# Output executable name
TARGET = motion_blur 

# Source files
SRCS = main.cc blur.cu

# Object files
OBJS = main.o blur.o

# Default target to build the executable
all: $(TARGET)

# Rule to link object files and create the executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

# Rule to compile the main C++ file
main.o: main.cc blur.h
	$(CXX) $(CXXFLAGS) -c main.cc -o main.o

# Rule to compile the CUDA file
blur.o: blur.cu blur.h
	$(NVCC) $(NVCCFLAGS) -c blur.cu -o blur.o

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean

