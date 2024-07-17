#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        hipGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    }

    // Test setting the device and performing a simple operation
    for (int i = 0; i < deviceCount; ++i) {
        std::cout << "Setting device " << i << std::endl;
        hipError_t err = hipSetDevice(i);
        if (err != hipSuccess) {
            std::cerr << "Failed to set device " << i << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        std::cout << "Successfully set device " << i << std::endl;

        // Perform a simple kernel launch to test the device
        int *d_a;
        hipMalloc(&d_a, sizeof(int) * 10);
        hipFree(d_a);
    }

    return 0;
}
