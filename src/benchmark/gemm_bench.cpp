
#include <iostream>

int gemm_bench(int device)
{
    // Set device
    {
        std::cout << "Device: " << device << std::endl;
        auto err = hipSetDevice(device);
        if(err != hipSuccess)
        {
            std::cout << "Unable to set HIP device." << std::endl;
            return 1;
        }
    }

    return 0;
}
