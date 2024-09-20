#ifndef _TOOLS_H
#define _TOOLS_H

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                       \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";   \
            std::cerr << "code: " << error << ", reason: "                    \
                      << cudaGetErrorString(error) << std::endl;             \
            exit(1);                                                         \
        }                                                                    \
    }

#endif // _TOOLS_H

