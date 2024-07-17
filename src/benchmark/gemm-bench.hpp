// gemm-bench.hpp
#pragma once

#include <cstring>
#include <string>
#include <vector>

class GEMMData;

typedef struct compiler_state_t iree_compiler_state_t;
typedef struct runtime_state_t  iree_runtime_state_t;

namespace GEMMBench
{
    struct Problem
    {
        Problem() = delete;

        Problem(std::string stag,
                uint        M,
                uint        N,
                uint        K,
                std::string sA,
                std::string sB,
                std::string sdtype)
            : M(M)
            , N(N)
            , K(K)
        {
            std::strncpy(tag, stag.c_str(), 64);
            std::strncpy(dtype, sdtype.c_str(), 8);
            A = sA[0];
            B = sB[0];
        }

        uint M, N, K;
        char tag[64];
        char dtype[8];
        char A, B;
    };

    struct Solution
    {
        Solution() = delete;

        Solution(std::string sname)
        {
            std::strncpy(name, sname.c_str(), 32);
        }

        char name[32];
    };

    struct Configuration
    {
        Configuration() = delete;

        Configuration(uint fclk, uint mclk, uint gfxclk)
            : fclk(fclk)
            , mclk(mclk)
            , gfxclk(gfxclk)
        {
        }

        uint fclk, mclk, gfxclk;
    };

    struct Result
    {
        bool   ok;
        int    warm_iterations;
        double mean_microseconds;
        double min_sclk, mean_sclk, max_sclk;
        double min_mclk, mean_mclk, max_mclk;
        double min_fclk, mean_fclk, max_fclk;

        int                device;
        std::vector<float> sclk, mclk, fclk, temperature, power;
    };

    class GEMMPipeline
    {
    public:
        void initialize() {};
        void destroy() {};
        void linkData(GEMMData* data)
        {
            this->data = data;
        };
        void           setDevice(int device_id) {};
        virtual Result run(Problem problem) = 0;

    protected:
        GEMMData* data = NULL;
    };

    class IREEGEMMBench : public GEMMPipeline
    {

    private:
        iree_runtime_state_t*  runtime_state;
        iree_compiler_state_t* compile_state;
        int                    device_id = 0;

    public:
        IREEGEMMBench(){};
        void   initialize();
        void   setDevice(int device_id);
        void   destroy();
        Result run(Problem problem) override;
    };

    class RocBLASGEMMBench : public GEMMPipeline
    {
    public:
        RocBLASGEMMBench(){};
        void   setDevice(int device_id);
        Result run(Problem problem) override;
    };

    int testDPM(int device);
    int run(int device);
}
