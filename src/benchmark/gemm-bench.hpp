// gemm-bench.hpp
#pragma once

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "FrequencyMonitor.hpp"
#include "Timer.hpp"

class GEMMData;

typedef struct compiler_state_t iree_compiler_state_t;
class IREEGemmDeviceStorage;
class IREEGemmRuntimeState;

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
        virtual void initialize() {};
        virtual void destroy() {};
        virtual void linkData(GEMMData* data)
        {
            this->data = data;
        };
        virtual void   setDevice(int device_id) {};
        virtual Result run(Problem problem) = 0;

    protected:
        GEMMData* data = NULL;
    };

    class IREEGEMMBench : public GEMMPipeline
    {
    private:
        IREEGemmRuntimeState*  runtime_state;
        IREEGemmDeviceStorage* storage_fp16;
        IREEGemmDeviceStorage* storage_bf16;

        iree_compiler_state_t* compile_state;
        int                    device_id = 0;

    public:
        IREEGEMMBench() {};
        void   initialize() override;
        void   linkData(GEMMData* data) override;
        void   setDevice(int device_id) override;
        void   destroy() override;
        Result run(Problem problem) override;
    };

    class SHARKFABench : public GEMMPipeline
    {
    private:
        IREEGemmRuntimeState*  runtime_state;
        IREEGemmDeviceStorage* storage_fp16;
        IREEGemmDeviceStorage* storage_fp32;
        int                    device_id = 0;

    public:
        SHARKFABench() {};
        void   initialize() override;
        void   linkData(GEMMData* data) override;
        void   setDevice(int device_id) override;
        void   destroy() override;
        Result run(Problem problem) override;
    };

    class RocBLASGEMMBench : public GEMMPipeline
    {
    public:
        RocBLASGEMMBench() {};
        void   setDevice(int device_id) override;
        Result run(Problem problem) override;
    };

    class HipBLASLtGEMMBench : public GEMMPipeline
    {
    public:
        HipBLASLtGEMMBench();
        ~HipBLASLtGEMMBench();

        void   initialize() override;
        void   destroy() override;
        void   setDevice(int device_id) override;
        Result run(Problem problem) override;

    private:
        void*             workspace;
        hipStream_t       stream;
        hipblasLtHandle_t handle;
        hipblasStatus_t   hipblaslt_status;

        void executeGemm(int                 num_iterations,
                         Timer*              timer,
                         Frequency::Monitor* monitor,
                         hipblasLtHandle_t   handle,
                         hipblasOperation_t  trans_a,
                         hipblasOperation_t  trans_b,
                         int64_t             m,
                         int64_t             n,
                         int64_t             k,
                         int64_t             batch_count,
                         float&              alpha,
                         float&              beta,
                         void*               d_a,
                         void*               d_b,
                         void*               d_c,
                         void*               d_d,
                         void*               d_workspace,
                         int64_t             max_workspace_size,
                         hipStream_t         stream);
    };

    int testDPM(int device);
    int run(int device, std::string backend);
}
