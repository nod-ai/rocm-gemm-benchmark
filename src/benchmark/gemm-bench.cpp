
#include <iostream>
#include <unordered_map>

#include "gemm-bench.hpp"
#include "zmq.hpp"

#include "DataInitialization.hpp"
#include "FrequencyMonitor.hpp"
#include "RotatingBuffer.hpp"

namespace GEMMBench
{
    const uint WORKERS_PORT = 7178;
    const uint RESULTS_PORT = 7179;

    std::unordered_map<std::string, GEMMPipeline*> benches{
        {"rocblas", new RocBLASGEMMBench()},
        {"iree", new IREEGEMMBench()},
    };

    /**
     * Set GPU configuration (DPM state etc).
     */
    bool set_configuration(Configuration configuration, bool verbose = false)
    {
        auto dpm = Frequency::getDPMManager();

        auto fclk   = dpm->getFCLK();
        auto mclk   = dpm->getMCLK();
        auto gfxclk = dpm->getGFXCLK();

        if(verbose)
        {
            std::cout << "current: fclk " << fclk << " mclk " << mclk << " gfxclk " << gfxclk
                      << std::endl;
            std::cout << "setting: fclk " << configuration.fclk << " mclk " << configuration.mclk
                      << " gfxclk " << configuration.gfxclk << std::endl;
        }

        if(fclk == configuration.fclk && mclk == configuration.mclk)
            return true;

        for(auto attempt = 0; attempt < 2; ++attempt)
        {
            dpm->setFCLK(configuration.fclk);
            dpm->setMCLK(configuration.mclk);
            dpm->setGFXCLK(configuration.gfxclk);

            fclk = dpm->getFCLK();
            mclk = dpm->getMCLK();

            if(verbose)
            {
                std::cout << "current: fclk " << fclk << " mclk " << mclk << " gfxclk " << gfxclk
                          << std::endl;
            }

            if(fclk == configuration.fclk && mclk == configuration.mclk
               && gfxclk == configuration.gfxclk)
                return true;
        }

        return false;
    }

    /**
     * Run a GEMM.
     * - Sets configuration (DPM state etc)
     * - Dispatch based on solver.
     */
    Result run_problem(Problem problem, Solution solution, int device, Configuration configuration)
    {
        auto result = Result{.ok = false};

        std::string backendName = std::string(solution.name);
        result                  = benches[backendName]->run(problem);
        result.device           = device;

        return result;
    }

    /**
     * Connect to ZMQ server and dispatch GEMM runs via `run_problem`.
     */
    int run(int device)
    {
        std::cout << "Initializing tensors with trig..." << std::endl;
        GEMMTrigInitializer initializer;
        GEMMData            data("fp32", 1e9, &initializer);

        std::cout << "Running on " << benches.size() << " benches" << std::endl;

        for(const auto& [name, pipeline] : benches)
        {
            std::cout << "Initializing " << name << std::endl;
            pipeline->initialize();
            pipeline->setDevice(device);
            pipeline->linkData(&data);
        }

        Frequency::getFrequencyMonitor()->setDevice(device);
        Frequency::getDPMManager()->setDevice(device);

        // Connect to server
        zmq::context_t ctx;
        zmq::socket_t  workerPullSocket(ctx, zmq::socket_type::pull);
        zmq::socket_t  resultPushSocket(ctx, zmq::socket_type::push);

        workerPullSocket.connect("tcp://127.0.0.1:" + std::to_string(WORKERS_PORT));
        resultPushSocket.connect("tcp://127.0.0.1:" + std::to_string(RESULTS_PORT));

        while(true)
        {
            zmq::message_t msgSerialNumber(sizeof(int));
            zmq::message_t msgProblem(sizeof(Problem));
            zmq::message_t msgSolution(sizeof(Solution));
            zmq::message_t msgConfiguration(sizeof(Configuration));

            auto res = workerPullSocket.recv(msgSerialNumber, zmq::recv_flags::none);
            res      = workerPullSocket.recv(msgProblem, zmq::recv_flags::none);
            res      = workerPullSocket.recv(msgSolution, zmq::recv_flags::none);
            res      = workerPullSocket.recv(msgConfiguration, zmq::recv_flags::none);

            Problem*       problem       = msgProblem.data<Problem>();
            Solution*      solution      = msgSolution.data<Solution>();
            Configuration* configuration = msgConfiguration.data<Configuration>();

            auto result = run_problem(*problem, *solution, device, *configuration);

            zmq::message_t msgResult((char*)&result, sizeof(Result));
            zmq::message_t msgSCLK((char*)result.sclk.data(), sizeof(float) * result.sclk.size());
            zmq::message_t msgMCLK((char*)result.mclk.data(), sizeof(float) * result.mclk.size());
            zmq::message_t msgTemperature((char*)result.temperature.data(),
                                          sizeof(float) * result.temperature.size());
            zmq::message_t msgPower((char*)result.power.data(),
                                    sizeof(float) * result.power.size());
            resultPushSocket.send(msgSerialNumber, zmq::send_flags::sndmore);
            resultPushSocket.send(msgResult, zmq::send_flags::sndmore);
            resultPushSocket.send(msgSCLK, zmq::send_flags::sndmore);
            resultPushSocket.send(msgMCLK, zmq::send_flags::sndmore);
            resultPushSocket.send(msgTemperature, zmq::send_flags::sndmore);
            resultPushSocket.send(msgPower, zmq::send_flags::none);
        }

        for(const auto& [name, pipeline] : benches)
        {
            pipeline->destroy();
        }

        return 0;
    }

    int testDPM(int device)
    {
        Frequency::getFrequencyMonitor()->setDevice(device);
        Frequency::getDPMManager()->setDevice(device);

        auto problem = Problem("test", 8192, 8192, 8192, "T", "N", "fp16");

        for(auto fclk : {1, 0})
        {
            for(auto mclk : {1, 0})
            {
                auto configuration = Configuration(fclk, mclk, 0);
                if(!set_configuration(configuration, true))
                {
                    std::cout << "Setting DPM failed: fclk " << fclk << " mclk " << mclk
                              << std::endl;
                    return -1;
                }

                auto result = benches["rocblas"]->run(problem);
                std::cout << " time " << result.mean_microseconds << "us";
                std::cout << " mclks " << result.min_mclk << "/" << result.mean_mclk << "/"
                          << result.max_mclk;
                std::cout << " sclks " << result.min_sclk << "/" << result.mean_sclk << "/"
                          << result.max_sclk;
                std::cout << std::endl;
            }
        }

        return 0;
    }
}
