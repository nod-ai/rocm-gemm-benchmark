#include <iostream>
#include <optional>
#include <string>

#include "gemm-bench.hpp"

template <typename T>
std::optional<T> option(std::string const& prefix, int argc, char* argv[])
{
    if constexpr(std::is_same_v<T, int>)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            if(!arg.compare(0, prefix.size(), prefix))
                return std::stoi(arg.substr(prefix.size()));
        }
    }
    else if constexpr(std::is_same_v<T, bool>)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            if(!arg.compare(0, prefix.size(), prefix))
                return true;
        }
        return false;
    }
    else
    {
        std::cout << "Option type not implemented yet." << std::endl;
    }
    return {};
}

int main(int argc, char* argv[])
{
    // Parse arguments
    auto help    = option<bool>("--help", argc, argv).value_or(false);
    auto device  = option<int>("--device=", argc, argv).value_or(0);
    auto testDPM = option<bool>("--test-dpm", argc, argv).value_or(false);

    if(help)
    {
        std::cout << "Usage: gemm-bench [--help] [--test-dpm] [--device=<int>]" << std::endl;
        return 1;
    }

    if(testDPM)
        return GEMMBench::testDPM(device);

    return GEMMBench::run(device);
}
