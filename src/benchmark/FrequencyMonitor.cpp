
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>

#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>

#include "FrequencyMonitor.hpp"
#include "subprocess.hpp"

namespace Frequency
{
    Monitor* getFrequencyMonitor()
    {
        static Frequency::Monitor* ptr = nullptr;
        if(ptr == nullptr)
        {
            ptr = new Frequency::Monitor();

            rsmi_init(0);
        }
        return ptr;
    }

    DPMManager* getDPMManager()
    {
        static Frequency::DPMManager* ptr = nullptr;
        if(ptr == nullptr)
        {
            ptr = new Frequency::DPMManager();
        }
        return ptr;
    }

    Statistics statistics(std::vector<float> const& x)
    {
        auto y = x;
        std::sort(y.begin(), y.end());

        auto n = y.size();

        // XXX this isn't stable
        double mean = 0.0;
        for(auto i = 0; i < n; ++i)
        {
            mean += static_cast<double>(y[i]) / n;
        }

        double median = 0.0;
        if(n > 0)
        {
            median = static_cast<double>(y[(n - 1) / 2]);
            if(n % 2 == 0)
                median = (median + static_cast<double>(y[(n - 1) / 2 + 1])) / 2.0;
        }

        // Recall that y is sorted.
        auto min = (n == 0) ? 0.0f : y.front();
        auto max = (n == 0) ? 0.0f : y.back();

        return {n, mean, median, min, max};
    }

    std::string getPCIDeviceString(int hipDeviceID)
    {
        hipDeviceProp_t props;

        auto status = hipGetDeviceProperties(&props, hipDeviceID);
        if(status != HIP_SUCCESS)
            throw std::runtime_error("FrequencyMonitor: Unable to get device properties.");

        std::stringstream pciDevice;
        pciDevice << std::hex << std::setfill('0') << std::setw(4) << props.pciDomainID;
        pciDevice << ":" << std::hex << std::setfill('0') << std::setw(2) << props.pciBusID;
        pciDevice << ":" << std::hex << std::setfill('0') << std::setw(2) << props.pciDeviceID;
        pciDevice << ".0";

        return pciDevice.str();
    }

    uint32_t GetROCmSMIIndex(int hipDeviceIndex)
    {
        /* From rocBLAS */

        hipDeviceProp_t props;

        auto status = hipGetDeviceProperties(&props, hipDeviceIndex);
        if(status != HIP_SUCCESS)
        {
            throw std::runtime_error("Error getting ROCm SMI device.");
        }
#if HIP_VERSION >= 50220730
        int hip_version;
        status = hipRuntimeGetVersion(&hip_version);
        if(status != HIP_SUCCESS)
        {
            throw std::runtime_error("Error getting ROCm SMI device.");
        }

        if(hip_version >= 50220730)
        {
            status = hipDeviceGetAttribute(&props.multiProcessorCount,
                                           hipDeviceAttributePhysicalMultiProcessorCount,
                                           hipDeviceIndex);
            if(status != HIP_SUCCESS)
            {
                throw std::runtime_error("Error getting ROCm SMI device.");
            }
        }
#endif

        uint64_t hipPCIID = 0;
        hipPCIID |= (((uint64_t)props.pciDomainID & 0xffffffff) << 32);
        hipPCIID |= ((props.pciBusID & 0xff) << 8);
        hipPCIID |= ((props.pciDeviceID & 0x1f) << 3);

        uint32_t smiCount = 0;

        auto rstatus = rsmi_num_monitor_devices(&smiCount);
        if(rstatus != RSMI_STATUS_SUCCESS)
        {
            throw std::runtime_error("Error getting ROCm SMI device.");
        }

        std::ostringstream msg;
        msg << "RSMI Can't find a device with PCI ID " << hipPCIID << "(" << props.pciDomainID
            << "-" << props.pciBusID << "-" << props.pciDeviceID << ")" << std::endl;
        msg << "PCI IDs: [" << std::endl;

        for(uint32_t smiIndex = 0; smiIndex < smiCount; smiIndex++)
        {
            uint64_t rsmiPCIID = 0;

            rstatus = rsmi_dev_pci_id_get(smiIndex, &rsmiPCIID);
            if(rstatus != RSMI_STATUS_SUCCESS)
            {
                throw std::runtime_error("Error getting ROCm SMI device.");
            }

            msg << smiIndex << ": " << rsmiPCIID << std::endl;

            if(hipPCIID == rsmiPCIID)
                return smiIndex;
        }

        msg << "]" << std::endl;

        throw std::runtime_error(msg.str());
    }

    Monitor::~Monitor()
    {
        if(active())
            stop();
    }

    Monitor& Monitor::setDevice(int deviceId)
    {
        m_smiDeviceIndex = GetROCmSMIIndex(deviceId);
        return *this;
    }

    Monitor& Monitor::setResolution(resolution_type const& resolution)
    {
        m_resolution = resolution;
        return *this;
    }

    Monitor& Monitor::start()
    {
        assertNotActive();

        if(m_smiDeviceIndex == -1)
            throw std::runtime_error(
                "FrequencyMonitor: Unable to collect frequencies, device not set.");

        m_systemFrequency.clear();
        m_memoryFrequency.clear();
        m_temperature.clear();
        m_power.clear();

        m_stop   = false;
        m_thread = std::thread([&]() { this->collect(); });

        return *this;
    }

    Monitor& Monitor::stop()
    {
        assertActive();

        m_stop = true;
        m_thread.join();

        return *this;
    }

    std::tuple<Statistics, Statistics, Statistics> Monitor::statistics() const
    {
        assertNotActive();

        return {Frequency::statistics(m_systemFrequency),
                Frequency::statistics(m_memoryFrequency),
                Frequency::statistics(m_dataFrequency)};
    }

    void Monitor::collect()
    {
        rsmi_status_t      rstatus;
        rsmi_frequencies_t freq;
        uint64_t           power;
        RSMI_POWER_TYPE    powerType;
        int64_t            temperature;

        do
        {
            rstatus = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_SYS, &freq);
            if(rstatus == RSMI_STATUS_SUCCESS)
                m_systemFrequency.push_back(freq.frequency[freq.current] * 1.e-3f);

            rstatus = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_MEM, &freq);
            if(rstatus == RSMI_STATUS_SUCCESS)
                m_memoryFrequency.push_back(freq.frequency[freq.current] * 1.e-3f);

            rstatus = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_DF, &freq);
            if(rstatus == RSMI_STATUS_SUCCESS)
                m_dataFrequency.push_back(freq.frequency[freq.current] * 1.e-3f);

            rstatus = rsmi_dev_power_get(m_smiDeviceIndex, &power, &powerType);
            if(rstatus == RSMI_STATUS_SUCCESS)
                m_power.push_back(power * 1.e-6f);

            rstatus = rsmi_dev_temp_metric_get(
                m_smiDeviceIndex, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &temperature);
            if(rstatus == RSMI_STATUS_SUCCESS)
                m_temperature.push_back(temperature * 1.e-3f);

            std::this_thread::sleep_for(m_resolution);
        } while(!m_stop);
    }

    bool Monitor::active() const
    {
        return m_thread.joinable();
    }

    void Monitor::assertActive() const
    {
        if(!active())
            throw std::runtime_error("FrequencyMonitor: Invalid state (not active).");
    }

    void Monitor::assertNotActive() const
    {
        if(active())
            throw std::runtime_error("FrequencyMonitor: Invalid state (active).");
    }

    std::vector<float> Monitor::systemFrequency() const
    {
        assertNotActive();
        return m_systemFrequency;
    }

    std::vector<float> Monitor::memoryFrequency() const
    {
        assertNotActive();
        return m_memoryFrequency;
    }

    std::vector<float> Monitor::dataFrequency() const
    {
        assertNotActive();
        return m_dataFrequency;
    }

    std::vector<float> Monitor::temperature() const
    {
        assertNotActive();
        return m_temperature;
    }

    std::vector<float> Monitor::power() const
    {
        assertNotActive();
        return m_power;
    }

    DPMManager& DPMManager::setDevice(int deviceId)
    {
        auto pciDevice   = getPCIDeviceString(deviceId);
        m_smiDeviceIndex = GetROCmSMIIndex(deviceId);
        return *this;
    }

    int DPMManager::getFCLK() const
    {
        rsmi_frequencies_t freq;
        auto rstatus = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_DF, &freq);
        if(rstatus == RSMI_STATUS_SUCCESS)
            return freq.current;
        return -1;
    }

    int DPMManager::getMCLK() const
    {
        rsmi_frequencies_t freq;
        auto rstatus = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_MEM, &freq);
        if(rstatus == RSMI_STATUS_SUCCESS)
            return freq.current;
        return -1;
    }

    int DPMManager::getGFXCLK() const
    {
        rsmi_frequencies_t freq;
        auto rstatus = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_SYS, &freq);
        if(rstatus == RSMI_STATUS_SUCCESS)
            return freq.current;
        return -1;
    }

    DPMManager& DPMManager::setFCLK(int dpm)
    {
        // no internal dependencies
        return *this;
    }

    DPMManager& DPMManager::setMCLK(int dpm)
    {
        // no internal dependencies
        return *this;
    }

    DPMManager& DPMManager::setGFXCLK(int eng)
    {
        // no internal dependencies
        return *this;
    }

}