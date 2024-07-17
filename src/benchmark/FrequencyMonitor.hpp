#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

namespace Frequency
{
    struct Statistics
    {
        uint64_t count;
        double   mean, median;
        float    min, max;
    };

    Statistics statistics(std::vector<float> const& x);

    class Monitor
    {
    public:
        using resolution_type = std::chrono::duration<size_t, std::nano>;

        ~Monitor();

        /**
	 * Set HIP device to monitor.
	 */
        Monitor& setDevice(int deviceId);

        /**
	 * Set monitoring resolution (time between reading samples).
	 */
        Monitor& setResolution(resolution_type const& resolution);

        /**
	 * Start monitoring.
	 */
        Monitor& start();

        /**
	 * Stop monitoring.
	 */
        Monitor& stop();

        /**
	 * Tuple of (system, memory, datafrabric) frequency statistics
	 */
        std::tuple<Statistics, Statistics, Statistics> statistics() const;

        /**
	 * Raw system frequencies in MHz.
	 */
        std::vector<float> systemFrequency() const;

        /**
	 * Raw memory frequencies in MHz.
	 */
        std::vector<float> memoryFrequency() const;

        /**
	 * Raw data fabric frequencies in MHz.
	 */
        std::vector<float> dataFrequency() const;

        /**
	 * Raw temperatures.
	 */
        std::vector<float> temperature() const;

        /**
	 * Raw power draws.
	 */
        std::vector<float> power() const;

    private:
        void collect();
        bool active() const;
        void assertActive() const;
        void assertNotActive() const;

        std::atomic<bool> m_stop;
        std::thread       m_thread;

        resolution_type m_resolution = std::chrono::microseconds(100);

        std::vector<float> m_systemFrequency, m_memoryFrequency, m_dataFrequency, m_temperature,
            m_power;

        int m_smiDeviceIndex = -1;
    };

    Monitor* getFrequencyMonitor();

    uint32_t GetROCmSMIIndex(int hipDeviceIndex);

    class DPMManager
    {
    public:
        ~DPMManager();

        DPMManager& setDevice(int deviceId);
        DPMManager& setFCLK(int dpm);
        DPMManager& setMCLK(int dpm);
        DPMManager& setGFXCLK(int eng);

        int getFCLK() const;
        int getMCLK() const;
        int getGFXCLK() const;

    private:
        int m_smiDeviceIndex = -1;
    };

    DPMManager* getDPMManager();
}