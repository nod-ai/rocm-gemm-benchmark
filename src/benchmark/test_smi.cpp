#include <iostream>
#include <rocm_smi/rocm_smi.h>

void check_status(rsmi_status_t status, const char* msg)
{
    if(status != RSMI_STATUS_SUCCESS)
    {
        const char* err_str;
        rsmi_status_string(status, &err_str);
        std::cerr << "Error: " << msg << " - " << err_str << std::endl;
    }
}

int main()
{
    // Initialize ROCm SMI
    rsmi_status_t status = rsmi_init(0);
    check_status(status, "Initializing ROCm SMI");
    if(status != RSMI_STATUS_SUCCESS)
        return -1;

    // Get the number of devices
    uint32_t device_count = 0;
    status                = rsmi_num_monitor_devices(&device_count);
    check_status(status, "Getting the number of devices");
    if(status != RSMI_STATUS_SUCCESS)
    {
        rsmi_shut_down();
        return -1;
    }

    // Assume device 0 for this example
    uint32_t device_index = 0;
    if(device_index >= device_count)
    {
        std::cerr << "Invalid device index" << std::endl;
        rsmi_shut_down();
        return -1;
    }

    // Set the FCLK DPM level
    rsmi_clk_type_t fclk_type      = RSMI_CLK_TYPE_DF;
    uint32_t        fclk_dpm_level = 0b0001; // Example level, set as needed
    status = rsmi_dev_gpu_clk_freq_set(device_index, fclk_type, fclk_dpm_level);
    check_status(status, "Setting FCLK DPM level");
    if(status != RSMI_STATUS_SUCCESS)
    {
        rsmi_shut_down();
        return -1;
    }
    std::cout << "FCLK DPM level set to " << fclk_dpm_level << std::endl;

    // Set the MCLK DPM level
    rsmi_clk_type_t mclk_type      = RSMI_CLK_TYPE_MEM;
    uint32_t        mclk_dpm_level = 0b0111; // Example level, set as needed
    status = rsmi_dev_gpu_clk_freq_set(device_index, mclk_type, mclk_dpm_level);
    check_status(status, "Setting MCLK DPM level");
    if(status != RSMI_STATUS_SUCCESS)
    {
        rsmi_shut_down();
        return -1;
    }
    std::cout << "MCLK DPM level set to " << mclk_dpm_level << std::endl;

    // Shut down ROCm SMI
    status = rsmi_shut_down();
    check_status(status, "Shutting down ROCm SMI");
    if(status != RSMI_STATUS_SUCCESS)
        return -1;

    return 0;
}
