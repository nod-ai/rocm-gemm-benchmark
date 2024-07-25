#include <algorithm>
#include <cstring>
#include <iostream>

#include "IREEGemm/Runtime.hpp"

IREEGemmRuntimeState::IREEGemmRuntimeState(const std::string& device_uri)
    : instance(nullptr), device(nullptr), hal_module(nullptr) {
  initialize(device_uri);
};

IREEGemmRuntimeState::~IREEGemmRuntimeState() { cleanup(); }

int IREEGemmRuntimeState::initialize(const std::string& device_uri) {
  iree_status_t status;

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(&instance_options,
                                        iree_allocator_system(), &instance);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create runtime instance" << std::endl;
    return 1;
  }

  status = iree_hal_create_device(
      iree_runtime_instance_driver_registry(instance),
      iree_make_cstring_view(device_uri.c_str()),
      iree_runtime_instance_host_allocator(instance), &device);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create HAL device" << std::endl;
    return 1;
  }

  status = iree_hal_module_create(
      iree_runtime_instance_vm_instance(instance), 1, &device,
      IREE_HAL_MODULE_FLAG_NONE, iree_runtime_instance_host_allocator(instance),
      &hal_module);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create HAL module" << std::endl;
    return 1;
  }

  return 0;
}

void IREEGemmRuntimeState::cleanup() {
  if (device) iree_hal_device_release(device);
  if (hal_module) iree_vm_module_release(hal_module);
  if (instance) iree_runtime_instance_release(instance);
}