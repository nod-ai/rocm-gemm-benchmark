#include <algorithm>
#include <cstring>
#include <iostream>

#include "IREEGemm/Runtime.hpp"

IREEGemmDeviceStorage::IREEGemmDeviceStorage(std::string dtype_str)
    : _dtype(parseDtype(dtype_str)), _capacity(0), _buffer(nullptr) {}

IREEGemmDeviceStorage::~IREEGemmDeviceStorage() { cleanup(); }

void IREEGemmDeviceStorage::cleanup() {
  if (_buffer) iree_hal_buffer_release(_buffer);
}

iree_device_size_t IREEGemmDeviceStorage::capacity() { return _capacity; }

int IREEGemmDeviceStorage::allocate(iree_hal_device_t* device,
                                    iree_device_size_t capacity,
                                    const float* input_buff) {
  if (_buffer) iree_hal_buffer_release(_buffer);

  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  iree_hal_buffer_params_t device_buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL};

  size_t host_capacity = capacity * sizeof(float);
  size_t device_capacity = deviceSize(capacity, _dtype);

  std::vector<float> host_buff(host_capacity / sizeof(float), 0);
  if (input_buff)
    std::copy(input_buff, input_buff + capacity, host_buff.begin());

  iree_status_t status = iree_ok_status();
  status = iree_hal_allocator_allocate_buffer(
      device_allocator, device_buffer_params, device_capacity, &_buffer);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to allocate bucket: "
              << iree_status_code_string(iree_status_code(status)) << std::endl;
    return 1;
  }

  status = iree_hal_device_transfer_h2d(
      device, host_buff.data(), _buffer, 0,
      std::min(host_capacity, device_capacity),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to write to bucket" << std::endl;
    return 1;
  }

  _capacity = capacity;
  return 0;
}

int IREEGemmDeviceStorage::subspan(iree_hal_dim_t offset, iree_hal_dim_t size,
                                   iree_hal_buffer_t** subspan_out) {
  iree_hal_buffer_t* subspan = nullptr;

  iree_status_t status = iree_hal_buffer_subspan(
      _buffer, deviceSize(offset, _dtype), deviceSize(size, _dtype), &subspan);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create buffer subspan A" << std::endl;
    return 1;
  }

  *subspan_out = subspan;
  return 0;
}