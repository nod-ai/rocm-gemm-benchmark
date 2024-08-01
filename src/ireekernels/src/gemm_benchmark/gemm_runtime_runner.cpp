#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>

#include "IREEGemm/Runtime.hpp"

#define MAX_ITERATIONS 200

struct IREEGemmRunner::RuntimeLoopData {
  bool failed;
  int num_iterations;
  int completed_iterations;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
  std::mutex mutex;
  std::condition_variable cv;
};

IREEGemmRunner::IREEGemmRunner(IREEGemmRuntimeState* state_ref,
                               IREEGemmDeviceStorage* device_storage_ref,
                               bool rotate_buffer)
    : vm_context(nullptr),
      vm_bytecode_module(nullptr),
      vm_inputs(nullptr),
      vm_outputs(nullptr),
      state(state_ref),
      device_storage(device_storage_ref),
      rotate_buffer(rotate_buffer) {
  loop_data = std::make_unique<RuntimeLoopData>();
  if (rotate_buffer) vm_rotating_inputs.resize(MAX_ITERATIONS);
}

IREEGemmRunner::~IREEGemmRunner() { cleanup(); }

void IREEGemmRunner::linkInput(std::vector<iree_hal_dim_t> shape,
                               std::string dtype, float* buffer) {
  iree_hal_dim_t size = 1;
  for (iree_hal_dim_t dim : shape) size *= dim;
  inputs.push_back((RuntimeInput){
      .shape = shape,
      .dtype = parseDtype(dtype),
      .input_buff = buffer,
      .device_buff = nullptr,
      .size = size,
      .offset = 0,
  });
}

int IREEGemmRunner::setupProblem(const std::string& module_path) {
  iree_status_t status;
  iree_file_contents_t* module_contents = nullptr;

  status =
      iree_file_read_contents(module_path.c_str(), IREE_FILE_READ_FLAG_DEFAULT,
                              iree_allocator_system(), &module_contents);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to load module path" << std::endl;
    return 1;
  }

  iree_vm_instance_t* vm_instance =
      iree_runtime_instance_vm_instance(state->instance);

  status = iree_vm_bytecode_module_create(
      vm_instance, module_contents->const_buffer,
      iree_file_contents_deallocator(module_contents), iree_allocator_system(),
      &vm_bytecode_module);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create bytecode buffer" << std::endl;
    return 1;
  }

  iree_vm_module_t* modules[] = {state->hal_module, vm_bytecode_module};
  status = iree_vm_context_create_with_modules(
      vm_instance, 0, IREE_ARRAYSIZE(modules), modules, iree_allocator_system(),
      &vm_context);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create VM context" << std::endl;
    return 1;
  }

  status = iree_vm_context_resolve_function(
      vm_context, iree_make_cstring_view("module.main_0"), &vm_function);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to resolve function" << std::endl;
    return 1;
  }

  int malloc_status = dispatchIOBuffers();
  if (malloc_status > 0) return malloc_status;

  return 0;
}

void IREEGemmRunner::preExecution(int num_iterations) {
  num_iterations = std::min(num_iterations, MAX_ITERATIONS);

  loop_data->num_iterations = num_iterations;
  loop_data->completed_iterations = 0;
  loop_data->start_time = std::chrono::steady_clock::now();

  runtime_states.resize(num_iterations);
}

void IREEGemmRunner::execute(int num_iterations) {
  iree_loop_inline_storage_t loop_storage = {{0xCD}, iree_ok_status()};
  iree_loop_t loop = iree_loop_inline_initialize(&loop_storage);

  for (int i = 0; i < num_iterations; ++i) {
    iree_vm_list_t* input = rotate_buffer ? vm_rotating_inputs[i] : vm_inputs;

    iree_vm_async_invoke(loop, &runtime_states[i], vm_context, vm_function,
                         IREE_VM_INVOCATION_FLAG_NONE, nullptr, input,
                         vm_outputs, iree_allocator_system(), runtimeCallback,
                         loop_data.get());
  }

  waitForCompletion(loop_data.get());
}

void IREEGemmRunner::cleanup() {
  for (auto& input : vm_rotating_inputs) {
    if (input) iree_vm_list_release(input);
  }
  if (vm_inputs) iree_vm_list_release(vm_inputs);
  if (vm_outputs) iree_vm_list_release(vm_outputs);
  if (vm_context) iree_vm_context_release(vm_context);
  if (vm_bytecode_module) iree_vm_module_release(vm_bytecode_module);
}

int IREEGemmRunner::dispatchIOBuffers() {
  iree_status_t status = iree_ok_status();

  // Create inputs and outputs
  if (rotate_buffer) {
    for (int i = 0; i < MAX_ITERATIONS; i++) {
      int input_status = createInputs(&vm_rotating_inputs[i]);
      if (input_status != 0) {
        std::cerr << "Failed to create rotating buffer input" << std::endl;
        return input_status;
      }
    }
  } else {
    int input_status = createInputs(&vm_inputs);
    if (input_status != 0) {
      std::cerr << "Failed to create input" << std::endl;
      return input_status;
    }
  }

  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                               iree_allocator_system(), &vm_outputs);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create VM list for outputs" << std::endl;
    return 1;
  }

  return 0;
}

int IREEGemmRunner::createInputs(iree_vm_list_t** inputs_out) {
  // Initialize input vm list
  iree_status_t status =
      iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                          iree_allocator_system(), inputs_out);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Failed to create VM list for inputs" << std::endl;
    return 1;
  }

  for (auto& input : inputs) {
    iree_hal_buffer_t* subspan = nullptr;

    // Create subspans
    if (device_storage->subspan(input.offset, input.size, &subspan) != 0)
      return 1;

    // Adjust offset (in case rotating buffer is used)
    input.offset += input.size;
    if (input.offset + input.size > device_storage->capacity())
      input.offset = 0;

    // Initialize buffer views
    iree_hal_buffer_view_t* input_buffer_view = nullptr;
    status = iree_hal_buffer_view_create(
        subspan, input.shape.size(), input.shape.data(), input.dtype,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(),
        &input_buffer_view);
    if (!iree_status_is_ok(status)) {
      std::cerr << "Failed to create buffer view" << std::endl;
      return 1;
    }

    iree_vm_ref_t input_ref = iree_hal_buffer_view_move_ref(input_buffer_view);
    iree_vm_list_push_ref_move(*inputs_out, &input_ref);

    iree_hal_buffer_release(subspan);
  }

  return 0;
}

iree_status_t IREEGemmRunner::runtimeCallback(void* user_data, iree_loop_t loop,
                                              iree_status_t status,
                                              iree_vm_list_t* outputs) {
  iree_vm_list_release(outputs);
  RuntimeLoopData* data = static_cast<RuntimeLoopData*>(user_data);
  std::unique_lock<std::mutex> lock(data->mutex);

  if (data->failed) {
    data->completed_iterations++;
    data->cv.notify_one();
    return status;
  }

  if (!iree_status_is_ok(status)) {
    data->completed_iterations++;
    data->failed = true;
    std::cerr << "Failed on iteration " << data->completed_iterations << "/"
              << data->num_iterations << std::endl;
    iree_status_fprint(stderr, status);
    data->cv.notify_one();
    return status;
  }

  if (++data->completed_iterations >= data->num_iterations) {
    data->end_time = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             data->end_time - data->start_time)
                             .count();
    double average_time_ms =
        static_cast<double>(total_time_ms) / data->num_iterations;
    // std::cout << "Executed " << data->num_iterations << " iterations\n"
    //           << "Total time: " << total_time_ms << " ms\n"
    //           << "Average time per iteration: " << average_time_ms << "
    //           ms\n";
    data->cv.notify_one();
  }

  return iree_ok_status();
}

void IREEGemmRunner::waitForCompletion(RuntimeLoopData* data) {
  std::unique_lock<std::mutex> lock(data->mutex);
  data->cv.wait(lock, [data] {
    return data->completed_iterations >= data->num_iterations;
  });
}

iree_hal_element_type_t parseDtype(const std::string& dtype) {
  if (dtype == "fp16") return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  if (dtype == "fp32") return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  if (dtype == "bf16") return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
  return IREE_HAL_ELEMENT_TYPE_NONE;
}

iree_device_size_t deviceSize(iree_hal_dim_t dim,
                              iree_hal_element_type_t dtype) {
  iree_device_size_t device_size = 0;
  iree_hal_buffer_compute_view_size(
      1, &dim, dtype, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &device_size);
  return device_size;
}