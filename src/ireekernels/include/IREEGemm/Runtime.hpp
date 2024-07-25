#ifndef IREE_GEMM_RUNTIME
#define IREE_GEMM_RUNTIME

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

iree_hal_element_type_t parseDtype(const std::string& dtype);
iree_device_size_t deviceSize(iree_hal_dim_t dim,
                              iree_hal_element_type_t dtype);

class IREEGemmRuntimeState {
 public:
  IREEGemmRuntimeState(const std::string& device_uri);
  ~IREEGemmRuntimeState();

  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_vm_module_t* hal_module;

 private:
  int initialize(const std::string& device_uri);
  void cleanup();
};

class IREEGemmDeviceStorage {
 public:
  IREEGemmDeviceStorage(std::string dtype_str);
  ~IREEGemmDeviceStorage();

  iree_device_size_t capacity();
  int allocate(iree_hal_device_t* device, iree_device_size_t capacity,
               float* input_buff);
  int subspan(iree_hal_dim_t offset, iree_hal_dim_t size,
              iree_hal_buffer_t** subspan_out);

 private:
  iree_hal_element_type_t _dtype;
  iree_hal_buffer_t* _buffer;
  iree_device_size_t _capacity;

  void cleanup();
};

class IREEGemmRunner {
 public:
  struct RuntimeInput {
    std::vector<iree_hal_dim_t> shape;
    iree_hal_element_type_t dtype;
    float* input_buff;
    iree_hal_buffer_t* device_buff;
    size_t size;
    size_t offset;
  };

  IREEGemmRunner(IREEGemmRuntimeState* state_ref,
                 IREEGemmDeviceStorage* device_storage_ref, bool rotate_buffer);
  ~IREEGemmRunner();

  void linkInput(std::vector<iree_hal_dim_t> shape, std::string dtype,
                 float* buffer = nullptr);
  int setupProblem(const std::string& module_path);
  void execute(int num_iterations);
  void cleanup();

 private:
  bool rotate_buffer;

  iree_vm_context_t* vm_context;
  iree_vm_module_t* vm_bytecode_module;
  iree_vm_function_t vm_function;
  iree_vm_list_t* vm_inputs;
  std::vector<iree_vm_list_t*> vm_rotating_inputs;
  iree_vm_list_t* vm_outputs;

  struct RuntimeLoopData;
  std::unique_ptr<RuntimeLoopData> loop_data;

  IREEGemmDeviceStorage* device_storage;
  IREEGemmRuntimeState* state;

  std::vector<RuntimeInput> inputs;

  int dispatchIOBuffers();
  int createInputs(iree_vm_list_t** inputs_out);
  static iree_status_t runtimeCallback(void* user_data, iree_loop_t loop,
                                       iree_status_t status,
                                       iree_vm_list_t* outputs);
  void waitForCompletion(RuntimeLoopData* data);
};

#endif  // IREE_GEMM_RUNTIME