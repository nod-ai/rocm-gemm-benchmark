#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "IREEGemm/Runtime.h"
#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#define CHECK_IREE_STATUS(status, message)                      \
  if (!iree_status_is_ok(status)) {                             \
    fprintf(stderr, "%s: %s\n", message,                        \
            iree_status_code_string(iree_status_code(status))); \
    return 1;                                                   \
  }

#define CHECK_INT_STATUS(error_code, message) \
  if (error_code) {                           \
    fprintf(stderr, "%s\n", message);         \
    return error_code;                        \
  }

#define MAX_ITERATIONS 200

typedef struct runtime_storage_t {
  bool rotate_buffer;
  iree_hal_buffer_t* bucket_A;
  iree_hal_buffer_t* bucket_B;
  iree_hal_element_type_t dtype;
  size_t capacity;
  size_t size_A;
  size_t size_B;
  size_t offset_A;
  size_t offset_B;
} runtime_storage_t;

typedef struct runtime_loop_data_t {
  bool failed;
  int num_iterations;
  int completed_iterations;
  clock_t start_time;
  clock_t end_time;
  pthread_mutex_t* lock;
  pthread_cond_t* cond;
} runtime_loop_data_t;

typedef struct runtime_state_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_vm_context_t* context;
  iree_vm_module_t* hal_module;
  iree_vm_module_t* bytecode_module;
  iree_vm_function_t function;
  iree_vm_list_t* inputs;
  iree_vm_list_t** rotating_inputs;
  iree_vm_list_t* outputs;
  runtime_loop_data_t* loop_data;
  runtime_storage_t* storage;
} runtime_state_t;

iree_hal_element_type_t parse_dtype(const char* dtype) {
  if (strcmp(dtype, "fp16") == 0) return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  if (strcmp(dtype, "fp32") == 0) return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  if (strcmp(dtype, "bf16") == 0) return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
  return IREE_HAL_ELEMENT_TYPE_NONE;
}

iree_status_t ireeGemmRuntimeCallback(void* user_data, iree_loop_t loop,
                                      iree_status_t status,
                                      iree_vm_list_t* outputs) {
  iree_vm_list_release(outputs);
  runtime_loop_data_t* s = (runtime_loop_data_t*)user_data;
  pthread_mutex_lock(s->lock);

  if (s->failed) {
    s->completed_iterations++;
    pthread_cond_signal(s->cond);
    pthread_mutex_unlock(s->lock);
    return status;
  }

  if (!iree_status_is_ok(status)) {
    s->completed_iterations++;
    s->failed = true;
    fprintf(stderr, "Failed on iteration %d/%d\n", s->completed_iterations,
            s->num_iterations);
    iree_status_fprint(stderr, status);
    pthread_cond_signal(s->cond);
    pthread_mutex_unlock(s->lock);
    return status;
  }

  if (++s->completed_iterations >= s->num_iterations) {
    s->end_time = clock();
    double total_time_ms =
        (double)(s->end_time - s->start_time) / CLOCKS_PER_SEC * 1000;
    double average_time_ms = total_time_ms / s->num_iterations;
    // printf(
    //     "Executed %d iterations\nTotal time: %.3f ms\nAverage time per "
    //     "iteration: %.3f ms\n",
    //     s->num_iterations, total_time_ms, average_time_ms);
    pthread_cond_signal(s->cond);
  }

  pthread_mutex_unlock(s->lock);
  return iree_ok_status();
}

void ireeGemmRuntimeWaitForCompletion(runtime_loop_data_t* s) {
  pthread_mutex_lock(s->lock);
  while (s->completed_iterations < s->num_iterations) {
    pthread_cond_wait(s->cond, s->lock);
  }
  pthread_mutex_unlock(s->lock);
}

int ireeGemmRuntimeInitialize(const char* device_uri, bool rotate_buffer,
                              runtime_state_t** state_ref) {
  runtime_state_t* state = (runtime_state_t*)malloc(sizeof(runtime_state_t));
  memset(state, 0, sizeof(runtime_state_t));
  iree_status_t status;

  iree_runtime_instance_t* instance = NULL;
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(&instance_options,
                                        iree_allocator_system(), &instance);
  CHECK_IREE_STATUS(status, "Failed to create runtime instance");
  state->instance = instance;

  iree_hal_device_t* device = NULL;
  status = iree_hal_create_device(
      iree_runtime_instance_driver_registry(instance),
      iree_make_cstring_view(device_uri),
      iree_runtime_instance_host_allocator(instance), &device);
  CHECK_IREE_STATUS(status, "Failed to create HAL device");
  state->device = device;

  iree_vm_module_t* hal_module = NULL;
  status = iree_hal_module_create(
      iree_runtime_instance_vm_instance(instance), 1, &device,
      IREE_HAL_MODULE_FLAG_NONE, iree_runtime_instance_host_allocator(instance),
      &hal_module);
  CHECK_IREE_STATUS(status, "Failed to create HAL module");
  state->hal_module = hal_module;

  if (rotate_buffer) {
    state->rotating_inputs =
        (iree_vm_list_t**)malloc(sizeof(iree_vm_list_t*) * MAX_ITERATIONS);
  }

  runtime_storage_t* storage =
      (runtime_storage_t*)malloc(sizeof(runtime_storage_t));
  memset(storage, 0, sizeof(runtime_storage_t));
  storage->rotate_buffer = rotate_buffer;
  state->storage = storage;

  runtime_loop_data_t* loop_data =
      (runtime_loop_data_t*)malloc(sizeof(runtime_loop_data_t));
  memset(loop_data, 0, sizeof(runtime_loop_data_t));
  loop_data->lock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
  if (pthread_mutex_init(loop_data->lock, NULL) != 0) {
    fprintf(stderr, "Failed to create mutex\n");
    return 2;
  }
  loop_data->cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
  if (pthread_cond_init(loop_data->cond, NULL) != 0) {
    fprintf(stderr, "Failed to create condition variable\n");
    return 2;
  }
  state->loop_data = loop_data;

  *state_ref = state;

  return 0;
}

inline iree_device_size_t iree_device_size(iree_hal_dim_t dim,
                                           iree_hal_element_type_t dtype) {
  iree_device_size_t device_size = 0;
  iree_hal_buffer_compute_view_size(1, (iree_hal_dim_t[]){dim}, dtype,
                                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                    &device_size);
  return device_size;
}

int ireeGemmRuntimeCreateInputs(runtime_storage_t* storage,
                                const iree_hal_dim_t shape_A[2],
                                const iree_hal_dim_t shape_B[2],
                                iree_vm_list_t** inputs_out) {
  iree_status_t status = iree_ok_status();

  iree_hal_buffer_t* buffer_A = NULL;
  iree_hal_buffer_t* buffer_B = NULL;

  // create subspans
  // printf("\nCapacity %zu vs. Size_A %zu + Offset_A %zu = %zu\n",
  //        storage->capacity, storage->size_A, storage->offset_A,
  //        storage->size_A + storage->offset_A);
  status = iree_hal_buffer_subspan(
      storage->bucket_A, iree_device_size(storage->offset_A, storage->dtype),
      iree_device_size(storage->size_A, storage->dtype), &buffer_A);
  CHECK_IREE_STATUS(status, "Failed to create buffer subspan A");
  status = iree_hal_buffer_subspan(
      storage->bucket_B, iree_device_size(storage->offset_B, storage->dtype),
      iree_device_size(storage->size_B, storage->dtype), &buffer_B);
  CHECK_IREE_STATUS(status, "Failed to create buffer subspan B");

  // Adjust offset (in case rotating buffer is used)
  storage->offset_A += storage->size_A;
  storage->offset_B += storage->size_B;
  if (storage->offset_A + storage->size_A > storage->capacity)
    storage->offset_A = 0;
  if (storage->offset_B + storage->size_B > storage->capacity)
    storage->offset_B = 0;

  // initialize buffer views
  iree_hal_buffer_view_t* input_buffer_view_A = NULL;
  iree_hal_buffer_view_t* input_buffer_view_B = NULL;
  status = iree_hal_buffer_view_create(buffer_A, 2, shape_A, storage->dtype,
                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                       iree_allocator_system(),
                                       &input_buffer_view_A);
  CHECK_IREE_STATUS(status, "Failed to create buffer view A");
  status = iree_hal_buffer_view_create(buffer_B, 2, shape_B, storage->dtype,
                                       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                       iree_allocator_system(),
                                       &input_buffer_view_B);
  CHECK_IREE_STATUS(status, "Failed to create buffer view B");

  // initialize input vm list
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                               iree_allocator_system(), inputs_out);
  CHECK_IREE_STATUS(status, "Failed to create VM list for inputs");
  iree_vm_ref_t input_ref_A =
      iree_hal_buffer_view_move_ref(input_buffer_view_A);
  iree_vm_ref_t input_ref_B =
      iree_hal_buffer_view_move_ref(input_buffer_view_B);
  iree_vm_list_push_ref_move(*inputs_out, &input_ref_A);
  iree_vm_list_push_ref_move(*inputs_out, &input_ref_B);

  iree_hal_buffer_release(buffer_A);
  iree_hal_buffer_release(buffer_B);
}

int ireeGemmRuntimeDispatchIOBuffers(runtime_state_t* s, int M, int K, int N,
                                     bool transpose_A, bool transpose_B,
                                     const char* dtype, const float* input_A,
                                     const float* input_B) {
  iree_hal_element_type_t parsed_dtype = parse_dtype(dtype);
  if (parsed_dtype == IREE_HAL_ELEMENT_TYPE_NONE)
    return (int)IREE_STATUS_INVALID_ARGUMENT;

  runtime_storage_t* storage = s->storage;
  storage->bucket_A = NULL;
  storage->bucket_B = NULL;
  storage->offset_A = 0;
  storage->offset_B = 0;
  storage->size_A = (size_t)M * K;
  storage->size_B = (size_t)K * N;
  storage->dtype = parsed_dtype;
  if (!storage->rotate_buffer)
    storage->capacity = iree_max(storage->size_A, storage->size_B);

  size_t host_capacity = storage->capacity * sizeof(float);
  size_t device_capacity = iree_device_size(storage->capacity, parsed_dtype);

  float* A = (float*)input_A;
  float* B = (float*)input_B;

  if (!input_A) {
    A = (float*)malloc(host_capacity);
    memset(A, 0, host_capacity);
  }
  if (!input_B) {
    B = (float*)malloc(host_capacity);
    memset(B, 0, host_capacity);
  }

  iree_hal_dim_t shape_A[] = {transpose_A ? K : M, transpose_A ? M : K};
  iree_hal_dim_t shape_B[] = {transpose_B ? N : K, transpose_B ? K : N};
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(s->device);

  iree_status_t status;
  iree_hal_buffer_params_t device_buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL};

  status = iree_hal_allocator_allocate_buffer(
      allocator, device_buffer_params, device_capacity, &storage->bucket_A);
  CHECK_IREE_STATUS(status, "Failed to allocate bucket A");
  status = iree_hal_allocator_allocate_buffer(
      allocator, device_buffer_params, device_capacity, &storage->bucket_B);
  CHECK_IREE_STATUS(status, "Failed to allocate bucket B");

  // Write data to buffers
  status = iree_hal_device_transfer_h2d(
      s->device, A, storage->bucket_A, 0,
      iree_min(host_capacity, device_capacity),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  CHECK_IREE_STATUS(status, "Failed to write to bucket A");
  status = iree_hal_device_transfer_h2d(
      s->device, B, storage->bucket_B, 0,
      iree_min(host_capacity, device_capacity),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  CHECK_IREE_STATUS(status, "Failed to write to bucket B");

  // Create inputs and outputs
  if (storage->rotate_buffer) {
    for (int i = 0; i < MAX_ITERATIONS; i++) {
      int input_status = ireeGemmRuntimeCreateInputs(storage, shape_A, shape_B,
                                                     &s->rotating_inputs[i]);
      CHECK_INT_STATUS(input_status, "Failed to create rotating buffer input");
    }
  } else {
    int input_status =
        ireeGemmRuntimeCreateInputs(storage, shape_A, shape_B, &s->inputs);
    CHECK_INT_STATUS(input_status, "Failed to create rotating buffer input");
  }

  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                               iree_allocator_system(), &s->outputs);
  CHECK_IREE_STATUS(status, "Failed to create VM list for outputs");

  // Free host buffers
  if (!input_A) free(A);
  if (!input_B) free(B);

  return 0;
}

int ireeGemmRuntimeSetupProblem(runtime_state_t* s, const char* module_path,
                                int M, int K, int N, bool transpose_A,
                                bool transpose_B, const char* dtype,
                                const float* A, const float* B,
                                size_t mem_capacity) {
  iree_status_t status;
  iree_file_contents_t* module_contents = NULL;

  status = iree_file_read_contents(module_path, IREE_FILE_READ_FLAG_DEFAULT,
                                   iree_allocator_system(), &module_contents);
  CHECK_IREE_STATUS(status, "Failed to load module path");

  iree_vm_instance_t* vm_instance =
      iree_runtime_instance_vm_instance(s->instance);

  status = iree_vm_bytecode_module_create(
      vm_instance, module_contents->const_buffer,
      iree_file_contents_deallocator(module_contents), iree_allocator_system(),
      &s->bytecode_module);
  CHECK_IREE_STATUS(status, "Failed to create bytecode buffer");

  iree_vm_module_t* modules[] = {s->hal_module, s->bytecode_module};
  status = iree_vm_context_create_with_modules(
      vm_instance, 0, IREE_ARRAYSIZE(modules), modules, iree_allocator_system(),
      &s->context);
  CHECK_IREE_STATUS(status, "Failed to create VM context");

  status = iree_vm_context_resolve_function(
      s->context, iree_make_cstring_view("module.main_0"), &s->function);
  CHECK_IREE_STATUS(status, "Failed to resolve function");

  s->storage->capacity = mem_capacity;
  int malloc_status = ireeGemmRuntimeDispatchIOBuffers(
      s, M, K, N, transpose_A, transpose_B, dtype, A, B);
  if (malloc_status > 0) return malloc_status;

  return 0;
}

void ireeGemmRuntimeExecute(runtime_state_t* s, int num_iterations) {
  num_iterations = iree_min(num_iterations, MAX_ITERATIONS);

  iree_loop_inline_storage_t storage = {{0xCD}, iree_ok_status()};
  iree_loop_t loop = iree_loop_inline_initialize(&storage);

  runtime_loop_data_t* loop_data = s->loop_data;

  loop_data->num_iterations = num_iterations;
  loop_data->completed_iterations = 0;
  loop_data->start_time = clock();

  iree_vm_async_invoke_state_t states[num_iterations];

  for (int i = 0; i < num_iterations; ++i) {
    iree_vm_list_t* input =
        s->storage->rotate_buffer ? s->rotating_inputs[i] : s->inputs;

    iree_vm_async_invoke(loop, &states[i], s->context, s->function,
                         IREE_VM_INVOCATION_FLAG_NONE, NULL, input, s->outputs,
                         iree_allocator_system(), ireeGemmRuntimeCallback,
                         loop_data);
  }

  ireeGemmRuntimeWaitForCompletion(loop_data);
}

void ireeGemmRuntimeCleanup(runtime_state_t* s) {
  if (!s) return;
  if (s->storage) {
    runtime_storage_t* storage = s->storage;
    if (storage->bucket_A) iree_hal_buffer_release(storage->bucket_A);
    if (storage->bucket_B) iree_hal_buffer_release(storage->bucket_B);
    free(storage);
  }
  if (s->loop_data) {
    runtime_loop_data_t* loop_data = s->loop_data;
    pthread_mutex_destroy(loop_data->lock);
    pthread_cond_destroy(loop_data->cond);
    free(loop_data->lock);
    free(loop_data->cond);
    free(loop_data);
  }
  if (s->rotating_inputs) {
    iree_vm_list_t** rotating_inputs = s->rotating_inputs;
    for (int i = 0; i < MAX_ITERATIONS; i++)
      if (rotating_inputs[i]) iree_vm_list_release(s->rotating_inputs[i]);
    free(rotating_inputs);
  }
  if (s->inputs) iree_vm_list_release(s->inputs);
  if (s->outputs) iree_vm_list_release(s->outputs);
  if (s->context) iree_vm_context_release(s->context);
  if (s->bytecode_module) iree_vm_module_release(s->bytecode_module);
  if (s->device) iree_hal_device_release(s->device);
  if (s->hal_module) iree_vm_module_release(s->hal_module);
  if (s->instance) iree_runtime_instance_release(s->instance);
  free(s);
}