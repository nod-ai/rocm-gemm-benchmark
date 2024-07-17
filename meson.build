
project('gemm-bench', 'cpp', default_options : ['cpp_std=c++2a'])

arch = 'amd'

cpp = meson.get_compiler('cpp')

#
# gemm-bench worker
#

zmq_dep = cpp.find_library('libzmq')
hiprand_dep = cpp.find_library('libhiprand', dirs: ['/opt/rocm/lib'])

gemm_bench_src = [
        'src/benchmark/main.cpp',
        'src/benchmark/gemm-bench.cpp',
        'src/benchmark/FrequencyMonitor.cpp',
        'src/benchmark/Timer.cpp'
        ]

gemm_bench_deps = [zmq_dep, hiprand_dep]

rocblas_dep = cpp.find_library('librocblas', dirs: ['/opt/rocm/lib'])
rocsmi_dep = cpp.find_library('rocm_smi64', dirs: ['/opt/rocm/lib'])

iree_lib_folder = meson.current_source_dir() + '/src/ireekernelsbuild/lib'
iree_include_folder = 'src/ireekernels/include'
iree_deps = [
        cpp.find_library('libgemm_mlir_lib', dirs: [iree_lib_folder]),
        cpp.find_library('libiree_gemm_compile_lib', dirs: [iree_lib_folder]),
        cpp.find_library('libiree_gemm_benchmark_lib', dirs: [iree_lib_folder])
]

gemm_bench_src += 'src/benchmark/DataInitialization.cpp'
gemm_bench_src += 'src/benchmark/run-rocblas.cpp'
gemm_bench_src += 'src/benchmark/run_iree.cpp'
gemm_bench_deps += [rocblas_dep, rocsmi_dep]
gemm_bench_deps += iree_deps
gemm_bench_incs = [iree_include_folder]

fpic = ['-fPIC']

executable('gemm-bench', gemm_bench_src, 
        dependencies: gemm_bench_deps, 
        include_directories: gemm_bench_incs,
        c_args: fpic,
        cpp_args: fpic,
        link_args: fpic
)

#
# gbm python module
#

python_dep = dependency('python3-embed')

gemm_module_src = ['src/benchmark/gemm-bench_module.cpp']

gemm_module_deps = [python_dep]

shared_library('gbm', gemm_module_src, dependencies: gemm_module_deps, name_prefix: '')

#
# frequency monitor python module
#

frequency_monitor_module_src = ['src/benchmark/FrequencyMonitor.cpp']

frequency_monitor_module_deps = [python_dep, rocsmi_dep]

shared_library('frequencymonitor', frequency_monitor_module_src, dependencies: frequency_monitor_module_deps, name_prefix: '')