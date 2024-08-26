BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    (
        stdout_v,
        stderr_v,
    ) = (
        proc.stdout,
        proc.stderr,
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout
    logging.getLogger().error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stderr

def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results

def bench_summary_process(ret_value, output):
    if ret_value == 1:
        # Output should have already been logged earlier.
        logging.getLogger().error("Running convolution benchmark failed. Exiting.")
        return

    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)
    benchmark_mean_time = float(benchmark_results[10].time.split()[0])
    
    return benchmark_mean_time

def write_results_to_csv(results : list[tuple] | list[list] | list[dict], output_filename: str):
    if len(results) == 0:
        print('No valid results')
        return
    
    fieldnames = [
        'index', 
        'tag',
        'name',
        'image', 
        'conv_filter', 
        'output', 
        'input_dtype', 
        'output_dtype',
        'mean_microseconds',
        'arithmetic_intensity',
        'tflops',
        'ok'
    ]

    with open(output_filename, 'w', newline='') as f:
        if isinstance(results[0], list) or isinstance(results[0], tuple):
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        elif isinstance(results[0], dict):
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
        else:
            print('Invalid result format')
            return
        
        for result in results:
            writer.writerow(result)

def roofline(results=None, **kwargs):
    """Generate a roofline plot of GEMM performance from multiple result files and save raw data as CSV."""
    if results is None:
        raise ValueError("No result files provided")

    files = results.split(',')
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    
    plt.figure(figsize=(12, 8))

    for idx, result_file in enumerate(files):
        data = []
        with open(result_file.strip(), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = {k: float(v) if k not in ['ok', 'tag', 'name', 'A', 'B', 'dtype', "image", "conv_filter", "output", "input_dtype", "output_dtype"] else v for k, v in row.items()}
                row['ok'] = True if 'ok' not in row else row['ok'] == 'True'
                data.append(row)
        
        x = [item['arithmetic_intensity'] for item in data]
        y = [item['tflops'] for item in data]
        
        plt.scatter(x, y, alpha=0.6, color=next(colors), label=result_file.strip())
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (TFLOP/s)')
    plt.title('Roofline Plot of GEMM Performance')
    
    peak_memory_bandwidth = 5.3
    peak_compute = 1300

    x_range = np.logspace(np.log10(min(x)), np.log10(max(max(x), 150)), 100)
    y_memory = peak_memory_bandwidth * x_range
    y_compute = np.full_like(x_range, peak_compute)
    
    plt.plot(x_range, y_memory, 'r-', label='Memory Bound')
    plt.plot(x_range, y_compute, 'g-', label='Compute Bound')
    plt.plot(x_range, np.minimum(y_memory, y_compute), 'k-', linewidth=2, label='Roofline')
    
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.text(x_range[-1], peak_compute, f'{peak_compute:.1f} TFLOP/s', 
             verticalalignment='bottom', horizontalalignment='right')

    plt.savefig('roofline_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Roofline plot saved as 'roofline_plot.png'")
