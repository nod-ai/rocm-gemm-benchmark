import argparse
import csv
import datetime
import h5py
import logging
import pathlib
import sys

import gemmbench
import pandas

import numpy as np
import matplotlib.pyplot as plt

from gbm import Problem, Solution, Configuration

from operator import itemgetter
from itertools import cycle


def run(top=None, suite=None, output=None, no_shuffle=False, repeat=10, backends="rocblas", **kwargs):
    """Run problem suite."""

    problems = gemmbench.definitions.load_suite(top, suite)
    solutions = [Solution(backend) for backend in backends.split(',')]
    configurations = gemmbench.definitions.load_configurations(top)

    if repeat is None:
        repeat = 10

    if no_shuffle is None:
        no_shuffle = False

    if output is None:
        output = (
            datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
            + "-"
            + gemmbench.git.short_hash()
            + ".hdf"
        )
    output = pathlib.Path(output)

    print(f"Writing results to: {output}")

    with h5py.File(output, "w") as h5:
        for i, (problem, solution, configuration, result) in enumerate(
            gemmbench.server.run_problems(
                problems,
                solutions,
                configurations,
                shuffle=not no_shuffle,
                repeat=repeat,
            )
        ):
            params = {}
            params.update(problem)
            params.update(solution)
            params.update(configuration)

            experiment_group = h5.create_group(str(i))
            for k in params.keys():
                experiment_group.attrs[k] = params[k]
            for k in result.keys():
                if not isinstance(result[k], np.ndarray):
                    experiment_group.attrs[k] = result[k]

            hwmon_group = experiment_group.create_group("hwmon")
            hwmon_group.create_dataset("system", data=result["sclk"])
            hwmon_group.create_dataset("memory", data=result["mclk"])
            hwmon_group.create_dataset("temperature", data=result["temperature"])
            hwmon_group.create_dataset("power", data=result["power"])

            h5.flush()

def roofline(results=None, **kwargs):
    """Generate a roofline plot of GEMM performance from multiple result files."""
    if results is None:
        raise ValueError("No result files provided")

    files = results.split(',')
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    
    plt.figure(figsize=(12, 8))
    
    for idx, result_file in enumerate(files):
        data = []
        with h5py.File(result_file.strip(), "r") as h5:
            for serial in h5.keys():
                experiment_group = h5[serial]
                data.append(dict(serial=int(serial), **experiment_group.attrs))
        
        for item in data:
            M, N, K = item['M'], item['N'], item['K']
            flops = 2 * M * N * K
            bytes = M * K + N * K + M * N
            item['arithmetic_intensity'] = flops / bytes
            item['tflops'] = (flops / 1e12) / (item['mean_microseconds'] / 1e6)
        
        x = [item['arithmetic_intensity'] for item in data]
        y = [item['tflops'] for item in data]
        
        plt.scatter(x, y, alpha=0.6, color=next(colors), label=result_file.strip())
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (TFLOP/s)')
    plt.title('Roofline Plot of GEMM Performance')
    
    peak_memory_bandwidth = 5.3
    peak_compute = 980.6
    
    x_range = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
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

def summary(results=None, **kwargs):
    """Print a summary of the results in an HDF file."""

    sorting = [
        "tag",
        "dtype",
        "M",
        "N",
        "K",
        "A",
        "B",
        "fclk",
        "mclk",
        "mean_microseconds",
    ]
    columns = [
        "{tag:24s}",
        "{ok:1}",
        "{device:1d}",
        "{dtype:4s}",
        "{M:7d}",
        "{N:7d}",
        "{K:7d}",
        "{AB:2s}",
        "{fclk:1d}/{mclk:1d}",
        "{mean_microseconds:10.2f}us",
        "{serial:6d}",
        "{min_sclk:6.1f}/{mean_sclk:6.1f}/{max_sclk:6.1f}",
        "{min_mclk:6.1f}/{mean_mclk:6.1f}/{max_mclk:6.1f}",
        "{min_fclk:6.1f}/{mean_fclk:6.1f}/{max_fclk:6.1f}",
    ]
    columns = " ".join(columns)

    summary = []
    with h5py.File(results, "r") as h5:
        for serial in h5.keys():
            experiment_group = h5[serial]
            summary.append(dict(serial=int(serial), **experiment_group.attrs))

    summary.sort(key=itemgetter(*sorting))
    for result in summary:
        result["AB"] = result["A"] + result["B"]
        result["ok"] = {True: " ", False: "X"}[result["ok"]]
        print(columns.format(**result))

def compare(results=None, **kwargs):
    """Compare performance based on GEMM problem size (M * N * K) across different result files."""

    if not results or not isinstance(results, str):
        raise ValueError('Invalid type results')
    result_files = results.split(',')
    
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    symbols = cycle(['o', 's', 'D', '^', 'v', '<', '>'])
    
    data = []
    datatypes = set()
    
    # Extract data from each result file
    for file in result_files:
        with h5py.File(file, 'r') as h5:
            backend_data = []
            for serial in h5.keys():
                experiment_group = h5[serial]
                attrs = dict(serial=int(serial), **experiment_group.attrs)
                attrs["AB"] = attrs["A"] + attrs["B"]
                backend_data.append(attrs)
                datatypes.add(attrs["dtype"])
            data.append(backend_data)
    
    # Sort data for consistent plotting
    datatypes = sorted(datatypes)
    symbol_map = {dtype: next(symbols) for dtype in datatypes}
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    for backend_data, color, file in zip(data, colors, result_files):
        M_N_K = [d["M"] * d["N"] * d["K"] for d in backend_data]
        mean_milliseconds = [d["mean_microseconds"] / 1000 for d in backend_data]
        dtypes = [d["dtype"] for d in backend_data]
        algorithms = [str(d["A"] + d["B"]) for d in backend_data]
        
        for dtype in datatypes:
            dtype_data = [(mnk, mean) for mnk, mean, dt, algorithm in zip(M_N_K, mean_milliseconds, dtypes, algorithms) if dt == dtype]
            if dtype_data:
                x, y = zip(*dtype_data)
                plt.scatter(x, y, c=color, marker=symbol_map[dtype], label=f'{file} - {dtype}', alpha=0.6, edgecolors='w', linewidth=0.5)
        # if 'iree' in file:
        #     for algorithm in ["NN", "NT", "TN"]:
        #         dtype_data = [(mnk, mean) for mnk, mean, dt, algo in zip(M_N_K, mean_milliseconds, dtypes, algorithms) if algo == algorithm]
        #         if dtype_data:
        #             x, y = zip(*dtype_data)
        #             plt.scatter(x, y, c=next(colors), marker=symbol_map["fp16"], label=f'{file} - {algorithm}', alpha=0.6, edgecolors='w', linewidth=0.5)
        # else:
        #     dtype_data = [(mnk, mean) for mnk, mean, dt, algo in zip(M_N_K, mean_milliseconds, dtypes, algorithms)]
        #     if dtype_data:
        #         x, y = zip(*dtype_data)
        #         plt.scatter(x, y, c=next(colors), marker=symbol_map["bf16"], label=f'{file}', alpha=0.6, edgecolors='w', linewidth=0.5)

    
    plt.xlabel('GEMM Problem Size (M * N * K)')
    plt.ylabel('Mean Milliseconds')
    plt.title('Comparison of Mean Millisecond Performance by GEMM Problem Size')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('comparison.png')


def ls(top=None, suite=None, format=None, output=None, **kwargs):
    """List problems in suite."""

    if suite is None:
        suite = "all"
    problems = gemmbench.definitions.load_suite(top, suite)
    configurations = gemmbench.definitions.load_configurations(top)

    for problem in problems:
        print(problem.to_dict())

    for configuration in configurations:
        print(configuration.to_dict())


def specs(**kwargs):
    """Print machine specs."""

    print(gemmbench.specs.get_machine_specs())


def main(top: pathlib.Path):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    parser = argparse.ArgumentParser(description="GEMM performance tracking tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help=run.__doc__)
    run_parser.add_argument("--suite")
    run_parser.add_argument("--output")
    run_parser.add_argument("--repeat")
    run_parser.add_argument("--backends")
    run_parser.add_argument("--no-shuffle", action="store_true")

    ls_parser = subparsers.add_parser("ls", help=ls.__doc__)
    ls_parser.add_argument("--suite")
    ls_parser.add_argument("--format")
    ls_parser.add_argument("--output")

    specs_parser = subparsers.add_parser("specs", help=specs.__doc__)

    summary_parser = subparsers.add_parser("summary", help=summary.__doc__)
    summary_parser.add_argument("--results")

    summary_parser = subparsers.add_parser("compare", help=compare.__doc__)
    summary_parser.add_argument("--results")

    summary_parser = subparsers.add_parser("roofline", help=roofline.__doc__)
    summary_parser.add_argument("--results")

    args = parser.parse_args()

    globals()[args.command](top=top, **args.__dict__)
