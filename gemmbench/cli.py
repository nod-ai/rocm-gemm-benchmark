import os
import argparse
import csv
import datetime
import h5py
import logging
import pathlib
import sys

import gemmbench
from gemmbench.bench_utils import *
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
    repeat = int(repeat)

    if no_shuffle is None:
        no_shuffle = False

    if output is None:
        output = (
            datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
            + "-"
            + gemmbench.git.short_hash()
            + ".csv"
        )
    output = pathlib.Path(output)

    print(f"Writing results to: {output}")

    with open(output, mode='w', newline='') as csvfile:
        if 'sharkfa' in backends:
            fieldnames = [
                "index", 
                "tag",
                "name",
                "BATCH",
                "NH",
                "SEQ_Q",
                "SEQ_KV",
                "D_HEAD",
                "dtype",
                "mean_microseconds",
                "arithmetic_intensity",
                "tflops",
                "ok",
            ]
        else:
            fieldnames = [
                "index", 
                "tag",
                "name",
                "M",
                "N",
                "K",
                "dtype",
                "A",
                "B",
                "mean_microseconds",
                "arithmetic_intensity",
                "tflops",
                "ok",
            ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (problem, solution, configuration, result) in enumerate(
            gemmbench.server.run_problems(
                problems,
                solutions,
                configurations,
                shuffle=not no_shuffle,
                repeat=repeat,
            )
        ):
            row = {"index": i}
            row.update(problem)
            row.update(solution)
            row["ok"] = result["ok"]
            row["mean_microseconds"] = result["mean_microseconds"]

            if 'sharkfa' in backends:
                row["BATCH"] = ord(str(row['A'])[0])
                row["NH"] = ord(str(row['B'])[0])
                row["SEQ_Q"] = row['M']
                row["SEQ_KV"] = row['N']
                row["D_HEAD"] = row['K']
                flops, bytes = get_problem_compute('attention', row["BATCH"], row["NH"], row["SEQ_Q"], row["SEQ_KV"], row["D_HEAD"], row["dtype"])
            else:
                flops, bytes = get_problem_compute('gemm', row['M'], row['N'], row['K'], row['dtype'])
            
            if row['ok']:
                row['arithmetic_intensity'] = flops / bytes
                row['tflops'] = (flops / 1e12) / (row['mean_microseconds'] / 1e6)
            else:
                row['arithmetic_intensity'] = 0
                row['tflops'] = 0

            writer.writerow({k: row[k] for k in row.keys() if k in fieldnames})

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
                row = {k: float(v) if k not in ['ok', 'tag', 'name', 'A', 'B', 'dtype'] else v for k, v in row.items()}
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
        backend_data = []
        with open(file.strip(), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = {k: float(v) if k not in ['ok', 'tag', 'name', 'A', 'B', 'dtype'] else v for k, v in row.items()}
                row['ok'] = row['ok'] == 'True'
                row['AB'] = row['A'] + row['B']
                backend_data.append(row)
                datatypes.add(row["dtype"])
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
                plt.scatter(x, y, c=color, marker=symbol_map[dtype], label=f'{file.strip()} - {dtype}', alpha=0.6, edgecolors='w', linewidth=0.5)
    
    plt.xlabel('GEMM Problem Size (M * N * K)')
    plt.ylabel('Mean Milliseconds')
    plt.title('Comparison of Mean Millisecond Performance by GEMM Problem Size')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('comparison.png')
    plt.close()
    
    print("Comparison plot saved as 'comparison.png'")


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
    
    summary_parser = subparsers.add_parser("compare", help=compare.__doc__)
    summary_parser.add_argument("--results")

    summary_parser = subparsers.add_parser("roofline", help=roofline.__doc__)
    summary_parser.add_argument("--results")

    args = parser.parse_args()

    globals()[args.command](top=top, **args.__dict__)
