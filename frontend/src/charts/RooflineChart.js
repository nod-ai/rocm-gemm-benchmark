import React, { useState, useEffect } from "react";
import { Scatter } from "react-chartjs-2";
import { backendColors } from "../utils";

const RooflineChart = ({ data }) => {
  const peak_memory_bandwidth = 5.3;
  const peak_compute = 1300;

  // Define memory-bound and compute-bound lines
  const memoryBoundLine = {
    label: "Memory Bound",
    data: Array.from({ length: 50 }, (_, i) => ({
      x: Math.pow(10, i * 0.1),
      y: Math.pow(10, i * 0.1) * peak_memory_bandwidth, // Example slope for memory-bound
    })),
    borderColor: "rgba(255, 159, 64, 1)",
    borderWidth: 2,
    fill: false,
    showLine: true,
    pointRadius: 0,
  };

  const computeBoundLine = {
    label: "Compute Bound",
    data: Array.from({ length: 50 }, (_, i) => ({
      x: Math.pow(10, i * 0.1),
      y: peak_compute,
    })),
    borderColor: "rgba(153, 102, 255, 1)",
    borderWidth: 2,
    fill: false,
    showLine: true,
    pointRadius: 0,
  };

  const chartData = {
    datasets: [
      ...Object.entries(backendColors).map(([backend, color]) => ({
        label: backend,
        data: data
          .filter((item) => item.backendName === backend)
          .map((item) => ({
            x: item.arithmeticIntensity,
            y: item.tflops,
            ...item,
          })),
        backgroundColor: color,
      })),
      memoryBoundLine,
      computeBoundLine,
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: "logarithmic",
        position: "bottom",
        title: {
          display: true,
          text: "Arithmetic Intensity (FLOPs / byte)",
        },
      },
      y: {
        type: "logarithmic",
        title: {
          display: true,
          text: "Performance (TFLOPs / s)",
        },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: (context) => {
            const item = context.raw;
            return [
              `Backend: ${item.backendName}`,
              `Transpose A: ${item.transposeA}`,
              `Transpose B: ${item.transposeB}`,
              `M: ${item.M}, N: ${item.N}, K: ${item.K}`,
              `dtype: ${item.dtype}`,
              `Arithmetic Intensity: ${item.arithmeticIntensity.toFixed(2)}`,
              `TFLOPS: ${item.tflops.toFixed(2)}`,
            ];
          },
        },
      },
    },
  };

  return <Scatter data={chartData} options={options} />;
};

export { RooflineChart };
