import React, { useState, useEffect } from "react";
import { Scatter } from "react-chartjs-2";
import { backendColors } from "../utils";

const RuntimeChart = ({ data }) => {
  const chartData = {
    datasets: Object.entries(backendColors).map(([backend, color]) => ({
      label: backend,
      data: data
        .filter((item) => item.backendName === backend)
        .map((item) => ({
          x: item.M * item.N * item.K,
          y: item.meanMicroseconds,
          ...item,
        })),
      backgroundColor: color,
    })),
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
          text: "GEMM Problem Size (M * N * K)",
        },
      },
      y: {
        type: "logarithmic",
        title: {
          display: true,
          text: "Mean Microseconds",
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
              `Mean Microseconds: ${item.meanMicroseconds.toFixed(2)}`,
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

export { RuntimeChart };
