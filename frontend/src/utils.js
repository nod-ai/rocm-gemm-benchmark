import axios from "axios";

const backendColors = {
  iree: "rgba(255, 99, 132, 0.8)",
  rocblas: "rgba(54, 162, 235, 0.8)",
  hipblaslt: "rgba(75, 192, 192, 0.8)",
  shark: "rgba(255, 99, 132, 0.8)",
  triton: "rgba(54, 162, 235, 0.8)",
  torch: "rgba(75, 192, 192, 0.8)",
};

const algToURL = {
  gemm: (backend) =>
    `https://sharkpublic.blob.core.windows.net/sharkpublic/surya/gemm-benchmarks/${backend}.csv`,
  attention: (backend) =>
    `https://sharkpublic.blob.core.windows.net/sharkpublic/surya/gemm-benchmarks/${backend}_llama_sdxl_attention.csv`,
};

const csvHandlers = {
  gemm: (backend, lines) =>
    lines.map((line) => {
      const [
        index,
        tag,
        name,
        M,
        N,
        K,
        dtype,
        A,
        B,
        meanMicroseconds,
        arithmeticIntensity,
        tflops,
        ok,
      ] = line.split(",");
      return {
        model: tag,
        backendName: backend,
        transposeA: A === "T",
        transposeB: B === "T",
        M: parseInt(M),
        N: parseInt(N),
        K: parseInt(K),
        dtype,
        meanMicroseconds: parseFloat(meanMicroseconds),
        arithmeticIntensity: parseFloat(arithmeticIntensity),
        tflops: parseFloat(tflops),
        ok,
      };
    }),
  attention: (backend, lines) =>
    lines.map((line) => {
      const [
        index,
        tag,
        name,
        batch,
        numHeads,
        seqLenQ,
        seqLenKV,
        dimHead,
        dtype,
        meanMicroseconds,
        arithmeticIntensity,
        tflops,
        ok,
      ] = line.split(",");
      return {
        model: tag,
        backendName: backend,
        batch,
        numHeads,
        seqLenQ,
        seqLenKV,
        dimHead,
        dtype,
        meanMicroseconds: parseFloat(meanMicroseconds),
        arithmeticIntensity: parseFloat(arithmeticIntensity),
        tflops: parseFloat(tflops),
        ok,
      };
    }),
};

const fetchAndParseCSV = async (backend, algorithm) => {
  const url = algToURL[algorithm](backend);
  const response = await axios.get(url);
  const text = response.data;
  const lines = text.split("\n").slice(1);
  return csvHandlers[algorithm](backend, lines);
};

export { backendColors, fetchAndParseCSV };
