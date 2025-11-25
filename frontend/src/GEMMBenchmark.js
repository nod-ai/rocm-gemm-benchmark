import React, { useState, useEffect } from "react";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  LogarithmicScale,
} from "chart.js";
import { fetchAndParseCSV } from "./utils";
import "./chart.css";
import { RuntimeChart } from "./charts/RuntimeChart";
import { RooflineChart } from "./charts/RooflineChart";

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  LogarithmicScale,
  Tooltip,
  Legend
);

const algBackends = {
  gemm: ["iree", "rocblas", "hipblaslt"],
  attention: ["amdshark", "torch", "triton"],
};

const algPlots = {
  gemm: {
    comparison: true,
    roofline: true,
  },
  attention: {
    comparison: false,
    roofline: true,
  },
};

const GEMMBenchmark = () => {
  const [algorithm, setAlgorithm] = useState("gemm");
  const [roofline, setRoofline] = useState(true);
  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const [allModels, setModels] = useState([]);
  const [allDtypes, setDtypes] = useState([]);

  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedDtypes, setSelectedDtypes] = useState([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);

  const [showFilters, setShowFilters] = useState(false); // State for showing/hiding filters

  ChartJS.register(LogarithmicScale);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const backends = algBackends[algorithm];
        const allData = await Promise.all(
          backends.map((backend) => fetchAndParseCSV(backend, algorithm))
        );
        const flatData = allData.flat();
        setData(flatData);
        setFilteredData(flatData);
        const models = [...new Set(flatData.map((item) => item.model))].filter(
          (str) => str && str.trim() !== ""
        );
        const dtypes = [...new Set(flatData.map((item) => item.dtype))].filter(
          (str) => str && str.trim() !== ""
        );
        console.log(flatData);
        setModels(models);
        setDtypes(dtypes);
        setSelectedModels(models);
        setSelectedDtypes(dtypes);
        setSelectedAlgorithms(["AB", "AB'", "A'B", "A'B'"]);
        setIsLoading(false);
      } catch (err) {
        console.error(err);
        setError("Failed to fetch data. Please try again later.");
        setIsLoading(false);
      }
    };

    fetchData();
  }, [algorithm]);

  useEffect(() => {
    const filtered = data.filter(
      (item) =>
        item.ok &&
        selectedModels.includes(item.model) &&
        selectedDtypes.includes(item.dtype) &&
        selectedAlgorithms.includes(
          `${item.transposeA ? "A'" : "A"}${item.transposeB ? "B'" : "B"}`
        )
    );
    setFilteredData(filtered);
  }, [selectedModels, selectedDtypes, selectedAlgorithms, data]);

  const handleCheckboxChange = (setter, value) => {
    setter((prev) =>
      prev.includes(value)
        ? prev.filter((item) => item !== value)
        : [...prev, value]
    );
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <div style={{ width: "calc(100% - 40px)", padding: "20px" }}>
      <h2 style={{ textAlign: "center", marginBottom: "20px" }}>
        <select
          value={`${algorithm}:${roofline ? "roofline" : "comparison"}`}
          onChange={(e) => {
            const newAlgorithm = e.target.value.split(":")[0];
            const newRoofline = e.target.value.split(":")[1] === "roofline";
            setRoofline(newRoofline);
            setAlgorithm(newAlgorithm);
          }}
        >
          {Object.entries(algPlots).map(([algorithm, support]) => (
            <>
              {support.comparison && (
                <option
                  value={`${algorithm}:comparison`}
                >{`${algorithm.toUpperCase()} Benchmark - Comparison`}</option>
              )}
              {support.roofline && (
                <option
                  value={`${algorithm}:roofline`}
                >{`${algorithm.toUpperCase()} Benchmark - Roofline`}</option>
              )}
            </>
          ))}
        </select>
      </h2>
      <button
        onClick={() => setShowFilters((prev) => !prev)}
        className="toggle-filters"
      >
        {showFilters ? "Hide Filters" : "Show Filters"}
      </button>
      {showFilters && (
        <div className="filters-container">
          <div className="filter-row">
            <h4>Model Filter</h4>
            <div className="checkbox-row">
              {allModels.map((model) => (
                <label key={model} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model)}
                    onChange={() =>
                      handleCheckboxChange(setSelectedModels, model)
                    }
                  />
                  {model}
                </label>
              ))}
            </div>
          </div>
          <div className="filter-row">
            <h4>Datatype Filter</h4>
            <div className="checkbox-row">
              {allDtypes.map((dtype) => (
                <label key={dtype} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={selectedDtypes.includes(dtype)}
                    onChange={() =>
                      handleCheckboxChange(setSelectedDtypes, dtype)
                    }
                  />
                  {dtype}
                </label>
              ))}
            </div>
          </div>
          {algorithm === "gemm" && (
            <div className="filter-row">
              <h4>Algorithm Filter</h4>
              <div className="checkbox-row">
                {["AB", "AB'", "A'B", "A'B'"].map((alg) => (
                  <label key={alg} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={selectedAlgorithms.includes(alg)}
                      onChange={() =>
                        handleCheckboxChange(setSelectedAlgorithms, alg)
                      }
                    />
                    {alg}
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      <div style={{ width: "100%", height: "calc(100vh - 200px)" }}>
        {roofline ? (
          <RooflineChart data={filteredData} />
        ) : (
          <RuntimeChart data={filteredData} />
        )}
      </div>
    </div>
  );
};

export default GEMMBenchmark;
