import { useState, useMemo, useEffect } from "react";
import "./KnnVisualizer.css";

const initialPoints = [
  { x: 20, y: 20, label: "A" },
  { x: 25, y: 30, label: "A" },
  { x: 30, y: 25, label: "A" },
  { x: 35, y: 20, label: "A" },
  { x: 40, y: 30, label: "A" },

  { x: 70, y: 70, label: "B" },
  { x: 75, y: 65, label: "B" },
  { x: 80, y: 75, label: "B" },
  { x: 65, y: 80, label: "B" },
  { x: 72, y: 78, label: "B" },
];

const labelColors = {
  A: "#e74c3c",
  B: "#3498db",
  C: "#2ecc71",
};

const metricPrettyNames = {
  euclidean: "Euclidean (L2)",
  manhattan: "Manhattan (L1)",
  minkowski: "Minkowski (Lp)",
};

function formatDistance(d) {
  if (d === undefined || d === null || Number.isNaN(d)) return "-";
  return d.toFixed(3);
}

function distance(a, b, metric, p = 3) {
  const dx = Math.abs(a.x - b.x);
  const dy = Math.abs(a.y - b.y);

  switch (metric) {
    case "manhattan":
      return dx + dy;
    case "minkowski": {
      const sum = Math.pow(dx, p) + Math.pow(dy, p);
      return Math.pow(sum, 1 / p);
    }
    case "euclidean":
    default:
      return Math.sqrt(dx * dx + dy * dy);
  }
}

function getNeighbors(queryPoint, points, k, metric, minkowskiP) {
  if (!queryPoint || points.length === 0) return [];

  const withDistances = points.map((p, index) => ({
    ...p,
    index,
    dist: distance(p, queryPoint, metric, minkowskiP),
  }));

  return withDistances.sort((a, b) => a.dist - b.dist).slice(0, k);
}

function majorityLabel(neighbors) {
  if (!neighbors || neighbors.length === 0) return null;
  const counts = {};
  neighbors.forEach((n) => {
    counts[n.label] = (counts[n.label] || 0) + 1;
  });

  let bestLabel = null;
  let bestCount = -1;
  Object.entries(counts).forEach(([label, count]) => {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  });

  return { label: bestLabel, counts };
}

export default function KnnVisualizer() {
  const [points, setPoints] = useState(initialPoints);
  const [k, setK] = useState(3);
  const [queryPoint, setQueryPoint] = useState(null);

  const [distanceMetric, setDistanceMetric] = useState("euclidean");
  const [minkowskiP, setMinkowskiP] = useState(3);

  const [selectedLabelForNewPoint, setSelectedLabelForNewPoint] = useState("A");
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(true);

  useEffect(() => {
    setK((prev) => {
      if (points.length === 0) return 1;
      return Math.min(prev, points.length);
    });
  }, [points]);

  const neighbors = useMemo(
    () => getNeighbors(queryPoint, points, k, distanceMetric, minkowskiP),
    [queryPoint, points, k, distanceMetric, minkowskiP]
  );

  const prediction = useMemo(() => {
    if (!queryPoint || neighbors.length === 0) return null;
    return majorityLabel(neighbors);
  }, [neighbors, queryPoint]);

  const width = 400;
  const height = 400;

  // NEW: single example pair (query point vs closest neighbor)
  const examplePair = useMemo(() => {
    if (!queryPoint || neighbors.length === 0) return null;
    const n = neighbors[0];
    const dx = Math.abs(queryPoint.x - n.x);
    const dy = Math.abs(queryPoint.y - n.y);
    return {
      neighbor: n,
      dx,
      dy,
    };
  }, [neighbors, queryPoint]);

  // NEW: compute all three metrics for the same (dx, dy)
  const metricComparison = useMemo(() => {
    if (!examplePair) return null;
    const { dx, dy } = examplePair;
    const origin = { x: 0, y: 0 };
    const deltas = { x: dx, y: dy };

    return {
      euclidean: distance(origin, deltas, "euclidean"),
      manhattan: distance(origin, deltas, "manhattan"),
      minkowski: distance(origin, deltas, "minkowski", minkowskiP),
    };
  }, [examplePair, minkowskiP]);

  function getSvgCoords(evt) {
    const svg = evt.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;

    const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
    return { x: svgP.x, y: svgP.y }; // in viewBox coordinates (0–100)
  }

  function handleClick(e) {
    const { x, y } = getSvgCoords(e);

    const yMath = 100 - y;

    if (e.altKey) {
      addTrainingPoint({ x, y: yMath });
    } else {
      setQueryPoint({ x, y: yMath });
    }
  }

  function handleContextMenu(e) {
    e.preventDefault();
    const { x, y } = getSvgCoords(e);
    const yMath = 100 - y;
    addTrainingPoint({ x, y: yMath });
  }

  function addTrainingPoint(point) {
    setPoints((prev) => [
      ...prev,
      { x: point.x, y: point.y, label: selectedLabelForNewPoint },
    ]);
  }

  function handleDeletePoint(indexToDelete) {
    setPoints((prev) => {
      if (prev.length <= 1) return prev;
      const copy = [...prev];
      copy.splice(indexToDelete, 1);
      return copy;
    });
  }

  const neighborIndexes = new Set(neighbors.map((n) => n.index));

  const decisionCells = useMemo(() => {
    if (!showDecisionBoundary || points.length === 0) return [];

    const gridSize = 30;
    const cellSize = 100 / gridSize;
    const cells = [];

    for (let gx = 0; gx < gridSize; gx++) {
      for (let gy = 0; gy < gridSize; gy++) {
        const xCenter = (gx + 0.5) * cellSize;
        const yCenter = (gy + 0.5) * cellSize;

        const cellPoint = { x: xCenter, y: yCenter };
        const cellNeighbors = getNeighbors(
          cellPoint,
          points,
          k,
          distanceMetric,
          minkowskiP
        );
        const maj = majorityLabel(cellNeighbors);
        if (!maj || !maj.label) continue;

        const color = labelColors[maj.label] || "#bdc3c7";

        cells.push({
          x: xCenter - cellSize / 2,
          y: yCenter - cellSize / 2,
          size: cellSize,
          color,
        });
      }
    }

    return cells;
  }, [points, k, distanceMetric, minkowskiP, showDecisionBoundary]);

  return (
    <div className="knn-root">
      <div className="knn-shell">
        {/* Header */}
        <div className="knn-header">
          <div>
            <div className="knn-title">k-NN Decision Boundary Visualizer</div>
            <div className="knn-subtitle">
              You can try out as much as you want!
            </div>
          </div>
          <div className="knn-tag">Byambajav.M</div>
        </div>

        <div className="knn-main">
          {/* Graph at top, full width */}
          <div className="knn-plot-section">
            <div className="knn-plot-subheader">
              Click to place a query point. Alt+click or right-click adds a
              training point with the selected label.
            </div>

            <svg
              width={width}
              height={height}
              viewBox="0 0 100 100"
              className="knn-svg"
              onClick={handleClick}
              onContextMenu={handleContextMenu}
            >
              {/* Background grid */}
              <defs>
                <pattern
                  id="smallGrid"
                  width="5"
                  height="5"
                  patternUnits="userSpaceOnUse"
                >
                  <path
                    d="M 5 0 L 0 0 0 5"
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="0.2"
                  />
                </pattern>
                <pattern
                  id="grid"
                  width="10"
                  height="10"
                  patternUnits="userSpaceOnUse"
                >
                  <rect width="10" height="10" fill="url(#smallGrid)" />
                  <path
                    d="M 10 0 L 0 0 0 10"
                    fill="none"
                    stroke="#d1d5db"
                    strokeWidth="0.4"
                  />
                </pattern>
              </defs>

              <rect
                x="0"
                y="0"
                width="100"
                height="100"
                fill="url(#grid)"
                opacity="0.9"
              />

              {/* Decision cells */}
              {decisionCells.map((cell, i) => (
                <rect
                  key={i}
                  x={cell.x}
                  y={100 - cell.y - cell.size}
                  width={cell.size}
                  height={cell.size}
                  fill={cell.color}
                  opacity={0.16}
                />
              ))}

              {/* Training points */}
              {points.map((p, i) => {
                const isNeighbor = neighborIndexes.has(i);
                const color = labelColors[p.label] || "#7f8c8d";

                return (
                  <g key={i}>
                    {isNeighbor && (
                      <circle
                        cx={p.x}
                        cy={100 - p.y}
                        r={4.8}
                        fill="none"
                        stroke="#22c55e"
                        strokeWidth="0.9"
                      />
                    )}
                    <circle cx={p.x} cy={100 - p.y} r={3.2} fill={color} />
                  </g>
                );
              })}

              {/* Query point */}
              {queryPoint && (
                <circle
                  cx={queryPoint.x}
                  cy={100 - queryPoint.y}
                  r={4}
                  fill="#facc15"
                  stroke="#111827"
                  strokeWidth="1"
                />
              )}
            </svg>

            <p className="knn-tip">
              Click = query point · Alt+click / right-click = add training point
              with selected label.
            </p>
          </div>

          {/* All other components/cards below, using grid */}
          <div className="knn-cards-grid">
            {/* Controls */}
            <div className="knn-card">
              <div className="knn-card-title">k-NN Controls</div>

              <label>
                k (number of neighbors): {k}
                <input
                  type="range"
                  className="knn-slider"
                  min="1"
                  max={Math.max(points.length, 1)}
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                />
              </label>

              <label>
                Distance metric:
                <select
                  className="knn-select"
                  value={distanceMetric}
                  onChange={(e) => setDistanceMetric(e.target.value)}
                >
                  <option value="euclidean">Euclidean (L2)</option>
                  <option value="manhattan">Manhattan (L1)</option>
                  <option value="minkowski">Minkowski (Lp)</option>
                </select>
              </label>

              {distanceMetric === "minkowski" && (
                <label>
                  Minkowski p: {minkowskiP}
                  <input
                    type="range"
                    className="knn-slider"
                    min="1"
                    max="6"
                    step="0.5"
                    value={minkowskiP}
                    onChange={(e) => setMinkowskiP(Number(e.target.value))}
                  />
                </label>
              )}

              <label>
                <input
                  type="checkbox"
                  className="knn-checkbox"
                  checked={showDecisionBoundary}
                  onChange={(e) => setShowDecisionBoundary(e.target.checked)}
                />
                Show decision boundary
              </label>

              <label>
                Label for new training points:
                <select
                  className="knn-select"
                  value={selectedLabelForNewPoint}
                  onChange={(e) => setSelectedLabelForNewPoint(e.target.value)}
                >
                  <option value="A">Class A</option>
                  <option value="B">Class B</option>
                  <option value="C">Class C</option>
                </select>
              </label>
            </div>

            {/* Prediction */}
            <div className="knn-card">
              <div className="knn-card-title">Prediction</div>

              {!queryPoint && (
                <p style={{ fontSize: 12, color: "#6b7280" }}>
                  Click on the graph to add a query point and see its predicted
                  class.
                </p>
              )}

              {queryPoint && prediction && (
                <>
                  <p style={{ fontSize: 13 }}>
                    Predicted class:{" "}
                    <strong
                      style={{
                        color: labelColors[prediction.label] || "#111827",
                        fontSize: 15,
                      }}
                    >
                      {prediction.label}
                    </strong>
                  </p>
                  <p style={{ marginTop: 8, fontSize: 12 }}>Neighbor votes:</p>
                  <ul style={{ fontSize: 12 }}>
                    {Object.entries(prediction.counts).map(([label, count]) => (
                      <li key={label}>
                        <span
                          className="knn-label-dot"
                          style={{
                            background: labelColors[label] || "#9ca3af",
                          }}
                        />
                        {label}: {count}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>

            {/* Explanation */}
            <div className="knn-card">
              <div className="knn-card-title">How k-NN works here</div>

              {!queryPoint && (
                <p style={{ fontSize: 12, color: "#6b7280" }}>
                  Place a query point to see the step-by-step process.
                </p>
              )}

              {queryPoint && neighbors.length === 0 && (
                <p style={{ fontSize: 12 }}>
                  No training points yet, so k-NN cannot make a prediction.
                </p>
              )}

              {queryPoint && neighbors.length > 0 && (
                <>
                  <ol
                    style={{
                      paddingLeft: 18,
                      fontSize: 12,
                      color: "#374151",
                      marginBottom: 8,
                    }}
                  >
                    <li>
                      <strong>Distance computation:</strong> take the query
                      point{" "}
                      <code>
                        ({queryPoint.x.toFixed(2)}, {queryPoint.y.toFixed(2)})
                      </code>{" "}
                      and compute its distance to every training point using{" "}
                      <strong>{metricPrettyNames[distanceMetric]}</strong>
                      {distanceMetric === "minkowski" && (
                        <>
                          {" "}
                          with p = <strong>{minkowskiP}</strong>
                        </>
                      )}
                      .
                    </li>
                    <li>
                      <strong>Sorting:</strong> sort all training points from{" "}
                      <em>closest</em> to <em>farthest</em>.
                    </li>
                    <li>
                      <strong>Picking neighbors:</strong> select the first{" "}
                      <strong>{Math.min(k, neighbors.length)}</strong> points —
                      those are the <strong>k nearest neighbors</strong>.
                    </li>
                    <li>
                      <strong>Majority vote:</strong> count how many neighbors
                      are in each class; the class with the most votes becomes
                      the <strong>prediction</strong>.
                    </li>
                  </ol>

                  {prediction && (
                    <p style={{ fontSize: 12 }}>
                      Here, the majority class is{" "}
                      <strong
                        style={{
                          color: labelColors[prediction.label] || "#111827",
                        }}
                      >
                        {prediction.label}
                      </strong>
                      .
                    </p>
                  )}

                  <div style={{ marginTop: 10 }}>
                    <strong style={{ fontSize: 12 }}>
                      Current k nearest neighbors
                    </strong>
                    <table className="knn-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Label</th>
                          <th>Position (x, y)</th>
                          <th>Distance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {neighbors.map((n, index) => (
                          <tr key={index}>
                            <td>{index + 1}</td>
                            <td>
                              <span
                                className="knn-label-dot"
                                style={{
                                  background: labelColors[n.label] || "#9ca3af",
                                }}
                              />
                              {n.label}
                            </td>
                            <td>
                              ({n.x.toFixed(2)}, {n.y.toFixed(2)})
                            </td>
                            <td>{formatDistance(n.dist)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>

            {/* NEW: Distance metric intuition */}
            <div className="knn-card">
              <div className="knn-card-title">Distance metric intuition</div>

              {!queryPoint && (
                <p style={{ fontSize: 12, color: "#6b7280" }}>
                  Add a query point to see the distance formula in action.
                </p>
              )}

              {queryPoint && examplePair && metricComparison && (
                <>
                  <p style={{ fontSize: 12, marginBottom: 6 }}>
                    We look at the <strong>closest neighbor</strong>{" "}
                    <code>
                      ({examplePair.neighbor.x.toFixed(2)},{" "}
                      {examplePair.neighbor.y.toFixed(2)})
                    </code>{" "}
                    and compare it to the query point{" "}
                    <code>
                      ({queryPoint.x.toFixed(2)}, {queryPoint.y.toFixed(2)})
                    </code>
                    .
                  </p>

                  <p
                    style={{
                      fontSize: 12,
                      marginBottom: 4,
                    }}
                  >
                    Differences:
                    <br />
                    <code>
                      Δx = |x<sub>query</sub> - x<sub>neighbor</sub>| ={" "}
                      {examplePair.dx.toFixed(3)}
                    </code>
                    <br />
                    <code>
                      Δy = |y<sub>query</sub> - y<sub>neighbor</sub>| ={" "}
                      {examplePair.dy.toFixed(3)}
                    </code>
                  </p>

                  {/* Formula that changes with selected metric */}
                  <div className="knn-formula-block">
                    <div
                      style={{
                        fontSize: 12,
                        marginBottom: 4,
                        fontWeight: 600,
                      }}
                    >
                      {metricPrettyNames[distanceMetric]} formula
                    </div>

                    {distanceMetric === "euclidean" && (
                      <div className="knn-formula">
                        d =
                        <span className="knn-formula-dynamic">
                          √(Δx² + Δy²)
                        </span>
                        =
                        <span className="knn-formula-example">
                          √(
                          {examplePair.dx.toFixed(3)}² +{" "}
                          {examplePair.dy.toFixed(3)}²) ≈{" "}
                          {formatDistance(metricComparison.euclidean)}
                        </span>
                      </div>
                    )}

                    {distanceMetric === "manhattan" && (
                      <div className="knn-formula">
                        d =
                        <span className="knn-formula-dynamic">|Δx| + |Δy|</span>
                        =
                        <span className="knn-formula-example">
                          {" "}
                          {examplePair.dx.toFixed(3)} +{" "}
                          {examplePair.dy.toFixed(3)} ≈{" "}
                          {formatDistance(metricComparison.manhattan)}
                        </span>
                      </div>
                    )}

                    {distanceMetric === "minkowski" && (
                      <div className="knn-formula">
                        d =
                        <span className="knn-formula-dynamic">
                          (|Δx|<sup>p</sup> + |Δy|<sup>p</sup>)<sup>1/p</sup>
                        </span>
                        =
                        <span className="knn-formula-example">
                          (|
                          {examplePair.dx.toFixed(3)}|<sup>{minkowskiP}</sup> +
                          |{examplePair.dy.toFixed(3)}|<sup>{minkowskiP}</sup>)
                          <sup>1/{minkowskiP}</sup> ≈{" "}
                          {formatDistance(metricComparison.minkowski)}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Compare all metrics on the same pair */}
                  <div style={{ marginTop: 10 }}>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        marginBottom: 4,
                      }}
                    >
                      Same pair, different metrics
                    </div>
                    <table className="knn-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Symbol</th>
                          <th>Distance</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          className={
                            distanceMetric === "euclidean"
                              ? "knn-row-active-metric"
                              : ""
                          }
                        >
                          <td>Euclidean (L2)</td>
                          <td>√(Δx² + Δy²)</td>
                          <td>{formatDistance(metricComparison.euclidean)}</td>
                        </tr>
                        <tr
                          className={
                            distanceMetric === "manhattan"
                              ? "knn-row-active-metric"
                              : ""
                          }
                        >
                          <td>Manhattan (L1)</td>
                          <td>|Δx| + |Δy|</td>
                          <td>{formatDistance(metricComparison.manhattan)}</td>
                        </tr>
                        <tr
                          className={
                            distanceMetric === "minkowski"
                              ? "knn-row-active-metric"
                              : ""
                          }
                        >
                          <td>
                            Minkowski (Lp)
                            <br />
                            <span style={{ fontSize: 11 }}>
                              (current p = {minkowskiP})
                            </span>
                          </td>
                          <td>
                            (|Δx|<sup>p</sup> + |Δy|<sup>p</sup>)<sup>1/p</sup>
                          </td>
                          <td>{formatDistance(metricComparison.minkowski)}</td>
                        </tr>
                      </tbody>
                    </table>
                    <p style={{ fontSize: 11, marginTop: 4, color: "#6b7280" }}>
                      Notice how the <strong>same pair of points</strong> gets
                      different distances depending on the metric. k-NN&apos;s
                      behavior changes because &quot;closest&quot; is defined by
                      this choice.
                    </p>
                  </div>
                </>
              )}
            </div>

            {/* Legend */}
            <div className="knn-card">
              <div className="knn-card-title">Legend & interactions</div>
              <ul className="knn-legend-list">
                <li>Colored dots = training points (A, B, C).</li>
                <li>Yellow dot = query point.</li>
                <li>Green rings = current k nearest neighbors.</li>
                <li>Background tint = predicted class region.</li>
                <li>
                  Alt+click or right-click on the graph to add a training point
                  with the selected label.
                </li>
              </ul>
            </div>

            {/* Training points */}
            <div className="knn-card">
              <div className="knn-card-title">
                Training points ({points.length})
              </div>
              <ul className="knn-list">
                {points.map((p, i) => (
                  <li key={i}>
                    <span
                      className="knn-label-dot"
                      style={{
                        background: labelColors[p.label] || "#9ca3af",
                      }}
                    />
                    {p.label} @ ({p.x.toFixed(1)}, {p.y.toFixed(1)})
                    <button
                      className="knn-delete-btn"
                      onClick={() => handleDeletePoint(i)}
                    >
                      delete
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
