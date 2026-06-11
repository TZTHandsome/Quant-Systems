const state = {
  config: null,
  lastResponse: null,
};

const factorsInput = document.getElementById("factors-input");
const paperStartDate = document.getElementById("paper-start-date");
const paperEndDate = document.getElementById("paper-end-date");
const paperBars = document.getElementById("paper-bars");
const initialCapital = document.getElementById("initial-capital");
const positionLimit = document.getElementById("position-limit");
const grossLeverage = document.getElementById("gross-leverage");
const smoothing = document.getElementById("smoothing");
const marketNeutral = document.getElementById("market-neutral");
const marketAdaptation = document.getElementById("market-adaptation");
const bullFollowStrength = document.getElementById("bull-follow-strength");
const factorWeightScheme = document.getElementById("factor-weight-scheme");
const bullTrendPreference = document.getElementById("bull-trend-preference");
const autoWeightLookbackBars = document.getElementById("auto-weight-lookback-bars");
const runButton = document.getElementById("run-button");
const loadDefaultsButton = document.getElementById("load-defaults-button");
const runStatus = document.getElementById("run-status");
const datasetSummary = document.getElementById("dataset-summary");
const progressLabel = document.getElementById("progress-label");
const progressValue = document.getElementById("progress-value");
const progressFill = document.getElementById("progress-fill");

init();

async function init() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    state.config = config;
    datasetSummary.textContent = `${config.symbol_count} 个币 · ${config.date_min} 到 ${config.date_max}`;
    paperStartDate.min = config.date_min;
    paperStartDate.max = config.date_max;
    paperEndDate.min = config.date_min;
    paperEndDate.max = config.date_max;
    paperStartDate.value = config.default_paper_start_date || config.date_min;
    paperEndDate.value = config.default_paper_end_date || config.date_max;
    paperBars.value = config.default_paper_bars;
    initialCapital.value = config.default_initial_capital;
    positionLimit.value = config.defaults.position_limit;
    grossLeverage.value = config.defaults.gross_leverage;
    smoothing.value = config.defaults.smoothing;
    marketNeutral.checked = config.defaults.market_neutral;
    marketAdaptation.value = config.defaults.market_adaptation || "standard";
    bullFollowStrength.value = config.defaults.bull_follow_strength ?? 0.65;
    factorWeightScheme.value = config.defaults.factor_weight_scheme || "equal";
    bullTrendPreference.value = config.defaults.bull_trend_preference || "standard";
    autoWeightLookbackBars.value = config.defaults.auto_weight_lookback_bars || 252;
    setDefaultFactors(config.default_factors || []);
  } catch (error) {
    datasetSummary.textContent = "读取配置失败";
    setStatus("Load Failed", true);
  }
}

runButton.addEventListener("click", runPaperTrading);
loadDefaultsButton.addEventListener("click", () => setDefaultFactors(state.config?.default_factors || []));

function setDefaultFactors(expressions) {
  factorsInput.value = expressions.join("\n");
}

function normalizeDateInput(rawValue, minValue, maxValue) {
  const value = String(rawValue || "").trim();
  if (!value) {
    return null;
  }
  if (minValue && value < minValue) {
    return minValue;
  }
  if (maxValue && value > maxValue) {
    return maxValue;
  }
  return value;
}

async function runPaperTrading() {
  setStatus("Running...", false);
  runButton.disabled = true;
  updateProgress(0, "提交任务");
  const normalizedPaperStartDate = normalizeDateInput(
    paperStartDate.value,
    state.config?.date_min,
    state.config?.date_max,
  );
  const normalizedPaperEndDate = normalizeDateInput(
    paperEndDate.value,
    state.config?.date_min,
    state.config?.date_max,
  );
  if (normalizedPaperStartDate && normalizedPaperEndDate && normalizedPaperStartDate > normalizedPaperEndDate) {
    updateProgress(100, "运行失败");
    setStatus("开始日期不能晚于结束日期", true);
    runButton.disabled = false;
    return;
  }
  paperStartDate.value = normalizedPaperStartDate || "";
  paperEndDate.value = normalizedPaperEndDate || "";
  const payload = {
    factor_expressions: factorsInput.value.trim(),
    paper_start_date: normalizedPaperStartDate || null,
    paper_end_date: normalizedPaperEndDate || null,
    paper_trading_bars: Number(paperBars.value || 90),
    initial_capital: Number(initialCapital.value || 100000),
    position_limit: Number(positionLimit.value || 0.15),
    gross_leverage: Number(grossLeverage.value || 1.2),
    smoothing: Number(smoothing.value || 0.35),
    market_neutral: marketNeutral.checked,
    market_adaptation: marketAdaptation.value || "standard",
    bull_follow_strength: Number(bullFollowStrength.value || 0.65),
    factor_weight_scheme: factorWeightScheme.value || "equal",
    bull_trend_preference: bullTrendPreference.value || "standard",
    auto_weight_lookback_bars: Number(autoWeightLookbackBars.value || 252),
  };

  try {
    const response = await fetch("/api/run-paper", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Paper trading failed");
    }
    const finalJob = await pollJob(result.job_id);
    if (finalJob.status !== "completed" || !finalJob.result) {
      throw new Error(finalJob.error?.message || finalJob.message || "Paper trading failed");
    }
    state.lastResponse = finalJob.result;
    renderResponse(finalJob.result);
    updateProgress(100, "运行完成");
    setStatus("Completed", false);
  } catch (error) {
    updateProgress(100, "运行失败");
    setStatus(error.message || "Run failed", true);
  } finally {
    runButton.disabled = false;
  }
}

async function pollJob(jobId) {
  for (;;) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.error || "Job polling failed");
    }
    const pct = Math.max(0, Math.min(100, Math.round((Number(job.progress || 0)) * 100)));
    updateProgress(pct, job.message || job.status || "运行中");
    if (job.status === "completed" || job.status === "failed") {
      return job;
    }
    await sleep(500);
  }
}

function renderResponse(result) {
  const summary = result.paper_summary;
  const benchmarks = result.benchmarks;
  setMetric("metric-final-equity", formatCurrency(summary.final_equity));
  setMetric("metric-total-return", formatSignedPct(summary.total_return));
  setMetric("metric-trades", String(summary.num_trades));
  setMetric("metric-turnover", `Mean Turnover ${summary.mean_turnover.toFixed(3)}`);
  setMetric("metric-btc-return", formatSignedPct(benchmarks.btc.total_return));
  setMetric("metric-eq-return", `EQ ${formatSignedPct(benchmarks.equal_weight.total_return)}`);
  setMetric("metric-cap-return", `CAP ${formatSignedPct(benchmarks.market_cap.total_return)}`);

  renderSelectedFactors(result.selected_factors);
  renderTable("daily-table", result.daily_metrics, (row) => [
    row.date,
    formatCurrency(row.equity_before),
    formatCurrency(row.equity_after),
    formatSignedCurrency(row.net_pnl),
    formatSignedPct(row.return),
    row.turnover?.toFixed(3) ?? "--",
  ]);
  renderTable("trades-table", result.trades, (row) => [
    row.execution_date || row.date,
    row.symbol,
    row.side,
    formatPct(row.weight_before),
    formatPct(row.target_weight),
    formatPct(row.weight_after),
    formatSignedPct(row.weight_delta),
  ]);
  renderTable("positions-table", result.positions, (row) => [
    row.execution_date,
    row.symbol,
    formatPct(row.weight),
    formatNumber(row.units_est),
    formatNumber(row.entry_open),
    formatNumber(row.exit_close),
    formatSignedCurrency(row.gross_pnl),
  ]);
  renderChart(result.equity_curve, result.benchmark_curves);
}

function renderSelectedFactors(expressions) {
  const root = document.getElementById("selected-factors");
  root.innerHTML = "";
  expressions.forEach((factor) => {
    const card = document.createElement("div");
    card.className = "selected-tag factor-card";
    const direction = factor.direction_label || (Number(factor.direction) < 0 ? "short" : "long");
    const weight = Number(factor.final_weight || 0);
    const fitness = Number(factor.fitness || 0);
    const metricLine = [
      `${direction}`,
      `w ${weight.toFixed(3)}`,
      `fit ${fitness.toFixed(3)}`,
    ].join(" · ");
    const extraLine = [
      `IC ${Number(factor.validation_rank_ic_mean || 0).toFixed(3)}`,
      `Sharpe ${Number(factor.sharpe || 0).toFixed(2)}`,
      `${factor.direction_source || "manual"} / ${factor.weight_source || "equal"}`,
    ].join(" · ");
    card.innerHTML = `
      <div class="factor-meta">${metricLine}</div>
      <div class="factor-expression">${escapeHtml(factor.expression || "")}</div>
      <div class="factor-sub">${extraLine}</div>
    `;
    root.appendChild(card);
  });
}

function renderTable(tableId, rows, mapper) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";
  rows.slice(0, 500).forEach((row) => {
    const tr = document.createElement("tr");
    mapper(row).forEach((value) => {
      const td = document.createElement("td");
      td.innerHTML = decorateValue(String(value));
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderChart(strategyCurve, benchmarkCurves) {
  const svg = document.getElementById("equity-chart");
  const width = 960;
  const height = 320;
  const padding = { top: 24, right: 18, bottom: 28, left: 18 };
  const allSeries = [
    { key: "strategy", values: strategyCurve.map((row) => ({ x: row.date, y: row.equity })) },
    { key: "btc", values: benchmarkCurves.btc || [] },
    { key: "equal", values: benchmarkCurves.equal_weight || [] },
    { key: "cap", values: benchmarkCurves.market_cap || [] },
  ].filter((series) => series.values.length > 0);

  if (allSeries.length === 0) {
    svg.innerHTML = "";
    return;
  }

  const yValues = allSeries.flatMap((series) => series.values.map((point) => Number(point.equity ?? point.y)));
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const range = Math.max(yMax - yMin, 1);
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const colorMap = {
    strategy: "#0e8c73",
    btc: "#f7931a",
    equal: "#2e62c6",
    cap: "#7a55c6",
  };

  svg.innerHTML = "";
  drawGrid(svg, width, height, padding, innerWidth, innerHeight);
  allSeries.forEach((series) => {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", colorMap[series.key]);
    path.setAttribute("stroke-width", series.key === "strategy" ? "3" : "2");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute(
      "d",
      series.values.map((point, index) => {
        const x = padding.left + (innerWidth * index / Math.max(series.values.length - 1, 1));
        const yVal = Number(point.equity ?? point.y);
        const y = padding.top + innerHeight - ((yVal - yMin) / range) * innerHeight;
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      }).join(" ")
    );
    svg.appendChild(path);
  });
}

function drawGrid(svg, width, height, padding, innerWidth, innerHeight) {
  const gridColor = "rgba(19, 32, 51, 0.08)";
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (innerHeight * i / 4);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padding.left);
    line.setAttribute("x2", width - padding.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", gridColor);
    line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
  }
}

function setMetric(id, value) {
  document.getElementById(id).textContent = value;
}

function setStatus(message, isError) {
  runStatus.textContent = message;
  runStatus.style.background = isError ? "rgba(191,75,75,0.12)" : "rgba(14,140,115,0.12)";
  runStatus.style.color = isError ? "#bf4b4b" : "#0b6f5b";
}

function updateProgress(percent, label) {
  progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  progressValue.textContent = `${Math.max(0, Math.min(100, percent))}%`;
  progressLabel.textContent = label;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function decorateValue(value) {
  if (value.startsWith("-")) {
    return `<span class="negative">${value}</span>`;
  }
  if (value.startsWith("+")) {
    return `<span class="positive">${value}</span>`;
  }
  return value;
}

function formatCurrency(value) {
  return Number(value).toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
}

function formatSignedCurrency(value) {
  const number = Number(value || 0);
  const text = formatCurrency(Math.abs(number));
  return `${number >= 0 ? "+" : "-"}${text}`;
}

function formatPct(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatSignedPct(value) {
  const number = Number(value || 0) * 100;
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString("en-US", { maximumFractionDigits: 4 });
}
