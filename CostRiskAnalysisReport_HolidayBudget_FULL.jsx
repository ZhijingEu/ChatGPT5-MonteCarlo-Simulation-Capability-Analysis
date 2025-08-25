import React, { useEffect, useMemo, useState } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, BarChart, Bar, CartesianGrid, Legend } from "recharts";

// ------------------------------------------------------------
// Holiday Budget Monte Carlo – Interactive Canvas (client-side)
// - Reproducible seeded RNG
// - PERT + Uniform marginals
// - Iman–Conover rank correlation for (Accom, Meals, Tours)
// - Risk events (Bernoulli triggers)
// - FX uncertainty
// - Outputs + stats (min, P0..P100, mean, std, skew, kurtosis), PDF & CDF
// - Tornado charts: (1) Spearman rank; (2) Stepwise OLS standardized coeffs
// ------------------------------------------------------------

// ---------- Seeded RNG (mulberry32) + Box–Muller for normals ----------
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296; // [0,1)
  };
}

function makeRNG(seed: number) {
  const u = mulberry32(seed);
  let spare: number | null = null;
  return {
    uniform: () => u(),
    normal: () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      const u1 = Math.max(u(), 1e-12);
      const u2 = u();
      const r = Math.sqrt(-2.0 * Math.log(u1));
      const th = 2.0 * Math.PI * u2;
      const z0 = r * Math.cos(th);
      const z1 = r * Math.sin(th);
      spare = z1; return z0;
    }
  };
}

// ---------- Gamma & Beta sampling ----------
function randGamma(k: number, rng: ReturnType<typeof makeRNG>) {
  if (k <= 0) return 0;
  if (k < 1) {
    const c = randGamma(k + 1, rng);
    const u = Math.max(rng.uniform(), 1e-12);
    return c * Math.pow(u, 1 / k);
  }
  const d = k - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    const x = rng.normal();
    let v = 1 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    const u = rng.uniform();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function randBeta(a: number, b: number, rng: ReturnType<typeof makeRNG>) {
  const x = randGamma(a, rng);
  const y = randGamma(b, rng);
  return x / (x + y);
}

function samplePERT(a: number, m: number, b: number, rng: ReturnType<typeof makeRNG>, lambda = 4) {
  if (a === b) return a;
  const alpha = 1 + lambda * (m - a) / (b - a);
  const beta = 1 + lambda * (b - m) / (b - a);
  const u = randBeta(alpha, beta, rng);
  return a + u * (b - a);
}

// ---------- Utilities: stats, percentiles, histogram, CDF, ranks ----------
function mean(arr: number[]) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function variance(arr: number[], m?: number) {
  const mu = m ?? mean(arr);
  let s = 0; for (const v of arr) s += (v - mu) ** 2;
  return s / Math.max(1, arr.length - 1);
}
function stddev(arr: number[]) { return Math.sqrt(variance(arr)); }
function skewness(arr: number[]) {
  const n = arr.length; const mu = mean(arr); const sd = Math.sqrt(variance(arr, mu));
  if (sd === 0) return 0;
  let m3 = 0; for (const v of arr) m3 += (v - mu) ** 3; m3 /= n;
  return (Math.sqrt(n * (n - 1)) / Math.max(1, n - 2)) * (m3 / (sd ** 3));
}
function kurtosisExcess(arr: number[]) {
  const n = arr.length; const mu = mean(arr); const sd = Math.sqrt(variance(arr, mu));
  if (sd === 0) return -3;
  let m4 = 0; for (const v of arr) m4 += (v - mu) ** 4; m4 /= n;
  const g2 = m4 / (sd ** 4) - 3;
  return g2;
}

function percentile(arr: number[], p: number) {
  if (arr.length === 0) return NaN;
  const a = [...arr].sort((x, y) => x - y);
  if (p <= 0) return a[0];
  if (p >= 100) return a[a.length - 1];
  const pos = (p / 100) * (a.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  const g = pos - lo; return a[lo] * (1 - g) + a[hi] * g;
}

function freedmanDiaconisBins(arr: number[]) {
  const n = arr.length;
  const a = [...arr].sort((x, y) => x - y);
  const q1 = percentile(a, 25), q3 = percentile(a, 75);
  const iqr = Math.max(q3 - q1, 1e-12);
  const minV = a[0], maxV = a[a.length - 1];
  let binWidth = 2 * iqr / Math.cbrt(Math.max(n, 1));
  if (!isFinite(binWidth) || binWidth <= 0) binWidth = (maxV - minV) || 1; // fallback
  let bins = Math.ceil((maxV - minV) / binWidth);
  if (!isFinite(bins) || bins <= 0) bins = 1;
  bins = Math.min(512, Math.max(1, bins)); // cap
  return { bins, minV, maxV };
}

function histogram(arr: number[]) {
  if (!arr || arr.length === 0) return [] as {x:number; density:number}[];
  const rawMin = Math.min(...arr), rawMax = Math.max(...arr);
  if (!isFinite(rawMin) || !isFinite(rawMax) || rawMin === rawMax) {
    return [{ x: rawMin, density: 1 }];
  }
  const { bins, minV, maxV } = freedmanDiaconisBins(arr);
  const safeBins = Math.max(1, Math.min(512, isFinite(bins) ? bins : 50));
  const counts = Array(safeBins).fill(0);
  const width = (maxV - minV) / safeBins || 1; // guard zero width
  for (const v of arr) {
    let idx = Math.floor((v - minV) / width);
    if (idx < 0) idx = 0;
    if (idx >= safeBins) idx = safeBins - 1;
    counts[idx] += 1;
  }
  const n = arr.length;
  return counts.map((c, i) => ({ x: minV + (i + 0.5) * width, density: c / (n * width) }));
}

function cdfData(arr: number[], maxPoints = 400) {
  if (!arr || arr.length === 0) return [] as {x:number; p:number}[];
  const a = [...arr].sort((x, y) => x - y);
  const n = a.length; const step = Math.max(1, Math.floor(n / maxPoints));
  const denom = Math.max(1, n - 1);
  const data: {x:number; p:number}[] = [];
  for (let i = 0; i < n; i += step) {
    data.push({ x: a[i], p: i / denom });
  }
  data.push({ x: a[n - 1], p: 1 });
  return data;
}

function rankArray(arr: number[]) {
  const n = arr.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((i, j) => arr[i] - arr[j]);
  const ranks = Array(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i + 1;
    while (j < n && arr[idx[j]] === arr[idx[i]]) j++;
    const avgRank = (i + j - 1) / 2 + 1; // 1-based average rank
    for (let k = i; k < j; k++) ranks[idx[k]] = avgRank;
    i = j;
  }
  return ranks;
}

function pearson(x: number[], y: number[]) {
  const n = x.length; const mx = mean(x), my = mean(y);
  let num = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx; const dy = y[i] - my;
    num += dx * dy; sx += dx * dx; sy += dy * dy;
  }
  return num / Math.sqrt(sx * sy);
}

function spearman(x: number[], y: number[]) {
  const rx = rankArray(x); const ry = rankArray(y);
  return pearson(rx, ry);
}

// ---------- Linear algebra helpers for OLS ----------
function transpose<T>(A: T[][]) { return A[0].map((_, i) => A.map(row => row[i])); }
function matmul(A: number[][], B: number[][]) {
  const r = A.length, c = B[0].length, k = B.length;
  const out = Array.from({ length: r }, () => Array(c).fill(0));
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let s = 0; for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      out[i][j] = s;
    }
  }
  return out;
}
function gaussJordanInverse(M: number[][]) {
  const n = M.length; const A = M.map(row => row.slice());
  const I = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__ , j) => i===j?1:0));
  for (let col = 0; col < n; col++) {
    let piv = col; let maxv = Math.abs(A[col][col]);
    for (let r = col+1; r < n; r++) { if (Math.abs(A[r][col]) > maxv) {maxv = Math.abs(A[r][col]); piv = r;} }
    if (maxv < 1e-12) throw new Error("Singular matrix");
    if (piv !== col) { [A[col], A[piv]] = [A[piv], A[col]]; [I[col], I[piv]] = [I[piv], I[col]]; }
    const diag = A[col][col];
    for (let j = 0; j < n; j++) { A[col][j] /= diag; I[col][j] /= diag; }
    for (let r = 0; r < n; r++) if (r !== col) {
      const f = A[r][col];
      for (let j = 0; j < n; j++) { A[r][j] -= f * A[col][j]; I[r][j] -= f * I[col][j]; }
    }
  }
  return I;
}

function olsCoefficients(X: number[][], y: number[]) {
  const Xt = transpose(X) as number[][];
  const XtX = matmul(Xt, X);
  const XtX_inv = gaussJordanInverse(XtX);
  const XtY = Xt.map(row => [ row.reduce((s, v, i) => s + v * y[i], 0) ]);
  const beta = matmul(XtX_inv, XtY).map(row => row[0]);
  return { beta, XtX_inv };
}

function AIC_from_RSS(n: number, k: number, rss: number) { return n * Math.log(rss / n) + 2 * k; }

// Forward stepwise selection on standardized variables
function forwardStepwiseStandardized(Xvars: number[][], y: number[]) {
  const n = y.length; const p = Xvars[0].length; // no intercept
  const Xz = Array.from({ length: n }, (_, i) => Array(p).fill(0));
  const means = Array(p).fill(0), sds = Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    const col = Xvars.map(row => row[j]);
    const m = mean(col); const sd = Math.sqrt(variance(col, m)) || 1;
    means[j] = m; sds[j] = sd;
    for (let i = 0; i < n; i++) Xz[i][j] = (Xvars[i][j] - m) / sd;
  }
  const my = mean(y); const sdy = Math.sqrt(variance(y, my)) || 1;
  const yz = y.map(v => (v - my) / sdy);

  const remaining = Array.from({ length: p }, (_, j) => j);
  const selected: number[] = [];
  let currentAIC = Infinity; let coefs: number[] = [];

  while (remaining.length > 0) {
    let bestAIC = Infinity, bestIdx = -1, bestModel: { cols: number[]; beta: number[] } | null = null;
    for (const j of remaining) {
      const cols = [...selected, j];
      const Xc = Xz.map(row => [1, ...cols.map(c => row[c])]);
      const { beta } = olsCoefficients(Xc, yz);
      const yhat = Xc.map(row => beta.reduce((s, b, k) => s + b * row[k], 0));
      let rss = 0; for (let i = 0; i < n; i++) { const e = yz[i] - yhat[i]; rss += e * e; }
      const aic = AIC_from_RSS(n, 1 + cols.length, rss);
      if (aic < bestAIC) { bestAIC = aic; bestIdx = j; bestModel = { cols, beta }; }
    }
    if (bestAIC + 1e-8 < currentAIC && bestModel) {
      currentAIC = bestAIC;
      selected.push(bestIdx);
      remaining.splice(remaining.indexOf(bestIdx), 1);
      coefs = bestModel.beta.slice(1); // drop intercept
    } else {
      break;
    }
  }
  return { selected, stdCoefs: coefs };
}

// ---------- Iman–Conover rank correlation imposing ----------
function imanConoverReorder(samples: number[][], targetCorr: number[][], rng: ReturnType<typeof makeRNG>) {
  const N = samples[0].length; const p = samples.length;
  function cholesky(M: number[][]) {
    const n = M.length; const L = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = M[i][j];
        for (let k = 0; k < j; k++) s -= (L[i][k] * L[j][k]);
        if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-12)); else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  const L = cholesky(targetCorr);
  const Z = Array.from({ length: N }, () => Array(p).fill(0));
  for (let i = 0; i < N; i++) for (let j = 0; j < p; j++) Z[i][j] = rng.normal();
  const Lt = transpose(L) as number[][];
  const Zc = Z.map(row => Lt.map(col => col.reduce((s, v, i) => s + v * row[i], 0)));
  const out = Array.from({ length: p }, () => Array(N).fill(0));
  for (let j = 0; j < p; j++) {
    const col = Zc.map(r => r[j]);
    const order = Array.from({ length: N }, (_, i) => i).sort((a,b) => col[a] - col[b]);
    const srt = [...samples[j]].sort((a,b) => a - b);
    for (let k = 0; k < N; k++) out[j][order[k]] = srt[k];
  }
  return out;
}

// ---------- Main Simulation ----------
function runSimulation({ iterations = 10000, seed = 12345 }: {iterations?: number; seed?: number} = {}) {
  const rng = makeRNG(seed);
  const N = iterations;
  const plane: number[] = Array(N), accom_iid: number[] = Array(N), meals_iid: number[] = Array(N), tours_iid: number[] = Array(N), clothing: number[] = Array(N);
  for (let i = 0; i < N; i++) {
    plane[i] = samplePERT(5000, 7000, 12000, rng);
    accom_iid[i] = samplePERT(2000, 2500, 4000, rng);
    meals_iid[i] = samplePERT(500, 600, 900, rng);
    tours_iid[i] = 1500 + rng.uniform() * (4000 - 1500);
    clothing[i] = samplePERT(300, 400, 500, rng);
  }
  const targetCorr = [ [1, 0.85, 0.85], [0.85, 1, 0.85], [0.85, 0.85, 1] ];
  const [accom, meals, tours] = imanConoverReorder([accom_iid, meals_iid, tours_iid], targetCorr, rng);

  const med_cost: number[] = Array(N), theft_cost: number[] = Array(N);
  for (let i = 0; i < N; i++) {
    const medTrig = rng.uniform() < 0.10;
    const theftTrig = rng.uniform() < 0.10;
    const medImpact = 1000 + rng.uniform() * (5000 - 1000);
    const theftImpact = samplePERT(500, 750, 2000, rng);
    med_cost[i] = medTrig ? medImpact : 0;
    theft_cost[i] = theftTrig ? theftImpact : 0;
  }
  const fx: number[] = Array(N); for (let i = 0; i < N; i++) fx[i] = samplePERT(3.75, 4.0, 4.5, rng);

  const misc: number[] = Array(N).fill(1000);
  const base_total_fib: number[] = Array(N), risk_total_fib: number[] = Array(N), risked_total_fib: number[] = Array(N), total_usd: number[] = Array(N);
  for (let i = 0; i < N; i++) {
    const base = plane[i] + accom[i] + meals[i] + misc[i] + tours[i] + clothing[i];
    const risk = med_cost[i] + theft_cost[i];
    base_total_fib[i] = base;
    risk_total_fib[i] = risk;
    risked_total_fib[i] = base + risk;
    total_usd[i] = risked_total_fib[i] / fx[i];
  }

  const inputs: Record<string, number[]> = {
    "Plane Fare (FIB)": plane,
    "Accommodation Costs (FIB)": accom,
    "Meals (FIB)": meals,
    "Holiday Tours (FIB)": tours,
    "Clothing / Travel Gear (FIB)": clothing,
    "Medical Emergency Cost (FIB)": med_cost,
    "Theft/Lost Baggage Cost (FIB)": theft_cost,
    "FOREX Rate (FIB/USD)": fx,
    "Misc. Shopping Expenses (FIB)": misc,
  };

  const outputs: Record<string, number[]> = {
    "Total Base Budget (FIB)": base_total_fib,
    "Base + Risk Events (FIB)": risked_total_fib,
    "Base + Risk + FOREX (USD)": total_usd,
  };

  return { inputs, outputs, N, seed };
}

function buildStats(values: number[]) {
  const pcts: Record<string, number> = {}; for (let p = 0; p <= 100; p += 10) pcts[`P${p}`] = percentile(values, p);
  const s = {
    min: Math.min(...values),
    ...pcts,
    max: Math.max(...values),
    mean: mean(values),
    std: stddev(values),
    skew: skewness(values),
    kurtosis_excess: kurtosisExcess(values),
  };
  return s;
}

function useMonteCarlo() {
  const [state, setState] = useState<null | {inputs:Record<string,number[]>; outputs:Record<string,number[]>; N:number; seed:number}>(null);
  useEffect(() => {
    const res = runSimulation({ iterations: 10000, seed: 12345 });
    setState(res);
  }, []);
  return state;
}

// ---------- UI Components ----------
function StatTable({ stats }: {stats: Record<string, number>}) {
  const order = ["min", ...Array.from({length:11}, (_,i)=>`P${i*10}`), "max", "mean", "std", "skew", "kurtosis_excess"];
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr>
        </thead>
        <tbody>
          {order.filter(k=>stats[k]!==undefined).map(k => (
            <tr key={k} className="border-t"><td className="p-2">{k}</td><td className="p-2">{Number(stats[k]).toLocaleString(undefined,{maximumFractionDigits:6})}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PDFChart({ data }: {data: {x:number; density:number}[]}) {
  return (
    <div className="h-56 bg-gradient-to-tr from-indigo-50 to-emerald-50 rounded-xl p-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Area type="monotone" dataKey="density" fillOpacity={0.4} strokeWidth={1.2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function CDFChart({ data }: {data: {x:number; p:number}[]}) {
  return (
    <div className="h-56 bg-gradient-to-tr from-pink-50 to-sky-50 rounded-xl p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={{ fontSize: 11 }} />
          <YAxis domain={[0,1]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Line type="monotone" dataKey="p" dot={false} strokeWidth={1.2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function TornadoChart({ items, xLabel }: {items: {name:string; value:number}[]; xLabel: string}) {
  return (
    <div className="h-80 bg-gradient-to-tr from-rose-50 to-indigo-50 rounded-xl p-2">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart layout="vertical" data={items} margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" name={xLabel} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Card({ title, children }: {title: string; children: React.ReactNode}) {
  return (
    <section className="bg-white rounded-2xl shadow p-4 hover:shadow-lg transition">
      <h3 className="text-base font-semibold mb-3">{title}</h3>
      {children}
    </section>
  );
}

function ExecSummary({ stats, seed, iterations, topSpearman, topReg }: {stats: Record<string,number>; seed:number; iterations:number; topSpearman:{name:string; value:number}[]; topReg:{name:string; value:number}[]}) {
  const fmt = (v: number) => Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
  return (
    <section className="bg-white rounded-2xl shadow p-4 mb-4">
      <h2 className="text-lg font-semibold mb-2">Executive Summary</h2>
      <p className="text-slate-700 text-sm">
        We ran a Monte Carlo simulation ({iterations.toLocaleString()} iterations, seed {seed}) in-browser using a seeded RNG (mulberry32), PERT/Uniform marginals, and Iman–Conover to impose the stated correlations between Accommodation, Meals, and Holiday Tours. Risk events (Medical Emergency; Theft/Lost Baggage) were modeled as 10% Bernoulli triggers with their respective impact distributions. FX uncertainty (FIB/USD) was modeled as PERT(3.75, 4.00, 4.50). USD totals are computed as FIB / FX.
      </p>
      <div className="mt-3 grid gap-3 md:grid-cols-3">
        <div className="bg-indigo-50 rounded-xl p-3">
          <div className="text-xs text-slate-600">Primary Output (USD)</div>
          <div className="text-sm mt-1">Mean: <b>{fmt(stats.mean)}</b></div>
          <div className="text-sm">P50: <b>{fmt(stats.P50)}</b></div>
          <div className="text-sm">P10–P90: <b>{fmt(stats.P10)} – {fmt(stats.P90)}</b></div>
        </div>
        <div className="bg-emerald-50 rounded-xl p-3">
          <div className="text-xs text-slate-600">Top Impacts (Spearman)</div>
          {topSpearman.map((t,i)=> (
            <div key={i} className="text-sm">{i+1}. {t.name}: <b>{t.value.toFixed(3)}</b></div>
          ))}
        </div>
        <div className="bg-rose-50 rounded-xl p-3">
          <div className="text-xs text-slate-600">Top Impacts (Std. Coeffs)</div>
          {topReg.map((t,i)=> (
            <div key={i} className="text-sm">{i+1}. {t.name}: <b>{t.value.toFixed(3)}</b></div>
          ))}
        </div>
      </div>
      <p className="text-slate-600 text-xs mt-3">
        Base budget elements are always included; uncertainties in Plane Fare, Accommodation, Meals, Holiday Tours, Clothing, and FX are captured via distributions; Misc. Shopping is deterministic (1000 FIB). Sensitivity rankings reflect monotonic and linear contributions respectively and may diverge when relationships are non-linear.
      </p>
    </section>
  );
}

export default function App() {
  const state = useMonteCarlo();

  const prepared = useMemo(() => {
    if (!state) return null;
    const { inputs, outputs } = state;

    // Build stats + charts for inputs & outputs
    const inputCards = Object.entries(inputs).map(([name, arr]) => {
      const stats = buildStats(arr);
      const pdf = histogram(arr);
      const cdf = cdfData(arr);
      return { name, stats, pdf, cdf };
    });

    const outputCardsAll = Object.entries(outputs).map(([name, arr]) => {
      const stats = buildStats(arr);
      const pdf = histogram(arr);
      const cdf = cdfData(arr);
      return { name, stats, pdf, cdf };
    });

    // Sensitivity for primary output
    const primaryName = "Base + Risk + FOREX (USD)";
    const Y = outputs[primaryName];
    const uncertainInputs = Object.entries(inputs)
      .filter(([k]) => k !== "Misc. Shopping Expenses (FIB)")
      .map(([k, v]) => ({ name: k, values: v }));

    // Spearman (sorted: highest impact first)
    const spearmanItems = uncertainInputs.map(({ name, values }) => ({ name, value: spearman(values, Y) }));
    spearmanItems.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    // Standardized stepwise regression (sorted: highest impact first)
    const Xvars = uncertainInputs.map(u => u.values);
    const Xcols = Xvars[0].map((_, i) => Xvars.map(col => col[i])); // n x p
    const { selected, stdCoefs } = forwardStepwiseStandardized(Xcols, Y);
    const itemsReg = selected.map((j, idx) => ({ name: uncertainInputs[j].name, value: stdCoefs[idx] }));
    itemsReg.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    // Separate primary output card first, others after
    const primaryCard = outputCardsAll.find(c => c.name === primaryName)!;
    const otherOutputCards = outputCardsAll.filter(c => c.name !== primaryName);

    // Top impacts for executive summary
    const topSpearman = spearmanItems.slice(0, 3);
    const topReg = itemsReg.slice(0, 3);

    return { inputCards, primaryCard, otherOutputCards, spearmanItems, itemsReg, primaryName, topSpearman, topReg };
  }, [state]);

  if (!prepared) return <div className="p-6">Running simulation…</div>;

  const { inputCards, primaryCard, otherOutputCards, spearmanItems, itemsReg, primaryName, topSpearman, topReg } = prepared;

  return (
    <div className="max-w-6xl mx-auto p-4">
      <header className="sticky top-0 bg-white/80 backdrop-blur z-10 rounded-xl shadow px-4 py-3 mb-4">
        <h1 className="text-xl font-semibold">Holiday Budget – Monte Carlo Simulation</h1>
        <p className="text-slate-600 text-sm mt-1">Iterations: 10,000 · Seed: 12345 · Assumption: FX is FIB/USD (USD = FIB / FX)</p>
      </header>

      {/* Executive Summary at the top */}
      <ExecSummary stats={primaryCard.stats} seed={state.seed} iterations={state.N} topSpearman={topSpearman} topReg={topReg} />

      {/* Outputs first */}
      <h2 className="text-lg font-semibold mb-2">Outputs</h2>

      {/* Primary output first, followed by tornado charts */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card key={primaryCard.name} title={primaryCard.name}>
          <div className="grid gap-3 md:grid-cols-2">
            <PDFChart data={primaryCard.pdf} />
            <CDFChart data={primaryCard.cdf} />
          </div>
          <div className="mt-3"><StatTable stats={primaryCard.stats} /></div>
        </Card>

        <Card title="Tornado – Spearman Rank Correlation">
          <p className="text-slate-600 text-sm mb-2">Highest absolute correlations shown at the top.</p>
          <TornadoChart items={spearmanItems} xLabel={`Spearman with ${primaryName}`} />
        </Card>
        <Card title="Tornado – Stepwise Regression (Standardized Coefficients)">
          <p className="text-slate-600 text-sm mb-2">Forward stepwise OLS on standardized variables.</p>
          {itemsReg.length > 0 ? (
            <TornadoChart items={itemsReg} xLabel="Std. Coefficient" />
          ) : (
            <div className="p-4 text-sm text-slate-600">No variables selected by stepwise regression.</div>
          )}
        </Card>
      </div>

      {/* Remaining outputs */}
      <div className="grid gap-4 md:grid-cols-2 mt-4">
        {otherOutputCards.map(card => (
          <Card key={card.name} title={card.name}>
            <div className="grid gap-3 md:grid-cols-2">
              <PDFChart data={card.pdf} />
              <CDFChart data={card.cdf} />
            </div>
            <div className="mt-3"><StatTable stats={card.stats} /></div>
          </Card>
        ))}
      </div>

      {/* Inputs after outputs */}
      <h2 className="text-lg font-semibold mt-6 mb-2">Inputs</h2>
      <div className="grid gap-4 md:grid-cols-2">
        {inputCards.map(card => (
          <Card key={card.name} title={card.name}>
            <div className="grid gap-3 md:grid-cols-2">
              <PDFChart data={card.pdf} />
              <CDFChart data={card.cdf} />
            </div>
            <div className="mt-3"><StatTable stats={card.stats} /></div>
          </Card>
        ))}
      </div>

      <footer className="mt-8 text-xs text-slate-500">
        Built client-side for reproducibility. Values may vary slightly from server-run due to numeric differences. Designed with subtle gradients & smooth UI.
      </footer>
    </div>
  );
}
