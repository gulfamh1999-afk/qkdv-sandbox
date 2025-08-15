import os, io, time, hashlib, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import altair as alt

# ---------- Optional deps (graceful fallback) ----------
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    QISKIT_OK = True
except Exception:
    QISKIT_OK = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ----------------- Page / security -----------------
st.set_page_config(page_title="Quantum Kernel DevKit — Sandbox (Upgraded)", layout="wide")
st.title("🧬 Quantum Kernel DevKit — Sandbox (Upgraded, Redacted)")

EXPIRES   = os.getenv("SANDBOX_EXPIRES_AT", "")
PASSWORD  = os.getenv("SANDBOX_PASSWORD") or st.secrets.get("SANDBOX_PASSWORD", None)
WATERMARK = os.getenv("DEMO_WATERMARK") or st.secrets.get("DEMO_WATERMARK", "© Gfam Quantum Kernel DevKit — Demo")

def expired(expires_str: str) -> bool:
    if not expires_str:
        return False
    try:
        return dt.datetime.utcnow() > dt.datetime.fromisoformat(expires_str.replace("Z",""))
    except Exception:
        return False

if EXPIRES and expired(EXPIRES):
    st.error("🔒 This sandbox has expired. Ask the maintainer for a refreshed link.")
    st.stop()

if PASSWORD:
    if st.text_input("Enter access password", type="password") != PASSWORD:
        st.warning("Access password required.")
        st.stop()

st.info("Demo runs on public, de-identified data. Internals are redacted; compiled core not included.")

# ----------------- Load data -----------------
CANDIDATES = ["public_demo.csv", os.path.join("data", "public_demo.csv")]
DEMO_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not DEMO_PATH:
    st.error("⚠️ Demo dataset missing. Put it in repo root as `public_demo.csv` or inside `data/public_demo.csv`.")
    st.stop()

df = pd.read_csv(DEMO_PATH)

st.subheader("1) Preview demo dataset")
st.dataframe(df.head(), use_container_width=True)

# ----------------- θ-embedding (redacted) -----------------
st.subheader("2) θ-embedding (redacted demo)")
st.caption("This uses a stub embedding for demo purposes. The production transform lives in a compiled core.")

num_df = df.select_dtypes(include=[np.number])
if num_df.shape[1] == 0:
    num_df = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
if num_df.shape[1] == 0:
    st.error("No numeric columns found to embed. Add a couple of numeric columns to the demo CSV.")
    st.stop()

features = num_df.to_numpy(dtype=float)
n_components = min(3, features.shape[1])
pca_model = PCA(n_components=n_components)
pca_feats = pca_model.fit_transform(features).astype(float)

eps = 1e-9
mins   = np.min(pca_feats, axis=0)
ranges = np.ptp(pca_feats, axis=0)
ranges[ranges < eps] = 1.0
normed  = (pca_feats - mins) / (ranges + eps)
bounded = 2.0*normed - 1.0
theta   = np.arcsin(np.clip(bounded, -1.0, 1.0))
st.write("θ shape:", theta.shape)

# ----------------- Objective choices -----------------
st.subheader("3) Choose objective & toggles")
use_estimator = st.checkbox("Use Estimator-backed QAOA (4-qubit MaxCut)", value=False, disabled=not QISKIT_OK)
if use_estimator and not QISKIT_OK:
    st.warning("Qiskit not available; falling back to toy objective. Add `qiskit>=1.2` to requirements.txt to enable.")

iters_default = 30 if use_estimator else 60
iterations = st.slider("Iterations", min_value=10, max_value=120, value=iters_default, step=5)

col_tog1, col_tog2, col_sigma = st.columns([1,1,2])
with col_tog1:
    noise_on = st.checkbox("Noise ON", value=True)
with col_tog2:
    mitigation_on = st.checkbox("Mitigation ON (demo ZNE)", value=True, disabled=not noise_on)
with col_sigma:
    sigma = st.slider("Noise level (σ, demo)", 0.0, 0.2, 0.05, 0.01)

tol = st.slider("Tolerance (absolute)", 0.001, 0.05, 0.01, 0.001)
mins_per_iter = st.slider("Minutes per iteration (proxy)", 0.1, 5.0, 0.5, 0.1)

# ----------------- Objective functions -----------------
def toy_objective(params: np.ndarray) -> float:
    """Smooth multimodal objective (expectation-like)."""
    val = float(np.mean(np.cos(params) + 0.2*np.sin(3*params)))
    return val

def add_noise_and_mitigate(val: float, rng: np.random.RandomState) -> float:
    """Demo noise + zero-noise extrapolation (2-point)."""
    if not noise_on:
        return val
    e1 = val + rng.normal(0, sigma)  # base noise
    if mitigation_on:
        e2 = val + rng.normal(0, 2*sigma)  # scaled twice
        # simple linear extrapolation to zero-noise
        return float(2*e1 - e2)
    return float(e1)

# QAOA MaxCut on 4-cycle (0-1-2-3-0)
edges = [(0,1),(1,2),(2,3),(3,0)]
n_qubits = 4

def qaoa_expectation(params: np.ndarray, p_layers: int, rng: np.random.RandomState) -> float:
    """Estimator-backed cost <Z_i Z_j> sum for MaxCut 4-cycle."""
    if not QISKIT_OK:
        return toy_objective(params)
    # params: [gamma_1..gamma_p, beta_1..beta_p]
    gammas = params[:p_layers]
    betas  = params[p_layers:2*p_layers]

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for l in range(p_layers):
        gamma = gammas[l]
        beta  = betas[l]
        # ZZ on edges
        for (i,j) in edges:
            qc.cx(i, j)
            qc.rz(2*gamma, j)
            qc.cx(i, j)
        # mixer
        for q in range(n_qubits):
            qc.rx(2*beta, q)

    # Observable: sum Z_i Z_j (we'll minimize negative of MaxCut to behave like "energy")
    paulis = []
    coeffs = []
    for (i,j) in edges:
        z = ["I"]*n_qubits
        z[i] = "Z"; z[j] = "Z"
        paulis.append("".join(reversed(z)))  # Qiskit little-endian
        coeffs.append(1.0)
    observable = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

    est = Estimator()
    exp = est.run([qc], [observable]).result().values[0]  # expectation of sum Z_iZ_j
    # map to "energy-like" to minimize: smaller is better -> use negative of cut value proxy
    val = -float(exp) / len(edges)
    return add_noise_and_mitigate(val, rng)

# ----------------- Optimizers -----------------
def spsa(objective, x0, iters, seed, p_layers=None):
    rng = np.random.RandomState(seed)
    alpha, gamma = 0.1, 0.101
    x = x0.copy().astype(float)
    hist = []
    for k in range(1, iters+1):
        ck = 1.0/(k**gamma)
        delta = rng.choice([-1, 1], size=x.shape)
        f_plus  = objective(x + ck*delta) if p_layers is None else objective(x + ck*delta, p_layers, rng)
        f_minus = objective(x - ck*delta) if p_layers is None else objective(x - ck*delta, p_layers, rng)
        gk = (f_plus - f_minus)/(2*ck*delta + 1e-9)
        ak = alpha/(k**0.602)
        x = x - ak*gk
        hist.append(float(f_plus))
    return np.array(hist)

def random_search(objective, x0, iters, seed, p_layers=None):
    rng = np.random.RandomState(seed)
    x = x0.copy().astype(float)
    best = np.inf
    hist = []
    for k in range(1, iters+1):
        cand = x + rng.normal(0, 0.2, size=x.shape)
        f = objective(cand) if p_layers is None else objective(cand, p_layers, rng)
        if f < best:
            best = f
            x = cand
        hist.append(float(best))
    return np.array(hist)

# ----------------- Run experiments (3 seeds for bands) -----------------
st.subheader("4) Run optimizers & compare")
start = time.time()

if use_estimator and QISKIT_OK:
    dim = 2*1  # start with p=1 (two params gamma,beta) for the optimizer curves
    p_layers_for_curves = 1
    objective_fn = lambda params: qaoa_expectation(params, p_layers_for_curves, np.random.RandomState(0))
else:
    dim = theta.shape[1]
    objective_fn = lambda params: add_noise_and_mitigate(toy_objective(params), np.random.RandomState(0))

seeds = [7, 17, 27]
spsa_curves = []
base_curves = []
for s in seeds:
    x0 = np.mean(theta, axis=0)[:dim]
    spsa_curves.append(spsa(objective_fn, x0, iterations, seed=s))
    base_curves.append(random_search(objective_fn, x0, iterations, seed=s))

spsa_arr = np.vstack(spsa_curves)
base_arr = np.vstack(base_curves)

# Means & bands
spsa_mean = spsa_arr.mean(axis=0); spsa_min = spsa_arr.min(axis=0); spsa_max = spsa_arr.max(axis=0)
base_mean = base_arr.mean(axis=0); base_min = base_arr.min(axis=0); base_max = base_arr.max(axis=0)

# Iterations-to-tolerance (vs best SPSA mean)
best_val = float(np.min(spsa_mean))
def iters_to_tol(arr, best, tol_abs):
    run_min = np.minimum.accumulate(arr)
    hit = np.where(run_min <= best + tol_abs)[0]
    return int(hit[0]+1) if hit.size else arr.size

spsa_iters = iters_to_tol(spsa_mean, best_val, tol)
base_iters = iters_to_tol(base_mean, best_val, tol)
ratio = base_iters / max(1, spsa_iters)

runtime_s = time.time() - start

# ----------------- Plot with Altair (band + minima point) -----------------
st.markdown("**Comparison chart (mean ± band across 3 seeds)**")

def band_df(curve_mean, curve_min, curve_max, label):
    it = np.arange(1, curve_mean.size+1)
    return pd.DataFrame({
        "iter": np.concatenate([it, it, it]),
        "value": np.concatenate([curve_min, curve_mean, curve_max]),
        "series": (["min"]*it.size) + (["mean"]*it.size) + (["max"]*it.size),
        "label": [label]*(3*it.size)
    })

df_spsa = band_df(spsa_mean, spsa_min, spsa_max, "SPSA")
df_base = band_df(base_mean, base_min, base_max, "Baseline")

chart = alt.Chart(df_spsa).mark_line().encode(
    x=alt.X('iter:Q', title='Iteration'),
    y=alt.Y('value:Q', title='Expectation-like'),
    color=alt.Color('series:N', scale=alt.Scale(domain=['mean','min','max']))
).properties(width=800, height=300)

band = alt.Chart(pd.DataFrame({
    "iter": np.arange(1, spsa_mean.size+1),
    "low": spsa_min,
    "high": spsa_max
})).mark_area(opacity=0.2).encode(x='iter:Q', y='low:Q', y2='high:Q')

min_idx = int(np.argmin(spsa_mean)) + 1
min_point = alt.Chart(pd.DataFrame({"iter":[min_idx], "value":[np.min(spsa_mean)]})).mark_point(size=80).encode(
    x='iter:Q', y='value:Q'
)

base_line = alt.Chart(pd.DataFrame({
    "iter": np.arange(1, base_mean.size+1),
    "value": base_mean
})).mark_line(strokeDash=[4,4]).encode(x='iter:Q', y='value:Q')

st.altair_chart(band + chart + base_line + min_point, use_container_width=True)

st.success(f"Reached tolerance in **{spsa_iters} iters** (SPSA) vs **{base_iters} iters** (baseline) — approx **{ratio:.1f}× faster**.")

# ----------------- KPIs -----------------
st.subheader("5) KPIs")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Iterations to tolerance (SPSA)", spsa_iters)
with col2:
    st.metric("Iterations to tolerance (Baseline)", base_iters)
with col3:
    final_delta = float(base_mean[-1] - spsa_mean[-1])
    st.metric("Final Δ vs baseline", f"{final_delta:+.4f}")
with col4:
    saved_min = max(0, base_iters - spsa_iters) * mins_per_iter
    st.metric("Time saved / scientist (proxy)", f"{saved_min:.1f} min")

# Decoder-stress slope (QAOA only, tiny sweep over depth p)
if use_estimator and QISKIT_OK:
    st.markdown("**Decoder-stress slope (QAOA depth sweep)**")
    p_max = 3
    sweep_vals = []
    for p in range(1, p_max+1):
        dim_p = 2*p
        x0 = np.zeros(dim_p)
        curve = spsa(lambda params: qaoa_expectation(params, p, np.random.RandomState(123)), x0, 15, seed=11, p_layers=p)
        sweep_vals.append(float(np.min(curve)))
    p_axis = np.arange(1, p_max+1)
    slope = float(np.polyfit(p_axis, sweep_vals, 1)[0])
    st.line_chart(pd.DataFrame({"p": p_axis, "min_expectation": sweep_vals}).set_index("p"))
    st.caption(f"Decoder-stress slope (Δ expectation per depth): **{slope:+.4f}**")

# ----------------- Reproducibility -----------------
st.subheader("6) Reproducibility")
st.json({
    "seeds": seeds,
    "sample_size": int(df.shape[0]),
    "runtime_seconds": round(runtime_s, 3),
    "tolerance": tol,
    "noise_sigma": sigma if noise_on else 0.0,
    "mitigation": bool(mitigation_on and noise_on),
    "objective": "QAOA MaxCut (Estimator)" if (use_estimator and QISKIT_OK) else "Toy expectation"
})

# ----------------- Shortlist + Export (CSV) -----------------
st.subheader("7) Shortlist")
# simple scoring from current theta & last SPSA params proxy
scores = (theta @ np.ones(theta.shape[1]))  # demo-only scoring
shortlist = pd.DataFrame({"id": df.index, "score": scores}).sort_values("score").head(25).reset_index(drop=True)
st.dataframe(shortlist, use_container_width=True)

st.subheader("8) Export (watermarked)")
def tamper_hash(d: pd.DataFrame) -> str:
    raw = d.to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()

export_df = shortlist.copy()
export_df["watermark"]   = WATERMARK
export_df["tamper_hash"] = tamper_hash(export_df)

st.download_button(
    "⬇️ Download shortlist (CSV, watermarked)",
    data=export_df.to_csv(index=False).encode(),
    file_name="shortlist_demo.csv",
    mime="text/csv"
)

# ----------------- Export summary PDF / TXT -----------------
st.subheader("9) Export summary report")
summary_lines = [
    "Quantum Kernel DevKit — Demo Summary",
    f"Objective: {'QAOA MaxCut (Estimator)' if (use_estimator and QISKIT_OK) else 'Toy expectation'}",
    f"Iterations (SPSA vs Baseline): {spsa_iters} vs {base_iters} (~{ratio:.1f}x faster)",
    f"Final Δ vs baseline: {final_delta:+.4f}",
    f"Time saved proxy: {saved_min:.1f} min (at {mins_per_iter} min/iter)",
    f"Tolerance: {tol}",
    f"Noise sigma: {sigma if noise_on else 0.0} | Mitigation: {bool(mitigation_on and noise_on)}",
    f"Seeds: {seeds} | Sample size: {int(df.shape[0])} | Runtime: {round(runtime_s,3)} s",
]

if REPORTLAB_OK:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 14); c.drawString(72, y, "Quantum Kernel DevKit — Demo Summary"); y -= 24
    c.setFont("Helvetica", 11)
    for line in summary_lines[1:]:
        c.drawString(72, y, line); y -= 16
    c.drawString(72, y-8, WATERMARK)
    c.showPage(); c.save()
    pdf_bytes = buf.getvalue()
    st.download_button("📄 Download summary (PDF)", data=pdf_bytes, file_name="qkdv_summary.pdf", mime="application/pdf")
else:
    txt_bytes = ("\n".join(summary_lines) + f"\n{WATERMARK}\n").encode()
    st.download_button("📄 Download summary (TXT)", data=txt_bytes, file_name="qkdv_summary.txt", mime="text/plain")

st.caption("© Gfam Quantum Kernel DevKit — Demo • No reverse engineering • Auto-delete policy applies.")
