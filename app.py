import os, hashlib, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

# ---------------- UI & page ----------------
st.set_page_config(page_title="Quantum Kernel DevKit — Sandbox (Redacted)", layout="wide")
st.title("🧬 Quantum Kernel DevKit — Sandbox (Redacted)")

# ---------------- Secrets / env ----------------
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

# ---------------- Load data (robust path) ----------------
CANDIDATES = ["public_demo.csv", os.path.join("data", "public_demo.csv")]
DEMO_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not DEMO_PATH:
    st.error("⚠️ Demo dataset missing. Put it in repo root as `public_demo.csv` or inside `data/public_demo.csv`.")
    st.stop()

df = pd.read_csv(DEMO_PATH)

st.subheader("1) Preview demo dataset")
st.dataframe(df.head(), use_container_width=True)

# ---------------- θ-embedding (redacted) ----------------
# --- Step 2: θ-embedding (redacted demo) ---
st.subheader("2) θ-embedding (redacted demo)")
st.caption("This uses a stub embedding for demo purposes. The production transform lives in a compiled core.")

# Numeric features only (coerce if needed)
num_df = df.select_dtypes(include=[np.number])
if num_df.shape[1] == 0:
    num_df = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
if num_df.shape[1] == 0:
    st.error("No numeric columns found to embed. Add a couple of numeric columns to the demo CSV.")
    st.stop()

features = num_df.to_numpy(dtype=float)

# PCA -> do NOT name the array 'pca'
n_components = min(3, features.shape[1])
pca_model = PCA(n_components=n_components)
pca_feats = pca_model.fit_transform(features).astype(float)

# Safe normalize and map to angles
eps = 1e-9
mins   = np.min(pca_feats, axis=0)
ranges = np.ptp(pca_feats, axis=0)           # use numpy.ptp, not array.ptp attribute
ranges[ranges < eps] = 1.0                   # avoid divide-by-zero on constant cols
normed  = (pca_feats - mins) / (ranges + eps)
bounded = 2.0 * normed - 1.0
theta   = np.arcsin(np.clip(bounded, -1.0, 1.0))

st.write("θ shape:", theta.shape)

# ---------------- Run loop (SPSA-like demo) ----------------
st.subheader("3) Run loop (SPSA-like demo)")
np.random.seed(7)

def obj(th: np.ndarray) -> float:
    # Smooth multimodal objective to mimic an expectation landscape
    return float(np.mean(np.cos(th) + 0.2 * np.sin(3 * th)))

iters = 60
alpha, gamma = 0.1, 0.101
x = theta.mean(axis=0)
hist = []

for k in range(1, iters + 1):
    ck = 1.0 / (k ** gamma)
    delta = np.random.choice([-1, 1], size=x.shape)
    f_plus  = obj(x + ck * delta)
    f_minus = obj(x - ck * delta)
    gk = (f_plus - f_minus) / (2 * ck * delta + 1e-9)
    ak = alpha / (k ** 0.602)
    x = x - ak * gk
    hist.append(f_plus)

st.line_chart(pd.DataFrame({"expectation_like": hist}), use_container_width=True)

# ---------------- Shortlist ----------------
st.subheader("4) Shortlist")
scores = theta @ x
shortlist = (
    pd.DataFrame({"id": df.index, "score": scores})
    .sort_values("score")
    .head(25)
    .reset_index(drop=True)
)
st.dataframe(shortlist, use_container_width=True)

# ---------------- Export (watermarked) ----------------
st.subheader("5) Export (watermarked)")
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
    mime="text/csv",
)

st.caption("© Gfam Quantum Kernel DevKit — Demo • No reverse engineering • Auto-delete policy applies.")
