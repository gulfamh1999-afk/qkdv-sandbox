import os, io, hashlib, datetime as dt
import numpy as np, pandas as pd, streamlit as st
from sklearn.decomposition import PCA

st.set_page_config(page_title="Quantum Kernel DevKit — Sandbox (Redacted)", layout="wide")
st.title("🧬 Quantum Kernel DevKit — Sandbox (Redacted)")

# --- Config via environment / secrets ---
EXPIRES = os.getenv("SANDBOX_EXPIRES_AT", "")
PASSWORD = os.getenv("SANDBOX_PASSWORD") or st.secrets.get("SANDBOX_PASSWORD", None)
WATERMARK = os.getenv("DEMO_WATERMARK") or st.secrets.get(
    "DEMO_WATERMARK", "© Gfam Quantum Kernel DevKit — Demo"
)

# --- Expiry check ---
def expired(expires_str):
    if not expires_str:
        return False
    try:
        return dt.datetime.utcnow() > dt.datetime.fromisoformat(expires_str.replace("Z", ""))
    except Exception:
        return False

if EXPIRES and expired(EXPIRES):
    st.error("🔒 This sandbox has expired. Please contact the maintainer for a refreshed link.")
    st.stop()

# --- Password gate ---
if PASSWORD:
    if st.text_input("Enter access password", type="password") != PASSWORD:
        st.warning("Access password required.")
        st.stop()

st.info("Demo runs on public, de-identified data. Internals are redacted; compiled core not included.")

# --- Load public demo data ---
DEMO_PATH = "public_demo.csv"   # Updated to match current repo structure
if not os.path.exists(DEMO_PATH):
    st.error(f"⚠️ Demo dataset missing at '{DEMO_PATH}'. Please ensure it exists in the repo root.")
    st.stop()

df = pd.read_csv(DEMO_PATH)

# --- Step 1: Preview dataset ---
st.subheader("1) Preview demo dataset")
st.dataframe(df.head())

# --- Step 2: θ-embedding (redacted) ---
st.subheader("2) θ-embedding (redacted demo)")
st.caption("This uses a stub embedding for demo purposes. The production transform lives in a compiled core.")
n_components = 3 if df.shape[1] < 6 else 3
features = df.select_dtypes(include=[float, int]).values
pca = PCA(n_components=min(n_components, features.shape[1])).fit_transform(features)
eps = 1e-9
normed = (pca - pca.min(axis=0)) / (pca.ptp(axis=0) + eps)
bounded = 2 * normed - 1
theta = np.arcsin(np.clip(bounded, -1, 1))
st.write("θ shape:", theta.shape)

# --- Step 3: Run loop (SPSA-like demo) ---
st.subheader("3) Run loop (SPSA-like demo)")
np.random.seed(7)
def obj(th):
    return float(np.mean(np.cos(th) + 0.2 * np.sin(3 * th)))

iters = 60
alpha, gamma = 0.1, 0.101
x = theta.mean(axis=0)
hist = []
for k in range(1, iters + 1):
    ck = 1.0 / (k ** gamma)
    delta = np.random.choice([-1, 1], size=x.shape)
    f_plus = obj(x + ck * delta)
    f_minus = obj(x - ck * delta)
    gk = (f_plus - f_minus) / (2 * ck * delta + 1e-9)
    ak = alpha / (k ** 0.602)
    x = x - ak * gk
    hist.append(f_plus)

st.line_chart(hist)

# --- Step 4: Shortlist ---
st.subheader("4) Shortlist")
scores = (theta @ x)
shortlist = (
    pd.DataFrame({"id": df.index, "score": scores})
    .sort_values("score")
    .head(25)
)
st.dataframe(shortlist)

# --- Step 5: Export (watermarked) ---
st.subheader("5) Export (watermarked)")
def tamper_hash(d: pd.DataFrame) -> str:
    b = d.to_csv(index=False).encode()
    return hashlib.sha256(b).hexdigest()

export_df = shortlist.copy()
export_df["watermark"] = WATERMARK
export_df["tamper_hash"] = tamper_hash(export_df)

st.download_button(
    "⬇️ Download shortlist (CSV, watermarked)",
    data=export_df.to_csv(index=False).encode(),
    file_name="shortlist_demo.csv",
    mime="text/csv"
)

st.caption("© Gfam Quantum Kernel DevKit — Demo • No reverse engineering • Auto-delete policy applies.")
