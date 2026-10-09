"""
app.py — Phishing URL Detector
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer

from features import extract_features, full_analysis


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "best_model.pkl")
DATA_PATH = os.path.join(BASE, "dataset.csv")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.card {
    background: #ffffff;
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    color: #111827;
}

.stat {
    background: #f6f8fa;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    color: #111827 !important;
    min-height: 100px;
}

.stat .lbl {
    font-size: 0.8rem;
    color: #4b5563 !important;
}

.stat .val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #111827 !important;
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.p-red {
    background: #fdecec;
    color: #c0392b !important;
}

.p-amber {
    background: #fef6e7;
    color: #b9770e !important;
}

.p-green {
    background: #eafaf1;
    color: #1e8449 !important;
}

.p-grey {
    background: #eef0f2;
    color: #566573 !important;
}

.mono {
    font-family: ui-monospace, Menlo, monospace;
    font-size: 0.85rem;
    word-break: break-all;
    color: #111827 !important;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1rem;
    color: #111827;
}

.topbar .brand {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1f2937 !important;
}

.topbar .brand small {
    font-size: 0.72rem;
    font-weight: 400;
    color: #6b7280 !important;
    margin-left: 8px;
}

.topbar .right {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 0.8rem;
    color: #6b7280;
}

.topbar .who {
    text-align: right;
    line-height: 1.25;
}

.topbar .who b {
    color: #1f2937 !important;
}

.topbar .mdl {
    background: #eafaf1;
    color: #1e8449 !important;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    white-space: nowrap;
}

header[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 1.2rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_train(cols):

    if not os.path.exists(DATA_PATH):
        return None

    try:
        return pd.read_csv(DATA_PATH)[cols].values
    except Exception:
        return None


# ============================================================
# PILL
# ============================================================

def pill(level):

    cls = {
        "High": "p-red",
        "Elevated": "p-amber",
        "Medium": "p-amber",
        "Low": "p-green",
        "Neutral": "p-green"
    }.get(level, "p-grey")

    return f"<span class='pill {cls}'>{level}</span>"


# ============================================================
# GAUGE
# ============================================================

def gauge(prob):

    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(3.6, 2.1))

    for a0, a1, col in [
        (0, 60, "#1e8449"),
        (60, 120, "#e67e22"),
        (120, 180, "#c0392b")
    ]:

        ax.add_patch(
            mp.Wedge(
                (0, 0),
                1,
                180 - a1,
                180 - a0,
                width=0.34,
                facecolor=col,
                alpha=0.9
            )
        )

    ang = np.pi * (1 - prob)

    ax.plot(
        [0, 0.8 * np.cos(ang)],
        [0, 0.8 * np.sin(ang)],
        lw=3,
        color="#2c3e50"
    )

    ax.add_patch(
        mp.Circle(
            (0, 0),
            0.05,
            color="#2c3e50"
        )
    )

    ax.text(
        0,
        -0.18,
        f"{prob * 100:.0f}%",
        ha="center",
        fontsize=17,
        weight="bold",
        color="#111827"
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.35, 1.15)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    return fig


# ============================================================
# SHAP
# ============================================================

def shap_fig(model, X1, cols):

    expl = shap.TreeExplainer(model)

    sv = expl.shap_values(
        X1,
        check_additivity=False
    )

    if isinstance(sv, list):
        sv = sv[1]

    sv = np.array(sv)

    if sv.ndim == 3:
        sv = sv[:, :, 1]

    vals = np.array(sv[0]).ravel()[:len(cols)]

    idx = np.argsort(np.abs(vals))[-10:]

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = [
        "#c0392b" if v > 0 else "#2980b9"
        for v in vals[idx]
    ]

    ax.barh(
        [cols[i] for i in idx],
        vals[idx],
        color=colors
    )

    ax.axvline(
        0,
        color="grey",
        lw=0.8,
        ls="--"
    )

    ax.set_xlabel(
        "← legitimate    SHAP contribution    phishing →"
    )

    ax.set_title(
        "SHAP — why this decision"
    )

    plt.tight_layout()

    return fig


# ============================================================
# LIME
# ============================================================

def lime_fig(model, X1, cols, Xtrain):

    ex = LimeTabularExplainer(
        Xtrain,
        feature_names=cols,
        class_names=["Legitimate", "Phishing"],
        mode="classification",
        discretize_continuous=True
    )

    e = ex.explain_instance(
        X1.values[0],
        model.predict_proba,
        num_features=8
    )

    fig = e.as_pyplot_figure()

    fig.set_size_inches(6, 4)

    plt.title(
        "LIME — local explanation"
    )

    plt.tight_layout()

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    art = load_model()

    if art:
        model_label = f"Model: {art['model_name']}"
        mdl_class = "mdl"
    else:
        model_label = "Run train.py first"
        mdl_class = "pill p-red"


    # ========================================================
    # TOP BAR
    # ========================================================

    topbar_html = f"""<div class="topbar">
<div class="brand">🛡️ URL Detector <small>Adaptive &amp; Explainable ML</small></div>
<div class="right">
<div class="who"><b>Poonam Thakur</b><br>Student ID: 23010203031</div>
<span class="{mdl_class}">{model_label}</span>
</div>
</div>"""

    st.markdown(
        topbar_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        "## Phishing URL threat analysis"
    )

    st.caption(
        "String-based analysis only. URLs are never visited."
    )


    # ========================================================
    # URL INPUT
    # ========================================================

    url = st.text_input(
        "URL to analyse",
        placeholder="http://secure-login-paypal.xyz/verify@account"
    )

    go = st.button(
        "🔍 Analyse",
        type="primary"
    )


    if not go or not url.strip():

        st.info(
            "Enter a URL and click Analyse."
        )

        return


    url = url.strip()


    # ========================================================
    # ANALYSIS
    # ========================================================

    a = full_analysis(url)

    ssl = a["ssl"]
    dns = a["dns"]
    geo = a["geo"]
    tok = a["tokens"]
    thr = a["threat"]


    # ========================================================
    # CURRENT URL
    # ========================================================

    url_html = f"""<div class="card">
<div style="font-size:0.8rem;color:#6b7280">Current URL</div>
<div class="mono">{url}</div>
</div>"""

    st.markdown(
        url_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # MODEL
    # ========================================================

    X1 = None
    is_phish = None
    proba = None

    if art:

        model = art["model"]
        cols = art["feature_cols"]

        X1 = pd.DataFrame(
            [extract_features(url)]
        )[cols]

        proba = model.predict_proba(X1)[0]

        is_phish = (
            model.predict(X1)[0] == 1
        )


    # ========================================================
    # TOP CARDS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    with c1:

        if is_phish is None:

            verdict = "—"

        elif is_phish:

            verdict = (
                "<span class='pill p-red'>Phishing</span>"
            )

        else:

            verdict = (
                "<span class='pill p-green'>Legitimate</span>"
            )

        html = f"""<div class="stat">
<div class="lbl">ML verdict</div>
<div style="margin-top:6px">{verdict}</div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PHISHING PROBABILITY
    # --------------------------------------------------------

    with c2:

        if proba is not None:

            phishing_probability = float(
                proba[1]
            )

            probability_text = (
                f"{phishing_probability * 100:.0f}%"
            )

        else:

            probability_text = "—"

        html = f"""<div class="stat">
<div class="lbl">Phishing probability</div>
<div class="val">{probability_text}</div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # THREAT SCORE
    # --------------------------------------------------------

    with c3:

        html = f"""<div class="stat">
<div class="lbl">Threat score</div>
<div class="val">{thr['score']}<span style="font-size:0.9rem;color:#6b7280">/100</span></div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # THREAT LEVEL
    # --------------------------------------------------------

    with c4:

        html = f"""<div class="stat">
<div class="lbl">Threat level</div>
<div style="margin-top:6px">{pill(thr['level'])}</div>
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    # ========================================================
    # GAUGE
    # ========================================================

    if proba is not None:

        g1, g2 = st.columns([1, 2])

        with g1:

            st.pyplot(
                gauge(proba[1])
            )

            plt.close()


        with g2:

            st.markdown(
                "**Threat score breakdown**"
            )

            if thr["reasons"]:

                st.markdown(
                    "\n".join(
                        f"- **+{p}** {m}"
                        for p, m in thr["reasons"]
                    )
                )

            else:

                st.markdown(
                    "- No risk indicators triggered"
                )


    st.markdown("---")


    # ========================================================
    # DETAILED ANALYSIS
    # ========================================================

    st.markdown(
        "#### Detailed analysis"
    )

    d1, d2, d3, d4 = st.columns(4)


    with d1:

        html = f"""<div class="card">
<b>🔒 SSL</b><br>
{ssl['verdict']}<br>
{pill(ssl['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with d2:

        html = f"""<div class="card">
<b>🌐 DNS</b><br>
{dns['tld']} · {dns['num_subdomains']} sub<br>
{pill(dns['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with d3:

        html = f"""<div class="card">
<b>📍 Region</b><br>
{geo['inferred_region']}<br>
{pill(geo['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    with d4:

        toks = (
            ", ".join(
                tok["suspicious_tokens"][:3]
            )
            or "none"
        )

        html = f"""<div class="card">
<b>🔑 Tokens</b><br>
{toks}<br>
{pill(tok['risk'])}
</div>"""

        st.markdown(
            html,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ========================================================
    # SHAP + LIME
    # ========================================================

    if art and X1 is not None:

        st.markdown(
            "#### Explainability"
        )

        e1, e2 = st.columns(2)


        with e1:

            try:

                st.pyplot(
                    shap_fig(
                        model,
                        X1,
                        cols
                    )
                )

                plt.close()

            except Exception as ex:

                st.warning(
                    f"SHAP unavailable: {ex}"
                )


        with e2:

            Xtr = load_train(cols)

            if Xtr is None:

                st.info(
                    "LIME needs training data."
                )

            else:

                try:

                    st.pyplot(
                        lime_fig(
                            model,
                            X1,
                            cols,
                            Xtr
                        )
                    )

                    plt.close()

                except Exception as ex:

                    st.warning(
                        f"LIME could not render: {ex}"
                    )


        # ====================================================
        # FEATURE TABLE
        # ====================================================

        with st.expander(
            "📊 Full feature table"
        ):

            feats = extract_features(url)

            st.dataframe(
                pd.DataFrame(
                    feats.items(),
                    columns=[
                        "Feature",
                        "Value"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.caption(
        "Adaptive & Explainable ML Framework · "
        "Poonam Thakur · "
        "Analysis is string-derived; URLs are not visited."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()