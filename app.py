import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Real Estate Valuation Hub",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .section-label { font-size: 0.75rem; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem; }
    .result-box { background: #1a1f2e; border-left: 4px solid #4a9eff; padding: 1.25rem 1.5rem; border-radius: 6px; margin-top: 1rem; }
    .result-value { font-size: 2.2rem; font-weight: 700; color: #4a9eff; }
    .result-label { font-size: 0.8rem; color: #aaa; margin-bottom: 0.25rem; }
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 1rem; }
    .stat-card { background: #1a1f2e; border-radius: 6px; padding: 0.85rem 1rem; }
    .stat-card .label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
    .stat-card .value { font-size: 1.25rem; font-weight: 600; color: #e8eaf0; }
    hr { border: none; border-top: 1px solid #2a2f3e; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    with open("housing_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("housing_model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model


try:
    scaler, model = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Run `python train.py` first to generate housing_scaler.pkl and housing_model.pkl.")
    st.stop()


st.title("Real Estate Valuation Hub")
st.write("Enter property details below to generate an AI-driven market value estimate.")

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-label">Property Size</p>', unsafe_allow_html=True)
    square_feet = st.number_input("Square Feet", min_value=500, max_value=6000, value=2500, step=50)
    lot_size = st.number_input("Lot Size (Acres)", min_value=0.10, max_value=3.00, value=1.20, step=0.05, format="%.2f")

with col2:
    st.markdown('<p class="section-label">Interior</p>', unsafe_allow_html=True)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=5, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    garage_spaces = st.slider("Garage Spaces", min_value=0, max_value=3, value=2)

with col3:
    st.markdown('<p class="section-label">Location & Community</p>', unsafe_allow_html=True)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=1995, step=1)
    zip_code = st.number_input("Zip Code", min_value=10000, max_value=99999, value=50001, step=1)
    crime_rate = st.slider("Crime Rate Index (0 = safest)", min_value=0.0, max_value=100.0, value=35.0, step=0.5)
    school_rating = st.slider("School Rating (1–10)", min_value=1, max_value=10, value=7)

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("Estimate Property Value", use_container_width=False, type="primary"):
    payload = pd.DataFrame([{
        "Bedrooms": bedrooms, "Bathrooms": bathrooms, "SquareFeet": square_feet,
        "YearBuilt": year_built, "GarageSpaces": garage_spaces, "LotSize": lot_size,
        "ZipCode": zip_code, "CrimeRate": crime_rate, "SchoolRating": school_rating
    }])
    scaled_payload = scaler.transform(payload)
    prediction = model.predict(scaled_payload)[0]

    r, c = st.columns([1, 2])
    with r:
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-value">${prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown(
            f"""<div class='stat-grid'>
            <div class='stat-card'><div class='label'>Bedrooms</div><div class='value'>{bedrooms}</div></div>
            <div class='stat-card'><div class='label'>Bathrooms</div><div class='value'>{bathrooms:.1f}</div></div>
            <div class='stat-card'><div class='label'>Square Feet</div><div class='value'>{square_feet:,}</div></div>
            <div class='stat-card'><div class='label'>Year Built</div><div class='value'>{year_built}</div></div>
            <div class='stat-card'><div class='label'>Crime Index</div><div class='value'>{crime_rate:.1f}</div></div>
            <div class='stat-card'><div class='label'>School Rating</div><div class='value'>{school_rating} / 10</div></div>
            </div>""",
            unsafe_allow_html=True
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Property Dataset")

try:
    raw_df = pd.read_csv("usa_housing_kaggle.csv")
    st.write(f"{len(raw_df):,} records loaded.")
    st.dataframe(
        raw_df.style.format({
            "Price": "${:,.0f}",
            "LotSize": "{:.2f}",
            "CrimeRate": "{:.2f}",
            "Bathrooms": "{:.1f}"
        }),
        use_container_width=True,
        height=400
    )
except FileNotFoundError:
    st.warning("usa_housing_kaggle.csv not found in the working directory.")
