import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ============================
# BASIC CONFIG
# ============================
st.set_page_config(
    page_title="Airbnb NYC – Full Project",
    page_icon="🏙️",
    layout="wide"
)

# ============================
# CUSTOM CSS FOR COLORFUL UI
# ============================
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: radial-gradient(circle at top left, #1d4ed8 0, #020617 35%, #0f172a 100%);
        color: #e5e7eb;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #111827);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
    }

    .subcaption {
        font-size: 0.95rem;
        color: #e5e7ebcc;
    }

    /* KPI cards */
    .kpi-card {
        padding: 1rem 1.2rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,64,175,0.85));
        border: 1px solid rgba(96, 165, 250, 0.5);
        box-shadow: 0 18px 40px rgba(15,23,42,0.9);
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e5e7eb;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ec4899, #a855f7, #6366f1);
        color: white;
        border-radius: 999px;
        padding: 0.5rem 1.3rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        filter: brightness(1.05);
        box-shadow: 0 12px 30px rgba(59,130,246,0.5);
    }

    /* Forms & containers */
    .glass-card {
        background: rgba(15,23,42, 0.85);
        border-radius: 1.3rem;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, 0.5);
        box-shadow: 0 18px 40px rgba(15,23,42,0.85);
    }

    /* Dataframes */
    .stDataFrame, .stTable {
        background-color: rgba(15,23,42,0.9) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# HEADER
# ============================
st.markdown('<div class="main-title">🏙️ Airbnb NYC – End-to-End Project</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subcaption">Phase 1: EDA  •  Phase 2: Market Explorer  •  Phase 3: Price Prediction</div>',
    unsafe_allow_html=True,
)
st.write("")

# ============================
# DATA + MODEL LOADING
# ============================
@st.cache_data
def load_data():
    # 👉 Change to your actual CSV path if needed
    df = pd.read_csv(
        "C:/Users/Radha/OneDrive/Pictures/Documents/DATA SCIENCE Airbnb project/data/AB_NYC_2019.csv"
    )
    return df

@st.cache_resource
def load_model():
    model_path = "C:/Users/Radha/OneDrive/Pictures/Documents/DATA SCIENCE Airbnb project/notebook/price_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# Columns we know are not used as features
DROP_COLS = ["id", "name", "host_id", "host_name", "last_review", "price"]

# Feature dataframe (same logic as training)
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X_full = df[feature_cols]

numeric_features = X_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_full.select_dtypes(include=["object"]).columns.tolist()

# ============================
# SIDEBAR NAVIGATION
# ============================
st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "",
    [
        "Phase 1 – Data Overview & EDA",
        "Phase 2 – Market Explorer Dashboard",
        "Phase 3 – Price Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** NYC Airbnb 2019\n\n**Tech:** Streamlit · Plotly · Sklearn")

# ============================
# KPI HELPER
# ============================
def show_kpis(filtered_df):
    total_listings = len(filtered_df)
    avg_price = filtered_df["price"].mean()
    median_price = filtered_df["price"].median()
    avg_reviews = filtered_df["number_of_reviews"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Listings</div>
                <div class="kpi-value">{total_listings:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Price</div>
                <div class="kpi-value">${avg_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Median Price</div>
                <div class="kpi-value">${median_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg. Reviews</div>
                <div class="kpi-value">{avg_reviews:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

# ============================
# PAGE 1 – PHASE 1: DATA OVERVIEW & EDA
# ============================
if page.startswith("Phase 1"):
    st.subheader("📊 Phase 1 – Data Understanding & EDA")

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("#### 🔍 Dataset Snapshot")
        st.write(f"**Shape:** `{df.shape[0]} rows × {df.shape[1]} columns`")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("#### 🧾 Column Info")
        st.write(df.dtypes)

        st.markdown("#### 📈 Summary Statistics (Numeric Columns)")
        st.dataframe(df.describe().T, use_container_width=True)

        st.markdown("---", unsafe_allow_html=True)

        st.markdown("#### 💰 Price Distribution")
        fig_price = px.histogram(
            df,
            x="price",
            nbins=50,
            title="Price Distribution (All Listings)",
        )
        fig_price.update_layout(
            bargap=0.05,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.9)",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig_price, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏠 Room Type Count")
            room_counts = df["room_type"].value_counts().reset_index()
            room_counts.columns = ["room_type", "count"]
            fig_room = px.bar(
                room_counts,
                x="room_type",
                y="count",
                title="Room Type Distribution",
            )
            fig_room.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.9)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig_room, use_container_width=True)

        with col2:
            st.markdown("#### 🌍 Listings by Neighbourhood Group")
            ng_counts = df["neighbourhood_group"].value_counts().reset_index()
            ng_counts.columns = ["neighbourhood_group", "count"]
            fig_ng = px.bar(
                ng_counts,
                x="neighbourhood_group",
                y="count",
                title="Listings per Neighbourhood Group",
            )
            fig_ng.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.9)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig_ng, use_container_width=True)

        st.markdown("#### 💸 Average Price by Neighbourhood Group")
        avg_price_ng = (
            df.groupby("neighbourhood_group", as_index=False)["price"]
            .mean()
            .sort_values("price", ascending=False)
        )
        fig_avg_ng = px.bar(
            avg_price_ng,
            x="neighbourhood_group",
            y="price",
            title="Average Price by Neighbourhood Group",
        )
        fig_avg_ng.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.9)",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig_avg_ng, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ============================
# PAGE 2 – PHASE 2: MARKET EXPLORER DASHBOARD
# ============================
elif page.startswith("Phase 2"):
    st.subheader("🧭 Phase 2 – Market Explorer Dashboard")

    st.sidebar.markdown("### 🎛 Filters")

    # Filters
    all_ng = sorted(df["neighbourhood_group"].dropna().unique().tolist())
    all_room_types = sorted(df["room_type"].dropna().unique().tolist())

    selected_ng = st.sidebar.multiselect(
        "Neighbourhood Group",
        options=all_ng,
        default=all_ng,
    )

    selected_room = st.sidebar.multiselect(
        "Room Type",
        options=all_room_types,
        default=all_room_types,
    )

    min_price, max_price = int(df["price"].min()), int(df["price"].max())
    selected_price = st.sidebar.slider(
        "Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=10,
    )

    min_nights_min, min_nights_max = int(df["minimum_nights"].min()), int(
        df["minimum_nights"].max()
    )
    selected_min_nights = st.sidebar.slider(
        "Minimum Nights (Max Filter)",
        min_value=min_nights_min,
        max_value=min_nights_max,
        value=min_nights_max,
    )

    # Apply filters
    filtered = df[
        (df["neighbourhood_group"].isin(selected_ng))
        & (df["room_type"].isin(selected_room))
        & (df["price"] >= selected_price[0])
        & (df["price"] <= selected_price[1])
        & (df["minimum_nights"] <= selected_min_nights)
    ]

    st.markdown("#### 📊 Filtered Overview")
    st.write(f"**Filtered Listings:** `{len(filtered)}` rows")
    show_kpis(filtered)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦 Price Distribution", "🏠 Room Types", "📍 Neighbourhoods", "🗺️ Map View"]
    )

    with tab1:
        st.markdown("#### 💰 Price Distribution (Filtered)")
        if len(filtered) > 0:
            fig = px.histogram(
                filtered,
                x="price",
                nbins=40,
                title="Filtered Price Distribution",
            )
            fig.update_layout(
                bargap=0.05,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.9)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected filters.")

    with tab2:
        st.markdown("#### 🏠 Room Type Breakdown")
        if len(filtered) > 0:
            room_counts_f = filtered["room_type"].value_counts().reset_index()
            room_counts_f.columns = ["room_type", "count"]
            fig = px.pie(
                room_counts_f,
                names="room_type",
                values="count",
                title="Room Type Share (Filtered)",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected filters.")

    with tab3:
        st.markdown("#### 🏙️ Top 20 Neighbourhoods by Average Price")
        if len(filtered) > 0:
            avg_price_nb = (
                filtered.groupby("neighbourhood", as_index=False)["price"]
                .mean()
                .sort_values("price", ascending=False)
                .head(20)
            )
            fig = px.bar(
                avg_price_nb,
                x="neighbourhood",
                y="price",
                title="Top 20 Neighbourhoods by Avg Price",
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.9)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected filters.")

    with tab4:
        st.markdown("#### 🗺️ Map View of Listings")
        if len(filtered) > 0:
            map_df = filtered.copy()
            if len(map_df) > 8000:
                map_df = map_df.sample(8000, random_state=42)

            fig = px.scatter_mapbox(
                map_df,
                lat="latitude",
                lon="longitude",
                color="price",
                size="number_of_reviews",
                hover_name="name",
                hover_data=["neighbourhood", "room_type", "price"],
                zoom=9,
                height=600,
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected filters.")

# ============================
# PAGE 3 – PHASE 3: PRICE PREDICTION
# ============================
elif page.startswith("Phase 3"):
    st.subheader("🧮 Phase 3 – Price Prediction")

    if model is None:
        st.error(
            "`price_model.pkl` not found. Please train and save the model first in your Phase 3 notebook."
        )
    else:
        st.write(
            "Fill in the details below to predict the **nightly price** of an Airbnb listing."
        )

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.markdown("#### 📍 Categorical Features")
            cat_inputs = {}
            for col in categorical_features:
                options = sorted(df[col].dropna().unique().tolist())
                default_val = options[0] if options else ""
                cat_inputs[col] = st.selectbox(
                    f"{col}", options=options, index=0 if options else None
                )

            st.markdown("#### 🔢 Numeric Features")
            num_inputs = {}
            for col in numeric_features:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_med = float(df[col].median())
                num_inputs[col] = st.number_input(
                    f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=col_med,
                )

            submitted = st.form_submit_button("Predict Price 💰")

        if submitted:
            # Build single-row DataFrame in same column order as training features
            input_data = {}
            for col in feature_cols:
                if col in cat_inputs:
                    input_data[col] = [cat_inputs[col]]
                elif col in num_inputs:
                    input_data[col] = [num_inputs[col]]
                else:
                    input_data[col] = [df[col].median() if col in df.columns else 0]

            input_df = pd.DataFrame(input_data)

            pred_price = model.predict(input_df)[0]

            st.success(f"🎉 Estimated Nightly Price: **${pred_price:,.2f}**")

            with st.expander("See input used for prediction"):
                st.write(input_df)

        st.markdown('</div>', unsafe_allow_html=True)
