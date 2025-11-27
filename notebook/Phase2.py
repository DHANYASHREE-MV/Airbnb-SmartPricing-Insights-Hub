import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------
# MUST BE THE FIRST STREAMLIT COMMAND
# -----------------------------------------------------------
st.set_page_config(
    page_title="Airbnb NYC – Market Explorer",
    layout="wide"
)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/Radha/OneDrive/Pictures/Documents/DATA SCIENCE Airbnb project/data/AB_NYC_2019.csv")  # <-- change if path is different

df = load_data()

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.title("🏙️ Airbnb NYC – Market Explorer Dashboard")
st.write("Explore New York City Airbnb listings using filters, charts and maps.")

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.header("Filters")

# Neighbourhood group filter
neigh_groups = sorted(df["neighbourhood_group"].unique())
selected_ng = st.sidebar.multiselect(
    "Neighbourhood Groups",
    neigh_groups,
    default=neigh_groups
)

# Room type filter
room_types = sorted(df["room_type"].unique())
selected_room = st.sidebar.multiselect(
    "Room Types",
    room_types,
    default=room_types
)

# Price range filter
min_price = int(df["price"].min())
max_price = int(df["price"].max())

selected_price = st.sidebar.slider(
    "Price Range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, 500)
)

# Apply filters
filtered = df[
    (df["neighbourhood_group"].isin(selected_ng)) &
    (df["room_type"].isin(selected_room)) &
    (df["price"] >= selected_price[0]) &
    (df["price"] <= selected_price[1])
]

st.write(f"### 🔎 Total Listings After Filters: {len(filtered)}")

if filtered.empty:
    st.warning("No listings match these filters. Try changing the values.")
    st.stop()

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Listings", f"{len(filtered):,}")
col2.metric("Avg Price", f"${filtered['price'].mean():.0f}")
col3.metric("Median Price", f"${filtered['price'].median():.0f}")
col4.metric("Avg Reviews", f"{filtered['number_of_reviews'].mean():.1f}")

st.markdown("---")

# -----------------------------------------------------------
# TABS FOR VISUALIZATIONS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Distribution",
    "🛏️ Room Type Breakdown",
    "📍 Neighbourhood Insights",
    "🗺️ Map View"
])

# TAB 1 – Price Distribution
with tab1:
    fig_hist = px.histogram(
        filtered,
        x="price",
        nbins=40,
        title="Price Distribution (Filtered Listings)",
        labels={"price": "Price (USD)"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 2 – Room Type Breakdown
with tab2:
    counts = filtered["room_type"].value_counts().reset_index()
    counts.columns = ["room_type", "count"]

    fig_pie = px.pie(
        counts,
        names="room_type",
        values="count",
        title="Room Type Share"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# TAB 3 – Neighbourhood Insights
with tab3:
    avg_price = (
        filtered.groupby("neighbourhood", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
        .head(20)
    )

    fig_bar = px.bar(
        avg_price,
        x="neighbourhood",
        y="price",
        title="Top 20 Neighbourhoods by Avg Price",
        labels={"neighbourhood": "Neighbourhood", "price": "Avg Price"}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 4 – MAP VIEW
with tab4:
    map_df = filtered.copy()

    # Limit data for performance
    if len(map_df) > 8000:
        map_df = map_df.sample(8000, random_state=42)

    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="price",
        size="number_of_reviews",
        hover_data=["neighbourhood", "room_type", "price"],
        color_continuous_scale="Viridis",
        zoom=10,
        height=650,
        title="Listings on Map (Price & Reviews)"
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
