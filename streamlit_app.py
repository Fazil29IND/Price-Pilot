import streamlit as st
import pandas as pd
import requests
import joblib
import os
import datetime

st.set_page_config(page_title="Price Pilot | ML Taxi Predictor", page_icon="🚖", layout="centered")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    div[data-testid="InputInstructions"] { display: none; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

VEHICLE_DATA = {
    "Bike": {"id": 1},
    "Auto": {"id": 2},
    "Mini": {"id": 3},
    "Prime Sedan": {"id": 0},
    "Premium / Luxury": {"id": 4},
}

# UPDATED: Matches your dataset (Sunday=0, Monday=1, etc.)
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

@st.cache_resource(show_spinner="Loading ML Model...")
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")

def geocode(address: str):
    params = {"q": address, "format": "json", "limit": 1}
    custom_headers = {"User-Agent": "PricePilot_AmanullahFazil/1.0"}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=custom_headers, timeout=10)
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None, None

def get_route(lat1, lon1, lat2, lon2):
    url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
    try:
        resp = requests.get(url, params={"overview": "false"}, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            return round(data["routes"][0]["distance"] / 1000, 4)
    except Exception:
        pass
    return None

# --- UI START ---
st.title("Price Pilot: ML Fare Predictor")

model = load_model()
if model is None:
    st.error("Error: model.pkl not found. Please train the model first.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Trip Details")
vehicle_choice = st.sidebar.selectbox("Vehicle Type", options=list(VEHICLE_DATA.keys()))
transport_mode_id = VEHICLE_DATA[vehicle_choice]["id"]

# Dynamic Day Selection
now = datetime.datetime.now()
# now.strftime('%w') returns 0 for Sunday, matching your dataset
current_day_idx = int(now.strftime('%w')) 

day_of_week_name = st.sidebar.selectbox("Day of Week", options=DAY_NAMES, index=current_day_idx)
day_of_week = DAY_NAMES.index(day_of_week_name)

pickup_hour = st.sidebar.slider("Pickup Hour", 0, 23, now.hour)
surge_val = st.sidebar.slider("Manual Surge Multiplier", 1.0, 3.0, 1.0, 0.1)

traffic = st.sidebar.select_slider(
    "Traffic Level", options=[0, 1, 2], value=1,
    format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
)

drivers = st.sidebar.select_slider(
    "Driver Availability", options=[0, 1, 2], value=1,
    format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
)

# --- MAIN FORM ---
with st.form("trip_form"):
    col1, col2 = st.columns(2)
    p_addr = col1.text_input("Pickup Area", placeholder="e.g., T. Nagar")
    d_addr = col2.text_input("Drop-off Area", placeholder="e.g., Adyar")
    submit_btn = st.form_submit_button("Calculate Accurate Fare")

if submit_btn:
    if not p_addr.strip() or not d_addr.strip():
        st.warning("Please enter both locations.")
    else:
        with st.spinner("Model calculating..."):
            p_lat, p_lon = geocode(p_addr)
            d_lat, d_lon = geocode(d_addr)

            if p_lat and d_lat:
                dist_km = get_route(p_lat, p_lon, d_lat, d_lon)
                
                if dist_km:
                    # ALL DATA PASSED DIRECTLY TO XGBOOST
                    input_df = pd.DataFrame([{
                        "distance_km": dist_km,
                        "transport_mode": transport_mode_id,
                        "driver_availability": drivers,
                        "week": day_of_week,
                        "surge_multiplier": surge_val,
                        "traffic": traffic,
                        "pickup_hour": pickup_hour
                    }])

                    prediction = model.predict(input_df)[0]

                    st.divider()
                    st.subheader("Fare Breakdown")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Machine Learning Model Fare", f"₹{prediction:,.2f}")
                    c2.metric("Distance", f"{dist_km} km")
                    c3.metric("Vehicle", vehicle_choice)

                    st.success(f"### Accurate Fare: ₹{prediction:,.2f}")
                    st.map(pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]}))
                else:
                    st.error("Route calculation failed.")
            else:
                st.error("Could not locate those areas in Chennai.")
