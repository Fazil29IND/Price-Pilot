import streamlit as st
import pandas as pd
import requests
import joblib
import os
import datetime

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(page_title="Price Pilot | ML Taxi Predictor", page_icon="Price Pilot Icon.jpeg", layout="centered")

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
OSRM_URL      = "https://router.project-osrm.org/route/v1/driving"
HEADERS       = {"User-Agent": "FarePredictionApp/1.0"}

# ── 5 vehicle types, mode 5 (SUV/XL) removed.
#    Mapping verified from dataset fare/km hierarchy:
#    Mode 1 → ₹9.4/km  → Bike
#    Mode 2 → ₹16.7/km → Auto
#    Mode 3 → ₹20.4/km → Mini
#    Mode 0 → ₹22.4/km → Prime Sedan
#    Mode 4 → ₹31.6/km → Premium / Luxury
VEHICLE_DATA = {
    "Bike":             {"id": 1},
    "Auto":             {"id": 2},
    "Mini":             {"id": 3},
    "Prime Sedan":      {"id": 0},
    "Premium / Luxury": {"id": 4},
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML Model...")
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")


def geocode(address: str):
    params = {"q": address, "format": "json", "limit": 1}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None, None


def get_route(lat1, lon1, lat2, lon2):
    url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
    try:
        resp = requests.get(url, params={"overview": "false"}, headers=HEADERS, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            route = data["routes"][0]
            return round(route["distance"] / 1000, 4)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("Price Pilot: ML Fare Predictor")
st.markdown("Enter Your Trip Details")

model = load_model()
if model is None:
    st.error(
        "**Model Error:** `model.pkl` not found. "
        "Please run `Machine_Learning_Model.py` first to generate the model file, "
        "then place it in the same folder as this app."
    )
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("Trip Details")

vehicle_choice    = st.sidebar.selectbox("Vehicle Type", options=list(VEHICLE_DATA.keys()))
transport_mode_id = VEHICLE_DATA[vehicle_choice]["id"]

today = datetime.date.today()
now   = datetime.datetime.now()

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    index=today.weekday(),
    format_func=lambda d: DAY_NAMES[d]
)

pickup_hour = st.sidebar.slider("Pickup Hour", 0, 23, now.hour)

surge_val = st.sidebar.slider("Manual Surge Multiplier", 1.0, 3.0, 1.0, 0.1)

traffic = st.sidebar.select_slider(
    "Traffic Level",
    options=[0, 1, 2],
    value=1,
    format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
)

# driver_availability is a category (0=Low, 1=Medium, 2=High) in the dataset.
# The model has only ever seen values 0, 1, 2 — sending any other value
# would push it out of its training distribution.
drivers = st.sidebar.select_slider(
    "Driver Availability",
    options=[0, 1, 2],
    value=1,
    format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
)

is_weekend   = day_of_week in [5, 6]
weekend_mult = 1.25 if is_weekend else 1.0
if is_weekend:
    st.sidebar.info("✨ **Weekend Surge Active (+25%)**")

# ─────────────────────────────────────────────
# MAIN FORM
# ─────────────────────────────────────────────
with st.form("trip_form"):
    col1, col2 = st.columns(2)
    p_addr = col1.text_input("Pickup Address", placeholder="Enter Pickup Address")
    d_addr = col2.text_input("Drop-off Address", placeholder="Enter Drop-off Address")
    submit_btn = st.form_submit_button("Calculate Estimated Fare")

if submit_btn:
    if not p_addr.strip() or not d_addr.strip():
        st.warning("Please enter both Pickup and Drop-off locations.")
    else:
        with st.spinner(f"Analyzing route for {vehicle_choice}..."):
            p_lat, p_lon = geocode(p_addr)
            d_lat, d_lon = geocode(d_addr)

            if p_lat is None or d_lat is None:
                st.error("Address not found. Please be more specific (e.g. 'T. Nagar, Chennai').")
            else:
                dist_km = get_route(p_lat, p_lon, d_lat, d_lon)

                if not dist_km:
                    st.error("Could not calculate a driving route. Please check the addresses.")
                else:
                    input_features = pd.DataFrame([{
                        "distance_km":         dist_km,
                        "transport_mode":      transport_mode_id,
                        "driver_availability": drivers,
                        "week":                day_of_week,
                        "surge_multiplier":    surge_val,
                        "traffic":             traffic,
                        "pickup_hour":         pickup_hour,
                    }])

                    try:
                        ml_base_fare = model.predict(input_features)[0]
                        final_fare   = ml_base_fare * weekend_mult

                        st.divider()

                        # ── FARE BREAKDOWN ──
                        st.subheader("Fare Breakdown")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("ML Base Fare",  f"₹{ml_base_fare:,.2f}")
                        col_b.metric("Weekend Surge", f"{weekend_mult:.2f}×" if is_weekend else "None")
                        col_c.metric("Final Fare",    f"₹{final_fare:,.2f}")

                        st.success(
                            f"### Estimated Fare: ₹{final_fare:,.2f}\n"
                            f"**Vehicle:** {vehicle_choice} &nbsp;|&nbsp; "
                            f"**Day:** {DAY_NAMES[day_of_week]} &nbsp;|&nbsp; "
                            f"**Hour:** {pickup_hour:02d}:00 &nbsp;|&nbsp; "
                            f"**Distance:** {dist_km:.2f} km"
                        )

                        # ── HOW THIS FARE WAS CALCULATED ──
                        with st.expander("ℹ️ How this fare was calculated"):
                            st.markdown(f"""
| Factor | Value | Source |
|---|---|---|
| Distance | {dist_km:.2f} km | OSRM routing API |
| Vehicle Type | {vehicle_choice} (mode {transport_mode_id}) | Your selection |
| Day of Week | {DAY_NAMES[day_of_week]} | Your selection |
| Pickup Hour | {pickup_hour:02d}:00 | Your selection |
| Traffic Level | {["Low","Medium","High"][traffic]} | Your selection |
| Driver Availability | {["Low","Medium","High"][drivers]} | Your selection |
| Manual Surge | {surge_val:.1f}× | Your selection (fed into ML) |
| **ML Base Fare** | **₹{ml_base_fare:,.2f}** | **XGBoost model output** |
| Weekend Surge | {weekend_mult:.2f}× | Applied post-prediction |
| **Final Fare** | **₹{final_fare:,.2f}** | **ML Base × Weekend Surge** |
                            """)

                        # ── ROUTE MAP ──
                        st.subheader("Route")
                        map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
                        st.map(map_data)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")