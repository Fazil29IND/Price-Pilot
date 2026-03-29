import streamlit as st
import pandas as pd
import requests
import joblib
import os
import datetime

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

VEHICLE_DATA = {
    "Bike":             {"id": 1},
    "Auto":             {"id": 2},
    "Mini":             {"id": 3},
    "Prime Sedan":      {"id": 0},
    "Premium / Luxury": {"id": 4},
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

@st.cache_resource(show_spinner="Loading ML Model...")
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")

def geocode(address: str):
    params = {"q": address, "format": "json", "limit": 1}
    custom_headers = {"User-Agent": "PricePilot_AmanullahFazil/1.0 (student_project)"}
    
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=custom_headers, timeout=10)
        if resp.status_code != 200:
            st.error(f"API Blocked Request: OpenStreetMap returned Error Code {resp.status_code}")
            return None, None
            
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        else:
            st.warning(f"Couldn't Find {address}. Use Chennai Areas Only!")
            return None, None
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None, None

def get_route(lat1, lon1, lat2, lon2):
    url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
    headers = {"User-Agent": "PricePilot_AmanullahFazil/1.0 (student_project)"}
    try:
        resp = requests.get(url, params={"overview": "false"}, headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            route = data["routes"][0]
            return round(route["distance"] / 1000, 4)
    except Exception:
        pass
    return None

st.title("Price Pilot: ML Fare Predictor")
st.markdown("Enter Your Trip Details")

model = load_model()
if model is None:
    st.error(
        "Error: Model Not Found"
        "Please run `Machine_Learning_Model.py` first to generate the model file."
    )
    st.stop()

st.sidebar.header("Trip Details")
vehicle_choice    = st.sidebar.selectbox("Vehicle Type", options=list(VEHICLE_DATA.keys()))
transport_mode_id = VEHICLE_DATA[vehicle_choice]["id"]

today = datetime.date.today()
now   = datetime.datetime.now()

day_of_week_name = st.sidebar.selectbox(
    "Day of Week",
    options=DAY_NAMES,
    index=0
)
day_of_week = DAY_NAMES.index(day_of_week_name)

pickup_hour = st.sidebar.slider("Pickup Hour", 0, 23, now.hour)
surge_val = st.sidebar.slider("Manual Surge Multiplier", 1.0, 3.0, 1.0, 0.1)

traffic = st.sidebar.select_slider(
    "Traffic Level",
    options=[0, 1, 2],
    value=1,
    format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
)

drivers = st.sidebar.select_slider(
    "Driver Availability",
    options=[0, 1, 2],
    value=1,
    format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
)

is_weekend   = day_of_week in [5, 6]
weekend_mult = 1.25 if is_weekend else 1.0
if is_weekend:
    st.sidebar.info("**Weekend Surge Active (+25%)**")

is_night     = pickup_hour >= 22 or pickup_hour <= 5
night_mult   = 1.25 if is_night else 1.0
if is_night:
    st.sidebar.info("**Night Surge Active (+25%)**")

with st.form("trip_form"):
    col1, col2 = st.columns(2)
    p_addr = col1.text_input("Pickup Area", placeholder="Enter Pickup Area")
    d_addr = col2.text_input("Drop-off Area", placeholder="Enter Drop-off Area")
    submit_btn = st.form_submit_button("Calculate Estimated Fare")

if submit_btn:
    if not p_addr.strip() or not d_addr.strip():
        st.warning("Please enter both Pickup and Drop-off locations.")
    else:
        with st.spinner(f"Analyzing route for {vehicle_choice}..."):
            p_lat, p_lon = geocode(p_addr)
            d_lat, d_lon = geocode(d_addr)

            if p_lat is not None and d_lat is not None:
                dist_km = get_route(p_lat, p_lon, d_lat, d_lon)

                if not dist_km:
                    st.error("Could not calculate a driving route. Please check the addresses.")
                else:
                    neutral_features = pd.DataFrame([{
                        "distance_km":         dist_km,
                        "transport_mode":      transport_mode_id,
                        "driver_availability": 1,
                        "week":                day_of_week,
                        "surge_multiplier":    surge_val,
                        "traffic":             1,
                        "pickup_hour":         pickup_hour,
                    }])

                    try:
                        raw_prediction = model.predict(neutral_features)[0]

                        traffic_map = {0: 0.75, 1: 1.00, 2: 1.35}
                        driver_map  = {0: 1.20, 1: 1.00, 2: 0.88}

                        corrected_base = raw_prediction \
                            * traffic_map.get(traffic, 1.0) \
                            * driver_map.get(drivers, 1.0)

                        final_fare = corrected_base * weekend_mult * night_mult

                        st.divider()
                        st.subheader("Fare Breakdown")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("ML Base Fare",  f"₹{raw_prediction:,.2f}")
                        col_b.metric("Weekend Surge", f"{weekend_mult:.2f}×" if is_weekend else "None")
                        col_c.metric("Final Fare",    f"₹{final_fare:,.2f}")
                        col_d.metric("Distance",      f"{dist_km:,.2f} km")

                        st.success(f"### Estimated Fare: ₹{final_fare:,.2f}")

                        with st.expander("See price adjustments"):
                            st.write(f"- **Traffic multiplier** ({['Low','Medium','High'][traffic]}): `×{traffic_map[traffic]}`")
                            st.write(f"- **Driver availability** ({['Low','Medium','High'][drivers]}): `×{driver_map[drivers]}`")
                            if is_weekend:
                                st.write(f"- **Weekend surge**: `×{weekend_mult}`")
                            if is_night:
                                st.write(f"- **Night surge (10PM–5AM)**: `×{night_mult}`")

                        st.divider()
                        st.subheader("How Your Fare Was Calculated")

                        after_traffic  = raw_prediction * traffic_map.get(traffic, 1.0)
                        after_driver   = after_traffic  * driver_map.get(drivers, 1.0)
                        after_weekend  = after_driver   * weekend_mult
                        after_night    = after_weekend  * night_mult 

                        calc_rows = [
                            {
                                "Step": "ML Base Fare",
                                "Multiplier": "—",
                                "Fare After Step (₹)": f"₹{raw_prediction:,.2f}",
                                "Status": "Applied"
                            },
                            {
                                "Step": f"Traffic Adjustment ({['Low','Medium','High'][traffic]})",
                                "Multiplier": f"×{traffic_map[traffic]}",
                                "Fare After Step (₹)": f"₹{after_traffic:,.2f}",
                                "Status": "Applied"
                            },
                            {
                                "Step": f"Driver Availability ({['Low','Medium','High'][drivers]})",
                                "Multiplier": f"×{driver_map[drivers]}",
                                "Fare After Step (₹)": f"₹{after_driver:,.2f}",
                                "Status": "Applied"
                            },
                            {
                                "Step": "Weekend Surge",
                                "Multiplier": f"×{weekend_mult}",
                                "Fare After Step (₹)": f"₹{after_weekend:,.2f}",
                                "Status": "Applied" if is_weekend else "Not Active"
                            },
                            {
                                "Step": "Night Surge (10PM–5AM)",
                                "Multiplier": f"×{night_mult}",
                                "Fare After Step (₹)": f"₹{after_night:,.2f}",
                                "Status": "Applied" if is_night else "Not Active"
                            },
                        ]

                        st.dataframe(
                            pd.DataFrame(calc_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown(
                            f"**Formula:** "
                            f"`₹{raw_prediction:,.2f}` (Base) "
                            f"× `{traffic_map[traffic]}` (Traffic) "
                            f"× `{driver_map[drivers]}` (Driver) "
                            f"× `{weekend_mult}` (Weekend) "
                            f"× `{night_mult}` (Night) "
                            f"= **₹{final_fare:,.2f}**"
                        )

                        st.subheader("Route")
                        map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
                        st.map(map_data)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
