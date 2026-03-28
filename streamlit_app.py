# ─────────────────────────────────────────────
# PREDICTION & CORRECTION LOGIC (replace the try block)
# ─────────────────────────────────────────────

try:
    # 1. Get raw prediction from ML model using NEUTRAL values for traffic & drivers
    #    so the model doesn't bake in a wrong learned direction.
    #    We pass traffic=1 (medium) and driver_availability=1 (medium) as neutral baseline.
    neutral_features = pd.DataFrame([{
        "distance_km":         dist_km,
        "transport_mode":      transport_mode_id,
        "driver_availability": 1,          # ← always pass neutral to model
        "week":                day_of_week,
        "surge_multiplier":    surge_val,
        "traffic":             1,          # ← always pass neutral to model
        "pickup_hour":         pickup_hour,
    }])

    raw_prediction = model.predict(neutral_features)[0]

    # 2. Logic Correction Layer (you fully own traffic & driver pricing)
    # Traffic: Low=cheaper, Medium=normal, High=more expensive
    traffic_map = {0: 0.85, 1: 1.00, 2: 1.30}

    # Driver Availability: Low supply=more expensive, High supply=cheaper
    driver_map = {0: 1.20, 1: 1.00, 2: 0.88}

    corrected_base = raw_prediction \
        * traffic_map.get(traffic, 1.0) \
        * driver_map.get(drivers, 1.0)

    # 3. Apply weekend surge
    final_fare = corrected_base * weekend_mult

    # ── Display ──
    st.divider()
    st.subheader("Fare Breakdown")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("ML Base Fare",  f"₹{raw_prediction:,.2f}")
    col_b.metric("Weekend Surge", f"{weekend_mult:.2f}×" if is_weekend else "None")
    col_c.metric("Final Fare",    f"₹{final_fare:,.2f}")

    st.success(f"### Estimated Fare: ₹{final_fare:,.2f}")

    # ── Show adjustment breakdown for transparency ──
    with st.expander("See price adjustments"):
        st.write(f"- **Traffic multiplier** ({['Low','Medium','High'][traffic]}): `×{traffic_map[traffic]}`")
        st.write(f"- **Driver availability** ({['Low','Medium','High'][drivers]}): `×{driver_map[drivers]}`")
        if is_weekend:
            st.write(f"- **Weekend surge**: `×{weekend_mult}`")

    # ── Route map ──
    st.subheader("Route")
    map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
    st.map(map_data)

except Exception as e:
    st.error(f"Prediction Error: {e}")
