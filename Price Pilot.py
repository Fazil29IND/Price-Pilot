"""
Price Pilot: ML Taxi Fare Predictor
====================================
A command-line application that predicts taxi fares using a trained ML model.
Uses OpenStreetMap (Nominatim) for geocoding and OSRM for route distance.
"""

import os
import sys
import datetime
import joblib
import requests
import pandas as pd


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL      = "https://router.project-osrm.org/route/v1/driving"
HTTP_HEADERS  = {"User-Agent": "PricePilot_AmanullahFazil/1.0 (student_project)"}

VEHICLE_OPTIONS = {
    "1": ("Bike",              1),
    "2": ("Auto",              2),
    "3": ("Mini",              3),
    "4": ("Prime Sedan",       0),
    "5": ("Premium / Luxury",  4),
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

TRAFFIC_LABELS  = {0: "Low", 1: "Medium", 2: "High"}
DRIVER_LABELS   = {0: "Low", 1: "Medium", 2: "High"}

TRAFFIC_MULTIPLIER = {0: 0.75, 1: 1.00, 2: 1.35}
DRIVER_MULTIPLIER  = {0: 1.20, 1: 1.00, 2: 0.88}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_model(path: str = "model.pkl"):
    """Load the trained ML model from disk."""
    if not os.path.exists(path):
        print(f"\n[ERROR] Model file '{path}' not found.")
        print("        Run 'Machine_Learning_Model.py' first to generate it.")
        sys.exit(1)
    print("[INFO] Loading ML model...")
    return joblib.load(path)


# ─────────────────────────────────────────────
# Geocoding & Routing
# ─────────────────────────────────────────────

def geocode(address: str) -> tuple[float | None, float | None]:
    """Convert a place name to (latitude, longitude) using Nominatim."""
    params = {"q": address, "format": "json", "limit": 1}
    try:
        resp = requests.get(NOMINATIM_URL, params=params,
                            headers=HTTP_HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"[ERROR] OpenStreetMap returned HTTP {resp.status_code}.")
            return None, None

        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])

        print(f"[WARNING] Location not found: '{address}'. Use Chennai areas only.")
        return None, None

    except Exception as exc:
        print(f"[ERROR] Connection error during geocoding: {exc}")
        return None, None


def get_route_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float | None:
    """Return the driving distance in kilometres between two coordinates."""
    url = f"{OSRM_URL}/{lon1},{lat1};{lon2},{lat2}"
    try:
        resp = requests.get(url, params={"overview": "false"},
                            headers=HTTP_HEADERS, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            return round(data["routes"][0]["distance"] / 1000, 4)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Fare Calculation
# ─────────────────────────────────────────────

def calculate_fare(model, dist_km: float, transport_mode_id: int,
                   day_of_week: int, pickup_hour: int, surge_val: float,
                   traffic: int, drivers: int) -> dict:
    """
    Run the ML model and apply post-processing multipliers.
    Returns a dict with base fare, final fare, and all applied surges.
    """
    is_weekend   = day_of_week in [5, 6]
    is_night     = pickup_hour >= 22 or pickup_hour <= 5
    weekend_mult = 1.25 if is_weekend else 1.0
    night_mult   = 1.25 if is_night  else 1.0

    features = pd.DataFrame([{
        "distance_km":      dist_km,
        "transport_mode":   transport_mode_id,
        "driver_availability": 1,          # neutral baseline for model
        "week":             day_of_week,
        "surge_multiplier": surge_val,
        "traffic":          1,             # neutral baseline for model
        "pickup_hour":      pickup_hour,
    }])

    raw_prediction = model.predict(features)[0]

    corrected_base = (raw_prediction
                      * TRAFFIC_MULTIPLIER.get(traffic, 1.0)
                      * DRIVER_MULTIPLIER.get(drivers, 1.0))

    final_fare = corrected_base * weekend_mult * night_mult

    return {
        "raw_prediction": raw_prediction,
        "final_fare":     final_fare,
        "is_weekend":     is_weekend,
        "is_night":       is_night,
        "weekend_mult":   weekend_mult,
        "night_mult":     night_mult,
    }


# ─────────────────────────────────────────────
# Input Helpers
# ─────────────────────────────────────────────

def prompt_choice(prompt: str, options: dict) -> str:
    """Display a numbered menu and return the user's chosen key."""
    print(prompt)
    for key, label in options.items():
        name = label[0] if isinstance(label, tuple) else label
        print(f"  {key}. {name}")
    while True:
        choice = input("  Your choice: ").strip()
        if choice in options:
            return choice
        print(f"  [!] Invalid choice. Enter one of: {', '.join(options)}")


def prompt_int(prompt: str, lo: int, hi: int, default: int) -> int:
    """Prompt for an integer within [lo, hi], falling back to default."""
    while True:
        raw = input(f"  {prompt} [{lo}-{hi}] (default {default}): ").strip()
        if raw == "":
            return default
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  [!] Enter a number between {lo} and {hi}.")


def prompt_float(prompt: str, lo: float, hi: float,
                 step: float, default: float) -> float:
    """Prompt for a float within [lo, hi]."""
    while True:
        raw = input(f"  {prompt} [{lo}-{hi}] (default {default}): ").strip()
        if raw == "":
            return default
        try:
            val = round(float(raw), 1)
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"  [!] Enter a value between {lo} and {hi}.")


def prompt_address(label: str) -> str:
    """Prompt until a non-empty address is entered."""
    while True:
        addr = input(f"  {label}: ").strip()
        if addr:
            return addr
        print("  [!] This field cannot be empty.")


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────

def separator(char: str = "─", width: int = 52) -> None:
    print(char * width)


def print_header() -> None:
    separator("═")
    print("  🚖  Price Pilot — ML Taxi Fare Predictor")
    separator("═")
    print()


def print_fare_breakdown(result: dict, dist_km: float,
                         vehicle_name: str, pickup_area: str,
                         dropoff_area: str) -> None:
    separator()
    print("  FARE BREAKDOWN")
    separator()
    print(f"  Route          : {pickup_area}  →  {dropoff_area}")
    print(f"  Vehicle        : {vehicle_name}")
    print(f"  Distance       : {dist_km:,.2f} km")
    print(f"  ML Base Fare   : ₹{result['raw_prediction']:,.2f}")

    if result["is_weekend"]:
        print(f"  Weekend Surge  : +25%  (×{result['weekend_mult']})")
    if result["is_night"]:
        print(f"  Night Surge    : +25%  (×{result['night_mult']})")

    separator()
    print(f"  ESTIMATED FARE : ₹{result['final_fare']:,.2f}")
    separator()
    print()


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────

def collect_trip_details() -> dict:
    """Interactively collect all trip parameters from the user."""
    now = datetime.datetime.now()

    # --- Vehicle ---
    print("\n[1/6] Vehicle Type")
    vehicle_key  = prompt_choice("", VEHICLE_OPTIONS)
    vehicle_name, transport_mode_id = VEHICLE_OPTIONS[vehicle_key]

    # --- Day of week ---
    print("\n[2/6] Day of Week")
    day_options = {str(i): (name,) for i, name in enumerate(DAY_NAMES)}
    day_key     = prompt_choice("", day_options)
    day_of_week = int(day_key)

    # --- Pickup hour ---
    print("\n[3/6] Pickup Hour (0 = midnight, 23 = 11 PM)")
    pickup_hour = prompt_int("Enter hour", 0, 23, now.hour)

    # --- Surge multiplier ---
    print("\n[4/6] Manual Surge Multiplier")
    surge_val = prompt_float("Enter multiplier", 1.0, 3.0, 0.1, 1.0)

    # --- Traffic level ---
    print("\n[5/6] Traffic Level")
    traffic_key = prompt_choice("", {"0": ("Low",), "1": ("Medium",), "2": ("High",)})
    traffic     = int(traffic_key)

    # --- Driver availability ---
    print("\n[6/6] Driver Availability")
    driver_key = prompt_choice("", {"0": ("Low",), "1": ("Medium",), "2": ("High",)})
    drivers    = int(driver_key)

    return {
        "vehicle_name":      vehicle_name,
        "transport_mode_id": transport_mode_id,
        "day_of_week":       day_of_week,
        "pickup_hour":       pickup_hour,
        "surge_val":         surge_val,
        "traffic":           traffic,
        "drivers":           drivers,
    }


def collect_addresses() -> tuple[str, str]:
    """Prompt for pickup and drop-off locations."""
    print("\n── Route ──────────────────────────────────────")
    pickup  = prompt_address("Pickup Area  (Chennai)")
    dropoff = prompt_address("Drop-off Area (Chennai)")
    return pickup, dropoff


def run() -> None:
    """Entry point: orchestrates the full prediction flow."""
    print_header()

    model = load_model()

    while True:
        trip    = collect_trip_details()
        pickup_area, dropoff_area = collect_addresses()

        print("\n[INFO] Geocoding addresses...")
        p_lat, p_lon = geocode(pickup_area)
        d_lat, d_lon = geocode(dropoff_area)

        if p_lat is None or d_lat is None:
            print("[ERROR] Could not resolve one or both addresses. Try again.\n")
        else:
            print("[INFO] Calculating route distance...")
            dist_km = get_route_distance(p_lat, p_lon, d_lat, d_lon)

            if dist_km is None:
                print("[ERROR] Could not compute a driving route. Check your addresses.\n")
            else:
                try:
                    result = calculate_fare(
                        model          = model,
                        dist_km        = dist_km,
                        transport_mode_id = trip["transport_mode_id"],
                        day_of_week    = trip["day_of_week"],
                        pickup_hour    = trip["pickup_hour"],
                        surge_val      = trip["surge_val"],
                        traffic        = trip["traffic"],
                        drivers        = trip["drivers"],
                    )
                    print_fare_breakdown(result, dist_km,
                                         trip["vehicle_name"],
                                         pickup_area, dropoff_area)
                except Exception as exc:
                    print(f"[ERROR] Prediction failed: {exc}\n")

        again = input("  Predict another fare? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Thank you for using Price Pilot. Safe travels! 🚖\n")
            break


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run()