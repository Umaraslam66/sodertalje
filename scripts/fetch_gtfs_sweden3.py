"""
Fetch GTFS Sweden 3 static feed from Trafiklab and build a route-filtered timetable
for 2024-04-10 covering the Södertälje ↔ Göteborg corridor.

Usage (PowerShell):
    $env:TRAFIKLAB_API_KEY="YOUR_KEY"
    python fetch_gtfs_sweden3.py

What it does:
- Downloads https://opendata.samtrafiken.se/gtfs-sweden/sweden.zip?key={apikey}
  into Data/GTFS/sweden.zip (reuse unless FORCE_DOWNLOAD=1).
- Extracts stops, trips, stop_times, routes, calendar, calendar_dates.
- Filters trips that operate on 2024-04-10.
- Keeps stop_times whose stop_name matches our canonical route stations
  (from analysis/sodertalje_capacity/analysis/april10_station_order.csv),
  using case-insensitive exact name matching.
- Writes combined timetable to analysis/sodertalje_capacity/analysis/timetable_20240410_sweden3.csv.

Notes:
- No stop name fuzzy matching yet; exact name (case-insensitive) only.
- Swedish characters are preserved.
"""

from __future__ import annotations
import os
import zipfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import certifi

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis" / "sodertalje_capacity" / "analysis"
GTFS_DIR = BASE_DIR / "Data" / "GTFS"
OUTPUT_CSV = ANALYSIS_DIR / "timetable_20240410_passenger.csv"

# Historical GTFS archive for April 10, 2024
GTFS_URL = "https://data.samtrafiken.se/trafiklab/gtfs-sverige-2/2024/04/sweden-20240410.zip"
TARGET_DATE = "2024-04-10"


def fetch_zip(force: bool = False) -> Path:
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = GTFS_DIR / "sweden-20240410.zip"
    if zip_path.exists() and not force:
        print(f"Using cached {zip_path}")
        return zip_path
    print(f"Downloading GTFS for {TARGET_DATE}...")
    resp = requests.get(GTFS_URL, timeout=180, verify=certifi.where())
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    print(f"Downloaded {len(resp.content) / 1024 / 1024:.1f} MB")
    return zip_path


def date_in_service(service_id: str, calendar: pd.DataFrame, cal_dates: pd.DataFrame, target: datetime) -> bool:
    target_int = int(target.strftime("%Y%m%d"))
    # calendar_dates overrides
    overrides = cal_dates[cal_dates["service_id"] == service_id]
    if not overrides.empty:
        hit = overrides[overrides["date"] == target_int]
        if not hit.empty:
            return hit.iloc[0]["exception_type"] == 1  # 1 added, 2 removed
    # calendar
    cal_row = calendar[calendar["service_id"] == service_id]
    if cal_row.empty:
        return False
    row = cal_row.iloc[0]
    weekday_flag = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][target.weekday()]
    if row.get(weekday_flag, 0) != 1:
        return False
    start = datetime.strptime(str(row["start_date"]), "%Y%m%d").date()
    end = datetime.strptime(str(row["end_date"]), "%Y%m%d").date()
    return start <= target.date() <= end


def load_gtfs_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path, "r") as zf:
        def read_csv(name):
            with zf.open(name) as f:
                return pd.read_csv(f)
        data = {
            "stops": read_csv("stops.txt"),
            "trips": read_csv("trips.txt"),
            "stop_times": read_csv("stop_times.txt"),
            "routes": read_csv("routes.txt"),
        }
        try:
            data["calendar"] = read_csv("calendar.txt")
        except KeyError:
            data["calendar"] = pd.DataFrame(columns=["service_id"])
        try:
            data["calendar_dates"] = read_csv("calendar_dates.txt")
        except KeyError:
            data["calendar_dates"] = pd.DataFrame(columns=["service_id", "date", "exception_type"])
    return data


def filter_trips_for_date(data: dict, target_date: datetime) -> pd.DataFrame:
    trips = data["trips"]
    cal = data["calendar"]
    cal_dates = data["calendar_dates"]
    keep_ids = []
    for sid in trips["service_id"].unique():
        if date_in_service(sid, cal, cal_dates, target_date):
            keep_ids.append(sid)
    return trips[trips["service_id"].isin(keep_ids)].copy()


def build_timetable(data: dict, gtfs_to_canonical: dict, target_date: datetime) -> pd.DataFrame:
    """Build timetable for all passenger trips on April 10-11 (5am to 5am window) at our mapped stations."""
    # Get trips for April 10 AND April 11 (since GTFS uses service dates)
    next_date = datetime(2024, 4, 11)
    
    trips_april10 = filter_trips_for_date(data, target_date)
    trips_april11 = filter_trips_for_date(data, next_date)
    
    # Combine trips from both days
    trips = pd.concat([trips_april10, trips_april11], ignore_index=True).drop_duplicates(subset=['trip_id'])
    
    if trips.empty:
        print("No trips found for target dates")
        return pd.DataFrame()
    
    print(f"Found {len(trips)} trips operating on {target_date.strftime('%Y-%m-%d')} and {next_date.strftime('%Y-%m-%d')}")
    
    stop_times = data["stop_times"]
    stops = data["stops"]
    routes = data["routes"]

    # Join all data
    st = stop_times.merge(trips[["trip_id", "service_id", "route_id", "trip_headsign"]], on="trip_id", how="inner")
    st = st.merge(routes[["route_id", "route_short_name", "route_long_name", "agency_id"]], on="route_id", how="left")
    st = st.merge(stops[["stop_id", "stop_name"]], on="stop_id", how="left")

    print(f"Total stop_times for these trips: {len(st)}")
    
    # Filter to only our mapped stations (GTFS names)
    gtfs_station_names = set(gtfs_to_canonical.keys())
    st = st[st["stop_name"].isin(gtfs_station_names)].copy()
    
    if st.empty:
        print("No stops match our station list")
        return pd.DataFrame()
    
    print(f"Stop_times at our {len(gtfs_station_names)} mapped stations: {len(st)}")
    
    # Filter to 5am-5am window (05:00:00 to 29:00:00 in GTFS format)
    # GTFS times can exceed 24:00 for trips that continue past midnight
    def time_to_minutes(time_str):
        """Convert HH:MM:SS to minutes from midnight."""
        if pd.isna(time_str):
            return None
        parts = str(time_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    st['arr_minutes'] = st['arrival_time'].apply(time_to_minutes)
    st['dep_minutes'] = st['departure_time'].apply(time_to_minutes)
    
    # Keep stops where either arrival or departure is in [05:00, 29:00) window
    # 05:00 = 300 minutes, 29:00 = 1740 minutes
    st = st[
        ((st['arr_minutes'] >= 300) & (st['arr_minutes'] < 1740)) |
        ((st['dep_minutes'] >= 300) & (st['dep_minutes'] < 1740))
    ].copy()
    
    if st.empty:
        print("No stops in the 5am-5am time window")
        return pd.DataFrame()
    
    print(f"Stop_times in 5am-5am window: {len(st)}")
    
    # Map GTFS stop names to canonical names
    st["Station_Full_Name"] = st["stop_name"].map(gtfs_to_canonical)

    # Build a train id (include route for uniqueness)
    st["train_id"] = st.apply(lambda r: f"{str(r['route_short_name']).strip()}_{r['trip_id']}", axis=1)

    st.rename(columns={
        "arrival_time": "Ankomst",
        "departure_time": "Avgång",
        "route_short_name": "Line",
        "agency_id": "Operator",
    }, inplace=True)

    # Select and reorder columns
    st = st[[
        "train_id", "Line", "route_long_name", "trip_headsign", "Operator",
        "Station_Full_Name", "Ankomst", "Avgång", "stop_sequence", "stop_name"
    ]].copy()
    
    st.rename(columns={
        "route_long_name": "Route_Name",
        "trip_headsign": "Destination",
        "stop_name": "GTFS_Station_Name"
    }, inplace=True)

    return st


def build_station_mapping(canonical_stations: list) -> dict:
    """Map GTFS station names to our canonical names.
    Returns dict: {gtfs_name: canonical_name}
    """
    # Manual mapping based on what we found in GTFS
    mapping = {
        # Exact matches
        "Pölsebo": "Pölsebo",
        "Aspen": "Aspen",
        "Aspedalen": "Aspedalen",
        "Lerum": "Lerum",
        "Stenkullen": "Stenkullen",
        "Floda": "Floda",
        "Alingsås": "Alingsås",
        "Vårgårda": "Vårgårda",
        "Herrljunga central": "Herrljunga central",
        "Floby": "Floby",
        "Stenstorp": "Stenstorp",
        "Regumatorp": "Regumatorp",
        "Väring": "Väring",
        "Moholm": "Moholm",
        "Töreboda": "Töreboda",
        "Slätte": "Slätte",
        "Älgarås": "Älgarås",
        "Linddalen": "Linddalen",
        "Östansjö": "Östansjö",
        "Tälle": "Tälle",
        "Högsjö": "Högsjö",
        "Baggetorp": "Baggetorp",
        "Sköldinge": "Sköldinge",
        "Skebokvarn": "Skebokvarn",
        "Stjärnhov": "Stjärnhov",
        "Björnlunda": "Björnlunda",
        "Gnesta": "Gnesta",
        "Mölnbo": "Mölnbo",
        "Järna": "Järna",
        "Bränninge": "Bränninge",
        "Södertälje hamn": "Södertälje hamn",
        
        # With " station" suffix
        "Partille station": "Partille",
        "Norsesund station": "Norsesund västra",  # Using for both Norsesund entries
        "Västra Bodarna station": "Västra Bodarna",
        "Gårdsjö station": "Gårdsjö",
        "Vingåker station": "Vingåker",
        "Flen station": "Flen",
        
        # Base name matches
        "Jonsered": "Jonsered Västra",  # Will catch both Jonsered Västra and östra
        "Herrljunga": "Herrljunga västra",
        "Falköpings central": "Falköpings c",
    }
    
    return mapping


def main():
    force = os.environ.get("FORCE_DOWNLOAD") == "1"
    target_dt = datetime.strptime(TARGET_DATE, "%Y-%m-%d")

    zip_path = fetch_zip(force=force)
    print(f"Using {zip_path}")

    print("Parsing GTFS...")
    data = load_gtfs_zip(zip_path)

    station_order = pd.read_csv(ANALYSIS_DIR / "april10_station_order.csv")
    canonical_stations = station_order["station"].tolist()
    
    # Build mapping from GTFS names to canonical names
    gtfs_to_canonical = build_station_mapping(canonical_stations)
    
    print(f"Mapped {len(gtfs_to_canonical)} GTFS stations to canonical names")
    print("Filtering trips for target date and building timetable...")

    frame = build_timetable(data, gtfs_to_canonical, target_dt)

    if frame.empty:
        raise SystemExit("No timetable rows found for target date and route stations.")

    frame.sort_values(["train_id", "stop_sequence"], inplace=True)
    frame.drop(columns=["stop_sequence"], inplace=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(frame)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
