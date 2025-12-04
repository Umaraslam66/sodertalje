# Södertälje Hamn - Skandiahamnen Capacity & Slot Analysis

## Overview

This analysis project examines railway capacity between **Göteborg Skandiahamnen** and **Södertälje Hamn** to find available timeslots for adding new freight trains via Västra Stambanan.

## Key Findings

### Route Statistics
- **Total Journeys Analyzed**: 364 (2024 data)
- **To Södertälje**: 177 journeys
- **From Södertälje**: 187 journeys
- **Primary Operator**: Green Cargo AB
- **Average Journey Time**: ~5.7 hours

### Available Capacity
- **To Södertälje**: 79 available slots (19.8 hours)
- **From Södertälje**: 70 available slots (17.5 hours)

### Key Stations on Route
The trains pass through approximately 67-72 stations, with key stops at:
- Falköpings c (99.9% stop frequency)
- Hallsbergs pbg (99.6% stop frequency)
- Skebokvarn (68% stop frequency)
- Hallsbergs rangerbangård (67% stop frequency)
- Lerum (55% stop frequency)
- Laxå (54% stop frequency)

## Project Structure

```
analysis/sodertalje_capacity/
├── scripts/
│   ├── 01_extract_routes.py      # Phase 1: Extract route journeys
│   ├── 02_map_stations.py        # Phase 2: Map station sequences
│   ├── 03_build_timetables.py    # Phase 3: Build timetables & analyze spacing
│   ├── 04_find_slots.py          # Phase 4: Find available capacity slots
│   └── streamlit_app.py          # Interactive web application
├── analysis/
│   ├── routes_all_journeys.csv
│   ├── routes_to_sodertalje.csv
│   ├── routes_from_sodertalje.csv
│   ├── routes_summary.csv
│   ├── station_sequences.csv
│   ├── station_details.csv
│   ├── station_statistics.csv
│   ├── route_variants.csv
│   ├── canonical_station_order.csv
│   ├── hourly_traffic_to_sodertalje.csv
│   ├── hourly_traffic_from_sodertalje.csv
│   ├── headway_analysis.csv
│   ├── headway_summary.csv
│   ├── typical_day_timetable.csv
│   ├── key_stations.csv
│   ├── available_slots_to_sodertalje.csv
│   ├── available_slots_from_sodertalje.csv
│   ├── station_travel_times.csv
│   └── slot_summary.csv
└── visuals/
    ├── en/    # English visualizations
    └── sv/    # Swedish visualizations
```

## How to Run

### Prerequisites
- Python 3.11+
- Virtual environment with required packages

### Run Analysis Scripts
```bash
# Phase 1: Extract routes
python analysis/sodertalje_capacity/scripts/01_extract_routes.py

# Phase 2: Map stations
python analysis/sodertalje_capacity/scripts/02_map_stations.py

# Phase 3: Build timetables
python analysis/sodertalje_capacity/scripts/03_build_timetables.py

# Phase 4: Find slots
python analysis/sodertalje_capacity/scripts/04_find_slots.py
```

### Run Streamlit Application
```bash
cd analysis/sodertalje_capacity/scripts
streamlit run streamlit_app.py
```

## Visualizations

### Generated Charts (EN & SV)
1. **Hourly Traffic Heatmaps** - Train traffic by station and hour
2. **Train Timeline** - Daily train schedule visualization
3. **Headway Analysis** - Distribution of time gaps between trains
4. **Trains per Hour** - Overall traffic by time of day
5. **Available Slots** - Capacity slot analysis
6. **Slot Availability Heatmap** - Hourly availability by direction

## Data Sources

- **Delay Data**: `Data/Delay/*.parquet` - Actual train operations in 2024
- **Cancelled Data**: `Data/Cancelled/*.parquet` - Cancelled trains in 2024
- **Capacity Data**: `Data/Capacity/sweco-gt.csv` - Capacity-related information

## Key Questions Answered

1. **How many trains run on this route?**
   - ~1 train per day in each direction (364 total in 2024)

2. **What is the typical journey time?**
   - Mean: 5.7 hours, Median: 6.2 hours

3. **When are the quiet periods?**
   - TO Södertälje: Quiet hours 08:00-22:00
   - FROM Södertälje: Quiet hours 00:00-10:00, 18:00-23:00

4. **How many new trains could be added?**
   - TO Södertälje: 79 potential slots (~5 trains/day)
   - FROM Södertälje: 70 potential slots (~4-5 trains/day)

## Notes

- Analysis is based on planned schedules from 2024 data
- Minimum headway used: 15 minutes
- Does not include passenger train conflicts (not in dataset)
- Actual implementation requires formal path requests through Trafikverket

## Author
AI Agent - December 2024
