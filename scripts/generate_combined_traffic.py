"""
Generate combined traffic data for April 10, 2024
Combines freight (GT) data with passenger timetable (RST, TJT) data

Strategy: Only use stations with ACTUAL times from the timetable - no interpolation.
This gives accurate paths even if they have fewer points.
"""
import pandas as pd
from pathlib import Path
import json

# Paths
script_dir = Path(__file__).parent
base_dir = script_dir.parent.parent.parent
analysis_dir = script_dir.parent / 'analysis'

print("Loading data...")

# Load passenger timetable
tt = pd.read_excel(analysis_dir / 'timetable_20240410.xlsx')

# Load station mapping
mapping = pd.read_excel(base_dir / 'Data' / 'Stråk-Mapping.xlsx')

# Load our station order
station_order = pd.read_csv(analysis_dir / 'april10_station_order.csv')

# Load freight traffic
freight = pd.read_csv(analysis_dir / 'april10_traffic.csv')
freight['plandatumtid'] = pd.to_datetime(freight['plandatumtid'])

print(f"Passenger timetable: {len(tt)} records")
print(f"Freight traffic: {len(freight)} records")

# Create station mappings
name_to_abbrev = {str(row['Plats']).lower(): str(row['Platssignatur']).upper() 
                  for _, row in mapping.iterrows()}
abbrev_to_name = {str(row['Platssignatur']).upper(): row['Plats'] 
                  for _, row in mapping.iterrows()}

# Our route stations and their order
our_stations = station_order['station'].tolist()
our_abbrevs = []
station_to_order = {}

for i, station in enumerate(our_stations):
    abbrev = name_to_abbrev.get(station.lower(), station.upper())
    our_abbrevs.append(abbrev)
    station_to_order[station] = i
    station_to_order[abbrev] = i

# Also add abbreviation to station name mapping for our route
abbrev_to_our_station = {}
for station in our_stations:
    abbrev = name_to_abbrev.get(station.lower(), station.upper())
    abbrev_to_our_station[abbrev] = station

print(f"Route stations: {len(our_stations)}")

# Process passenger timetable
tt['Station_Upper'] = tt['Station'].str.upper()
route_passenger = tt[tt['Station_Upper'].isin(our_abbrevs)].copy()

# Filter to trains with at least 2 stations on our route with ACTUAL times
# (we need times to plot them)
route_passenger['has_time'] = route_passenger['Ankomst'].notna() | route_passenger['Avgång'].notna()

# Group by train and count stations with times
train_time_counts = route_passenger[route_passenger['has_time']].groupby('Tåg id').size()
significant_trains = train_time_counts[train_time_counts >= 3].index.tolist()  # At least 3 timed stations

print(f"Passenger trains with 3+ timed stations on our route: {len(significant_trains)}")

# Filter to significant trains only
route_passenger = route_passenger[
    (route_passenger['Tåg id'].isin(significant_trains)) & 
    (route_passenger['has_time'])  # Only keep stations WITH times!
]

print(f"Passenger records with actual times: {len(route_passenger)}")

# Process passenger data into same format as freight
passenger_records = []

for train_id in significant_trains:
    train_data = route_passenger[route_passenger['Tåg id'] == train_id].drop_duplicates(subset=['Station'])
    
    if len(train_data) < 2:
        continue
        
    train_type = train_data['Tågslag'].iloc[0]
    operator = train_data['Operator'].iloc[0] if 'Operator' in train_data.columns else 'Unknown'
    
    # Create records ONLY for stations with actual times
    records_for_train = []
    
    for _, row in train_data.iterrows():
        station = row['Station_Upper']
        arr = row['Ankomst']
        dep = row['Avgång']
        
        # Get the time - prefer departure, then arrival
        time_val = dep if pd.notna(dep) else arr
        
        if pd.isna(time_val):
            continue
            
        # Parse time string to datetime
        if isinstance(time_val, str):
            try:
                h, m = map(int, time_val.split(':'))
                time_val = pd.Timestamp('2024-04-10') + pd.Timedelta(hours=h, minutes=m)
            except:
                continue
        
        station_name = abbrev_to_our_station.get(station, station)
        order = station_to_order.get(station, station_to_order.get(station_name, -1))
        
        if order == -1:
            continue
        
        records_for_train.append({
            'taglank': train_id,
            'train_type': train_type,
            'plats': station_name,
            'plandatumtid': time_val,
            'station_order': order,
            'operator': operator
        })
    
    # Sort by time and check for overnight crossing
    if records_for_train:
        records_for_train.sort(key=lambda x: x['plandatumtid'])
        
        # Handle overnight trains - if times wrap around midnight
        # Check if there's a backwards time jump (indicates day change)
        for i in range(1, len(records_for_train)):
            prev_time = records_for_train[i-1]['plandatumtid']
            curr_time = records_for_train[i]['plandatumtid']
            
            # If current time is earlier than previous, add a day
            if curr_time < prev_time:
                records_for_train[i]['plandatumtid'] = curr_time + pd.Timedelta(days=1)
        
        # Check for unrealistic time gaps (>60 minutes between consecutive points)
        # If found, exclude this train as it likely has a multi-hour layover
        has_large_gap = False
        for i in range(1, len(records_for_train)):
            prev_time = records_for_train[i-1]['plandatumtid']
            curr_time = records_for_train[i]['plandatumtid']
            gap_minutes = (curr_time - prev_time).total_seconds() / 60
            if gap_minutes > 60:  # More than 1 hour gap
                has_large_gap = True
                break
        
        if not has_large_gap:
            passenger_records.extend(records_for_train)

passenger_df = pd.DataFrame(passenger_records)
print(f"Passenger records created: {len(passenger_df)}")

# Add station order to freight data
freight['station_order'] = freight['plats'].map(station_to_order)
freight['train_type'] = 'GT'
freight['operator'] = freight.get('Operatör', 'Unknown')

# Filter to 5am-5am window
base_time = pd.Timestamp('2024-04-10 05:00:00')
end_time = pd.Timestamp('2024-04-11 05:00:00')

freight_filtered = freight[(freight['plandatumtid'] >= base_time) & (freight['plandatumtid'] <= end_time)]
passenger_filtered = passenger_df[(passenger_df['plandatumtid'] >= base_time) & (passenger_df['plandatumtid'] <= end_time)]

print(f"Freight in 5am-5am window: {len(freight_filtered)} records, {freight_filtered['taglank'].nunique()} trains")
print(f"Passenger in 5am-5am window: {len(passenger_filtered)} records, {passenger_filtered['taglank'].nunique()} trains")

# Combine data
combined = pd.concat([
    freight_filtered[['taglank', 'plats', 'plandatumtid', 'train_type']],
    passenger_filtered[['taglank', 'plats', 'plandatumtid', 'train_type']]
], ignore_index=True)

print(f"\nCombined: {len(combined)} records, {combined['taglank'].nunique()} trains")
print("By type:")
print(combined.groupby('train_type')['taglank'].nunique())

# Save combined data
output_path = analysis_dir / 'april10_combined_traffic.csv'
combined.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Save station mapping for the app
station_mapping = {
    'abbrev_to_name': abbrev_to_our_station,
    'name_to_abbrev': {v: k for k, v in abbrev_to_our_station.items()}
}
with open(analysis_dir / 'station_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(station_mapping, f, ensure_ascii=False, indent=2)
print("Saved station_mapping.json")
