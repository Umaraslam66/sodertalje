"""
Merge freight and GTFS passenger data into a clean combined timetable.
This script creates april10_combined_timetable.csv with standardized columns.
"""
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "Data"

# Load station order for reference
station_order = pd.read_csv(ANALYSIS_DIR / "april10_station_order.csv")
valid_stations = set(station_order['station'].tolist())

print(f"Valid stations in route: {len(valid_stations)}")

# --- Load Freight Data ---
print("\n=== Loading Freight Data ===")
freight_file = ANALYSIS_DIR / "april10_traffic.csv"

if freight_file.exists():
    freight_df = pd.read_csv(freight_file)
    print(f"Freight file columns: {list(freight_df.columns)}")
    print(f"Freight rows: {len(freight_df)}")
    
    # Standardize freight columns (using actual column names from the file)
    freight_clean = pd.DataFrame()
    freight_clean['train_id'] = 'FREIGHT_' + freight_df['taglank'].astype(str)
    freight_clean['Station_Full_Name'] = freight_df['plats']  # 'plats' is the station column
    
    # Parse time from plandatumtid (format: YYYY-MM-DD HH:MM:SS)
    # Convert early morning (00-04) to extended format (24-28) for 5am-5am timeline
    if 'plandatumtid' in freight_df.columns:
        times = pd.to_datetime(freight_df['plandatumtid'])
        
        def format_time_extended(dt):
            """Format time as HH:MM:SS, converting 00-04 to 24-28 for overnight."""
            if pd.isna(dt):
                return None
            hour = dt.hour
            minute = dt.minute
            second = dt.second
            
            # Convert early morning (00-04) to extended format (24-28)
            if 0 <= hour < 5:
                hour += 24
            
            return f"{hour:02d}:{minute:02d}:{second:02d}"
        
        freight_clean['Ankomst'] = times.apply(format_time_extended)
        freight_clean['Avgång'] = times.apply(format_time_extended)
    else:
        freight_clean['Ankomst'] = None
        freight_clean['Avgång'] = None
    
    freight_clean['Type'] = 'Freight'
    freight_clean['Operator'] = freight_df.get('avtalspart', 'Green Cargo')
    freight_clean['Line'] = freight_df.get('annonserat_tagnr', '').astype(str)
    freight_clean['Direction'] = freight_df.get('riktningny', '')
    
    # Filter to valid stations only
    freight_clean = freight_clean[freight_clean['Station_Full_Name'].isin(valid_stations)]
    
    # Remove rows without times
    freight_clean = freight_clean[
        freight_clean['Ankomst'].notna() | freight_clean['Avgång'].notna()
    ]
    
    print(f"Freight trains after cleaning: {freight_clean['train_id'].nunique()}")
    print(f"Freight stops: {len(freight_clean)}")
else:
    print("No freight file found")
    freight_clean = pd.DataFrame()

# --- Load GTFS Passenger Data ---
print("\n=== Loading GTFS Passenger Data ===")
passenger_file = ANALYSIS_DIR / "timetable_20240410_passenger.csv"

if passenger_file.exists():
    passenger_df = pd.read_csv(passenger_file)
    print(f"Passenger file columns: {list(passenger_df.columns)}")
    print(f"Passenger rows: {len(passenger_df)}")
    
    # Map operator IDs to names
    operator_mapping = {
        '279': 'Västtrafik',
        '252': 'SJ',
        '313': 'MTR',
        '289': 'Skånetrafiken',
        '253': 'Tågkompaniet',
        '256': 'Norrtåg',
        '267': 'Öresundståg',
        '273': 'Krösatågen',
        '254': 'Jönköpings Länstrafik',
        '629': 'Other',
    }
    
    # Standardize passenger columns
    passenger_clean = pd.DataFrame()
    passenger_clean['train_id'] = passenger_df['train_id']
    passenger_clean['Station_Full_Name'] = passenger_df['Station_Full_Name']
    passenger_clean['Ankomst'] = passenger_df['Ankomst']
    passenger_clean['Avgång'] = passenger_df['Avgång']
    passenger_clean['Type'] = 'Passenger'
    passenger_clean['Operator'] = passenger_df['Operator'].astype(str).map(operator_mapping).fillna(passenger_df['Operator'])
    passenger_clean['Line'] = passenger_df.get('Line', '').astype(str)
    passenger_clean['Direction'] = ''  # Will be calculated from station order
    
    # Convert early morning times (00:00-04:59) to extended format (24:00-28:59)
    # This ensures overnight trains display correctly on the 5am-5am timeline
    def convert_to_extended_time(time_str):
        """Convert 00:00-04:59 to 24:00-28:59 for overnight trains."""
        if pd.isna(time_str):
            return time_str
        parts = str(time_str).split(':')
        hour = int(parts[0])
        if 0 <= hour < 5:
            # Early morning time - add 24 hours
            new_hour = hour + 24
            return f"{new_hour}:{parts[1]}:{parts[2]}" if len(parts) > 2 else f"{new_hour}:{parts[1]}:00"
        return time_str
    
    passenger_clean['Ankomst'] = passenger_clean['Ankomst'].apply(convert_to_extended_time)
    passenger_clean['Avgång'] = passenger_clean['Avgång'].apply(convert_to_extended_time)
    
    # Filter to valid stations only
    passenger_clean = passenger_clean[passenger_clean['Station_Full_Name'].isin(valid_stations)]
    
    # Remove rows without times
    passenger_clean = passenger_clean[
        passenger_clean['Ankomst'].notna() | passenger_clean['Avgång'].notna()
    ]
    
    print(f"Passenger trains after cleaning: {passenger_clean['train_id'].nunique()}")
    print(f"Passenger stops: {len(passenger_clean)}")
else:
    print("No passenger file found")
    passenger_clean = pd.DataFrame()

# --- Combine Data ---
print("\n=== Combining Data ===")
combined = pd.concat([freight_clean, passenger_clean], ignore_index=True)

print(f"Total combined trains: {combined['train_id'].nunique()}")
print(f"Total combined stops: {len(combined)}")
print(f"\nBreakdown by type:")
print(combined['Type'].value_counts())
print(f"\nTop operators:")
print(combined['Operator'].value_counts().head(10))

# Save combined data
output_file = ANALYSIS_DIR / "april10_combined_timetable.csv"
combined.to_csv(output_file, index=False)
print(f"\n✅ Saved combined timetable to: {output_file}")
print(f"   {len(combined)} rows, {combined['train_id'].nunique()} unique trains")
