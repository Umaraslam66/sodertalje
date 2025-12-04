"""
Script: 01_extract_routes.py
Purpose: Extract all train journeys between Skandiahamnen and Södertälje Hamn
Author: AI Agent
Data sources: Data/Delay/*.parquet, Data/Cancelled/*.parquet
Date: 2024-12
Version: v1.0

This script:
1. Loads all delay data (12 months of 2024)
2. Identifies journeys that pass through both Skandiahamnen and Södertälje hamn
3. Determines direction (TO or FROM Södertälje)
4. Also includes cancelled trains for complete picture
5. Outputs route summaries to analysis folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent  # Final folder
DELAY_PATH = BASE_PATH / "Data" / "Delay"
CANCELLED_PATH = BASE_PATH / "Data" / "Cancelled"
OUTPUT_PATH = Path(__file__).parent.parent / "analysis"

# Station name patterns to search for
SKANDIA_PATTERNS = ['skandiahamnen', 'skandia', 'gbg skandia']
SODERTALJE_PATTERNS = ['södertälje hamn', 'sodertalje hamn', 'sth', 'södertälje h']


def normalize_station(name):
    """Normalize station name for matching."""
    if pd.isna(name):
        return ""
    return str(name).lower().strip()


def is_skandiahamnen(station):
    """Check if station is Skandiahamnen."""
    normalized = normalize_station(station)
    return any(pattern in normalized for pattern in SKANDIA_PATTERNS)


def is_sodertalje_hamn(station):
    """Check if station is Södertälje Hamn."""
    normalized = normalize_station(station)
    return any(pattern in normalized for pattern in SODERTALJE_PATTERNS)


def load_all_delay_data():
    """Load all delay data parquet files."""
    print("Loading delay data...")
    delay_files = list(DELAY_PATH.glob("*.parquet"))
    print(f"  Found {len(delay_files)} delay files")
    
    dfs = []
    for f in delay_files:
        df = pd.read_parquet(f)
        # Extract month from filename
        month = f.stem.split('_')[-2]  # e.g., '202401'
        df['source_month'] = month
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total delay records: {len(combined):,}")
    return combined


def load_all_cancelled_data():
    """Load all cancelled data parquet files."""
    print("\nLoading cancelled data...")
    cancelled_files = list(CANCELLED_PATH.glob("*.parquet"))
    print(f"  Found {len(cancelled_files)} cancelled files")
    
    dfs = []
    for f in cancelled_files:
        df = pd.read_parquet(f)
        month = f.stem.split('_')[-1]  # e.g., '202401'
        df['source_month'] = month
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total cancelled records: {len(combined):,}")
    return combined


def extract_route_journeys(df):
    """
    Extract journeys that pass through both Skandiahamnen and Södertälje Hamn.
    Returns DataFrame with journey info and direction.
    """
    print("\nExtracting route journeys...")
    
    # Get unique station names to check patterns
    stations = df['plats'].dropna().unique()
    skandia_matches = [s for s in stations if is_skandiahamnen(s)]
    sodertalje_matches = [s for s in stations if is_sodertalje_hamn(s)]
    
    print(f"  Skandiahamnen matches found: {skandia_matches}")
    print(f"  Södertälje Hamn matches found: {sodertalje_matches}")
    
    # Ensure plandatumtid is datetime
    if df['plandatumtid'].dtype == 'object':
        df['plandatumtid'] = pd.to_datetime(df['plandatumtid'])
    
    # Group by taglank and get station sequence
    journeys = []
    grouped = df.groupby('taglank')
    
    total_journeys = len(grouped)
    print(f"  Total unique journeys (taglank): {total_journeys:,}")
    
    for taglank, group in grouped:
        # Sort by planned time
        group = group.sort_values('plandatumtid')
        stations_list = group['plats'].tolist()
        
        # Check if both stations are in the journey
        has_skandia = any(is_skandiahamnen(s) for s in stations_list)
        has_sodertalje = any(is_sodertalje_hamn(s) for s in stations_list)
        
        if has_skandia and has_sodertalje:
            # Find indices
            skandia_idx = next(i for i, s in enumerate(stations_list) if is_skandiahamnen(s))
            sodertalje_idx = next(i for i, s in enumerate(stations_list) if is_sodertalje_hamn(s))
            
            # Determine direction
            if skandia_idx < sodertalje_idx:
                direction = "TO_SODERTALJE"
            else:
                direction = "FROM_SODERTALJE"
            
            # Get journey info
            first_row = group.iloc[0]
            last_row = group.iloc[-1]
            
            # Get Skandiahamnen and Södertälje times
            skandia_row = group[group['plats'].apply(is_skandiahamnen)].iloc[0]
            sodertalje_row = group[group['plats'].apply(is_sodertalje_hamn)].iloc[0]
            
            journeys.append({
                'taglank': taglank,
                'tagnr': first_row.get('tagnr', None),
                'direction': direction,
                'operator': first_row.get('avtalspart', None),
                'origin': first_row.get('plats', None),
                'destination': last_row.get('plats', None),
                'skandia_time': skandia_row['plandatumtid'],
                'sodertalje_time': sodertalje_row['plandatumtid'],
                'num_stations': len(stations_list),
                'stations_list': '|'.join(stations_list),
                'journey_date': first_row['plandatumtid'].date() if pd.notna(first_row['plandatumtid']) else None,
                'source_month': first_row.get('source_month', None)
            })
    
    result_df = pd.DataFrame(journeys)
    print(f"  Found {len(result_df)} journeys on Skandiahamnen ↔ Södertälje route")
    
    return result_df


def extract_cancelled_journeys(df):
    """
    Extract cancelled journeys between Skandiahamnen and Södertälje Hamn.
    """
    print("\nExtracting cancelled route journeys...")
    
    # Check column names (may differ from delay data)
    print(f"  Columns: {list(df.columns)}")
    
    # Filter to cancelled only
    if 'Inställtflagga' in df.columns:
        df = df[df['Inställtflagga'] == 'J'].copy()
    elif 'inställtflagga' in df.columns:
        df = df[df['inställtflagga'] == 'J'].copy()
    
    print(f"  Cancelled records: {len(df):,}")
    
    # Use appropriate column names
    station_col = 'Platssign' if 'Platssign' in df.columns else 'platssign'
    taglank_col = 'Tåglänk' if 'Tåglänk' in df.columns else 'taglank'
    tagnr_col = 'Tågnr' if 'Tågnr' in df.columns else 'tagnr'
    
    # Get unique stations
    stations = df[station_col].dropna().unique()
    skandia_matches = [s for s in stations if is_skandiahamnen(s)]
    sodertalje_matches = [s for s in stations if is_sodertalje_hamn(s)]
    
    print(f"  Skandiahamnen matches in cancelled: {skandia_matches}")
    print(f"  Södertälje Hamn matches in cancelled: {sodertalje_matches}")
    
    # Group by taglank
    cancelled_journeys = []
    grouped = df.groupby(taglank_col)
    
    for taglank, group in grouped:
        stations_list = group[station_col].tolist()
        
        has_skandia = any(is_skandiahamnen(s) for s in stations_list)
        has_sodertalje = any(is_sodertalje_hamn(s) for s in stations_list)
        
        if has_skandia and has_sodertalje:
            skandia_idx = next((i for i, s in enumerate(stations_list) if is_skandiahamnen(s)), -1)
            sodertalje_idx = next((i for i, s in enumerate(stations_list) if is_sodertalje_hamn(s)), -1)
            
            if skandia_idx >= 0 and sodertalje_idx >= 0:
                direction = "TO_SODERTALJE" if skandia_idx < sodertalje_idx else "FROM_SODERTALJE"
                
                first_row = group.iloc[0]
                cancelled_journeys.append({
                    'taglank': taglank,
                    'tagnr': first_row.get(tagnr_col, None),
                    'direction': direction,
                    'is_cancelled': True,
                    'num_stations': len(stations_list),
                    'source_month': first_row.get('source_month', None)
                })
    
    result_df = pd.DataFrame(cancelled_journeys)
    print(f"  Found {len(result_df)} cancelled journeys on route")
    
    return result_df


def analyze_and_save(journeys_df, cancelled_df):
    """Analyze journeys and save results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Direction breakdown
    print("\n--- Journeys by Direction ---")
    direction_counts = journeys_df['direction'].value_counts()
    print(direction_counts)
    
    # Operator breakdown
    print("\n--- Journeys by Operator ---")
    operator_counts = journeys_df['operator'].value_counts()
    print(operator_counts)
    
    # Monthly breakdown
    print("\n--- Journeys by Month ---")
    monthly_counts = journeys_df['source_month'].value_counts().sort_index()
    print(monthly_counts)
    
    # Journey times
    journeys_df['journey_duration'] = (
        journeys_df['sodertalje_time'] - journeys_df['skandia_time']
    ).abs()
    
    print("\n--- Journey Duration Statistics ---")
    duration_hours = journeys_df['journey_duration'].dt.total_seconds() / 3600
    print(f"  Mean: {duration_hours.mean():.2f} hours")
    print(f"  Median: {duration_hours.median():.2f} hours")
    print(f"  Min: {duration_hours.min():.2f} hours")
    print(f"  Max: {duration_hours.max():.2f} hours")
    
    # Save journeys
    output_file = OUTPUT_PATH / "routes_all_journeys.csv"
    journeys_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved all journeys to: {output_file}")
    
    # Save TO Södertälje journeys
    to_sodertalje = journeys_df[journeys_df['direction'] == 'TO_SODERTALJE']
    output_file = OUTPUT_PATH / "routes_to_sodertalje.csv"
    to_sodertalje.to_csv(output_file, index=False)
    print(f"✓ Saved TO Södertälje journeys ({len(to_sodertalje)}): {output_file}")
    
    # Save FROM Södertälje journeys
    from_sodertalje = journeys_df[journeys_df['direction'] == 'FROM_SODERTALJE']
    output_file = OUTPUT_PATH / "routes_from_sodertalje.csv"
    from_sodertalje.to_csv(output_file, index=False)
    print(f"✓ Saved FROM Södertälje journeys ({len(from_sodertalje)}): {output_file}")
    
    # Save summary statistics
    summary = {
        'total_journeys': len(journeys_df),
        'to_sodertalje_count': len(to_sodertalje),
        'from_sodertalje_count': len(from_sodertalje),
        'cancelled_count': len(cancelled_df) if cancelled_df is not None else 0,
        'unique_operators': journeys_df['operator'].nunique(),
        'operators': list(journeys_df['operator'].unique()),
        'mean_duration_hours': duration_hours.mean(),
        'median_duration_hours': duration_hours.median(),
        'date_range': f"{journeys_df['journey_date'].min()} to {journeys_df['journey_date'].max()}"
    }
    
    summary_df = pd.DataFrame([summary])
    output_file = OUTPUT_PATH / "routes_summary.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"✓ Saved summary statistics: {output_file}")
    
    # Save cancelled journeys if any
    if cancelled_df is not None and len(cancelled_df) > 0:
        output_file = OUTPUT_PATH / "routes_cancelled.csv"
        cancelled_df.to_csv(output_file, index=False)
        print(f"✓ Saved cancelled journeys ({len(cancelled_df)}): {output_file}")
    
    return summary


def main():
    print("="*60)
    print("PHASE 1: Route Discovery - Skandiahamnen ↔ Södertälje Hamn")
    print("="*60)
    
    # Load delay data
    delay_df = load_all_delay_data()
    
    # Extract route journeys
    journeys_df = extract_route_journeys(delay_df)
    
    # Load and process cancelled data
    try:
        cancelled_df = load_all_cancelled_data()
        cancelled_journeys = extract_cancelled_journeys(cancelled_df)
    except Exception as e:
        print(f"\nWarning: Could not process cancelled data: {e}")
        cancelled_journeys = pd.DataFrame()
    
    # Analyze and save
    if len(journeys_df) > 0:
        summary = analyze_and_save(journeys_df, cancelled_journeys)
        
        print("\n" + "="*60)
        print("PHASE 1 COMPLETE")
        print("="*60)
        print(f"\nKey findings:")
        print(f"  • Total journeys: {summary['total_journeys']}")
        print(f"  • TO Södertälje: {summary['to_sodertalje_count']}")
        print(f"  • FROM Södertälje: {summary['from_sodertalje_count']}")
        print(f"  • Cancelled: {summary['cancelled_count']}")
        print(f"  • Operators: {', '.join(str(o) for o in summary['operators'])}")
        print(f"  • Avg journey time: {summary['mean_duration_hours']:.1f} hours")
    else:
        print("\n⚠ No journeys found on this route!")
        print("  Checking available station names...")
        stations = delay_df['plats'].dropna().unique()
        print(f"  Total unique stations: {len(stations)}")
        # Show stations containing key words
        for s in sorted(stations):
            s_lower = str(s).lower()
            if any(x in s_lower for x in ['skandia', 'hamn', 'söder', 'soder', 'göteborg', 'goteborg']):
                print(f"    - {s}")


if __name__ == "__main__":
    main()
