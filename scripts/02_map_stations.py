"""
Script: 02_map_stations.py
Purpose: Extract station sequences for Skandiahamnen-Södertälje route and identify key stop stations
Author: AI Agent
Data sources: Data/Delay/*.parquet, analysis/sodertalje_capacity/analysis/routes_all_journeys.csv
Date: 2024-12
Version: v1.0

This script:
1. Loads all delay data and filtered journeys from Phase 1
2. Extracts the ordered station sequence for each journey
3. Identifies the most common route variants
4. Identifies stations where trains make planned stops (vs pass-through)
5. Outputs station sequences and stop analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DELAY_PATH = BASE_PATH / "Data" / "Delay"
ANALYSIS_PATH = Path(__file__).parent.parent / "analysis"


def load_route_journeys():
    """Load the filtered journeys from Phase 1."""
    journeys_file = ANALYSIS_PATH / "routes_all_journeys.csv"
    print(f"Loading route journeys from: {journeys_file}")
    df = pd.read_csv(journeys_file)
    print(f"  Loaded {len(df)} journeys")
    return df


def load_all_delay_data():
    """Load all delay data parquet files."""
    print("\nLoading delay data...")
    delay_files = list(DELAY_PATH.glob("*.parquet"))
    
    dfs = []
    for f in delay_files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total delay records: {len(combined):,}")
    return combined


def extract_station_sequences(delay_df, journeys_df):
    """
    Extract the full station sequence for each journey between 
    Skandiahamnen and Södertälje Hamn.
    """
    print("\nExtracting station sequences...")
    
    # Get unique taglank values for our route
    route_taglanks = set(journeys_df['taglank'].unique())
    print(f"  Processing {len(route_taglanks)} journeys")
    
    # Filter delay data to just our journeys
    route_data = delay_df[delay_df['taglank'].isin(route_taglanks)].copy()
    print(f"  Found {len(route_data):,} records for these journeys")
    
    # Ensure datetime
    route_data['plandatumtid'] = pd.to_datetime(route_data['plandatumtid'])
    
    sequences = []
    station_details = []
    
    for taglank in route_taglanks:
        journey_data = route_data[route_data['taglank'] == taglank].sort_values('plandatumtid')
        
        if len(journey_data) == 0:
            continue
        
        # Get direction from our journeys df
        journey_info = journeys_df[journeys_df['taglank'] == taglank].iloc[0]
        direction = journey_info['direction']
        
        # Extract stations in order
        stations = journey_data['plats'].tolist()
        
        # Find Skandiahamnen and Södertälje indices
        skandia_idx = None
        sodertalje_idx = None
        
        for i, s in enumerate(stations):
            s_lower = str(s).lower()
            if 'skandiahamnen' in s_lower and skandia_idx is None:
                skandia_idx = i
            if 'södertälje hamn' in s_lower and sodertalje_idx is None:
                sodertalje_idx = i
        
        if skandia_idx is None or sodertalje_idx is None:
            continue
        
        # Extract just the stations between (and including) our endpoints
        if direction == "TO_SODERTALJE":
            route_stations = stations[skandia_idx:sodertalje_idx+1]
        else:
            route_stations = stations[sodertalje_idx:skandia_idx+1]
        
        # Store sequence
        sequence_str = ' → '.join(route_stations)
        sequences.append({
            'taglank': taglank,
            'direction': direction,
            'station_count': len(route_stations),
            'station_sequence': sequence_str,
            'stations_list': route_stations
        })
        
        # Store individual station details
        for i, station in enumerate(route_stations):
            station_data = journey_data[journey_data['plats'] == station]
            
            # Check for stops (both arrival and departure at same station)
            has_arrival = (station_data['riktningny'] == 'Ankomst').any()
            has_departure = (station_data['riktningny'] == 'Avgång').any()
            is_stop = has_arrival and has_departure
            
            # Calculate dwell time if it's a stop
            dwell_minutes = None
            if is_stop:
                arrivals = station_data[station_data['riktningny'] == 'Ankomst']['plandatumtid']
                departures = station_data[station_data['riktningny'] == 'Avgång']['plandatumtid']
                if len(arrivals) > 0 and len(departures) > 0:
                    arr_time = arrivals.iloc[0]
                    dep_time = departures.iloc[0]
                    dwell_minutes = (dep_time - arr_time).total_seconds() / 60
            
            station_details.append({
                'taglank': taglank,
                'direction': direction,
                'station': station,
                'sequence_position': i,
                'is_stop': is_stop,
                'has_arrival': has_arrival,
                'has_departure': has_departure,
                'dwell_minutes': dwell_minutes
            })
    
    sequences_df = pd.DataFrame(sequences)
    details_df = pd.DataFrame(station_details)
    
    print(f"  Extracted {len(sequences_df)} complete sequences")
    print(f"  Total station-journey records: {len(details_df)}")
    
    return sequences_df, details_df


def analyze_route_variants(sequences_df):
    """Identify the most common route variants."""
    print("\n" + "="*60)
    print("ROUTE VARIANT ANALYSIS")
    print("="*60)
    
    # Count unique sequences
    sequence_counts = sequences_df['station_sequence'].value_counts()
    
    print(f"\nTotal unique route variants: {len(sequence_counts)}")
    print("\nTop 5 most common routes:")
    
    variant_analysis = []
    for i, (seq, count) in enumerate(sequence_counts.head(10).items()):
        stations = seq.split(' → ')
        print(f"\n{i+1}. Count: {count}")
        print(f"   Stations: {len(stations)}")
        print(f"   Route: {seq[:100]}..." if len(seq) > 100 else f"   Route: {seq}")
        
        variant_analysis.append({
            'rank': i+1,
            'count': count,
            'station_count': len(stations),
            'route': seq
        })
    
    return pd.DataFrame(variant_analysis)


def analyze_stop_stations(details_df):
    """Analyze which stations have stops most frequently."""
    print("\n" + "="*60)
    print("STOP STATION ANALYSIS")
    print("="*60)
    
    # Aggregate by station
    station_stats = details_df.groupby('station').agg({
        'taglank': 'count',
        'is_stop': ['sum', 'mean'],
        'dwell_minutes': ['mean', 'median', 'max']
    }).reset_index()
    
    station_stats.columns = ['station', 'total_trains', 'stop_count', 'stop_frequency', 
                             'avg_dwell_min', 'median_dwell_min', 'max_dwell_min']
    
    # Sort by stop frequency
    station_stats = station_stats.sort_values('stop_frequency', ascending=False)
    
    print("\n--- Stations with Most Frequent Stops ---")
    print(station_stats[station_stats['stop_count'] > 0].head(20).to_string())
    
    # Identify key stop stations (>50% of trains stop)
    key_stops = station_stats[station_stats['stop_frequency'] > 0.5].copy()
    print(f"\n\nKey stop stations (>50% of trains stop): {len(key_stops)}")
    
    # Also analyze by direction
    print("\n--- Stop Patterns by Direction ---")
    direction_stats = details_df.groupby(['direction', 'station']).agg({
        'taglank': 'count',
        'is_stop': 'mean',
        'dwell_minutes': 'mean'
    }).reset_index()
    direction_stats.columns = ['direction', 'station', 'train_count', 'stop_freq', 'avg_dwell']
    
    for direction in ['TO_SODERTALJE', 'FROM_SODERTALJE']:
        dir_data = direction_stats[direction_stats['direction'] == direction]
        dir_data = dir_data.sort_values('stop_freq', ascending=False)
        print(f"\n{direction} - Top stops:")
        top_stops = dir_data[dir_data['stop_freq'] > 0.3].head(10)
        print(top_stops[['station', 'train_count', 'stop_freq', 'avg_dwell']].to_string())
    
    return station_stats


def identify_primary_route(sequences_df, details_df):
    """Identify the primary route and its station order."""
    print("\n" + "="*60)
    print("PRIMARY ROUTE IDENTIFICATION")
    print("="*60)
    
    # Find most common sequence for each direction
    for direction in ['TO_SODERTALJE', 'FROM_SODERTALJE']:
        dir_seqs = sequences_df[sequences_df['direction'] == direction]
        most_common = dir_seqs['station_sequence'].value_counts().head(1)
        
        if len(most_common) > 0:
            primary_route = most_common.index[0]
            stations = primary_route.split(' → ')
            
            print(f"\n{direction}:")
            print(f"  Most common route ({most_common.values[0]} trains):")
            for i, station in enumerate(stations):
                print(f"    {i+1}. {station}")
    
    # Build a canonical station order based on average position
    avg_positions = details_df.groupby('station')['sequence_position'].mean().sort_values()
    
    print("\n\nCanonical station order (by average position):")
    for i, (station, pos) in enumerate(avg_positions.items()):
        print(f"  {i+1}. {station} (avg pos: {pos:.1f})")
    
    return avg_positions


def save_results(sequences_df, details_df, variants_df, station_stats, avg_positions):
    """Save all analysis results."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save station sequences
    output_file = ANALYSIS_PATH / "station_sequences.csv"
    sequences_df.to_csv(output_file, index=False)
    print(f"✓ Saved station sequences: {output_file}")
    
    # Save station details (for timetable building)
    output_file = ANALYSIS_PATH / "station_details.csv"
    details_df.to_csv(output_file, index=False)
    print(f"✓ Saved station details: {output_file}")
    
    # Save route variants
    output_file = ANALYSIS_PATH / "route_variants.csv"
    variants_df.to_csv(output_file, index=False)
    print(f"✓ Saved route variants: {output_file}")
    
    # Save station statistics
    output_file = ANALYSIS_PATH / "station_statistics.csv"
    station_stats.to_csv(output_file, index=False)
    print(f"✓ Saved station statistics: {output_file}")
    
    # Save canonical station order
    canonical_order = pd.DataFrame({
        'station': avg_positions.index,
        'average_position': avg_positions.values,
        'order': range(1, len(avg_positions)+1)
    })
    output_file = ANALYSIS_PATH / "canonical_station_order.csv"
    canonical_order.to_csv(output_file, index=False)
    print(f"✓ Saved canonical station order: {output_file}")


def main():
    print("="*60)
    print("PHASE 2: Station Sequence Mapping")
    print("="*60)
    
    # Load data
    journeys_df = load_route_journeys()
    delay_df = load_all_delay_data()
    
    # Extract station sequences
    sequences_df, details_df = extract_station_sequences(delay_df, journeys_df)
    
    # Analyze route variants
    variants_df = analyze_route_variants(sequences_df)
    
    # Analyze stop stations
    station_stats = analyze_stop_stations(details_df)
    
    # Identify primary route
    avg_positions = identify_primary_route(sequences_df, details_df)
    
    # Save results
    save_results(sequences_df, details_df, variants_df, station_stats, avg_positions)
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)
    
    print(f"\nKey findings:")
    print(f"  • Total unique route variants: {len(variants_df)}")
    print(f"  • Total stations on route: {len(station_stats)}")
    key_stops = station_stats[station_stats['stop_frequency'] > 0.5]
    print(f"  • Key stop stations (>50% stop): {len(key_stops)}")


if __name__ == "__main__":
    main()
