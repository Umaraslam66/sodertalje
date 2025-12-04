"""
Script: 04_find_slots.py
Purpose: Find available capacity windows for new trains on the Skandiahamnen-Södertälje route
Author: AI Agent
Data sources: Data/Delay/*.parquet, analysis/sodertalje_capacity/analysis/*.csv
Date: 2024-12
Version: v1.0

This script:
1. Defines slot criteria (minimum headway, station capacity)
2. Analyzes existing train schedules to find gaps
3. Identifies available time windows for new train paths
4. Outputs slot analysis with conflict information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DELAY_PATH = BASE_PATH / "Data" / "Delay"
ANALYSIS_PATH = Path(__file__).parent.parent / "analysis"
VISUALS_EN_PATH = Path(__file__).parent.parent / "visuals" / "en"
VISUALS_SV_PATH = Path(__file__).parent.parent / "visuals" / "sv"

# Colorblind-friendly palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Slot finding parameters
DEFAULT_MIN_HEADWAY_MINUTES = 15  # Minimum time between trains at same station
DEFAULT_JOURNEY_TIME_HOURS = 6   # Expected journey time based on Phase 1 analysis


def load_data():
    """Load all required data."""
    print("Loading data...")
    
    # Load journeys
    journeys_df = pd.read_csv(ANALYSIS_PATH / "routes_all_journeys.csv")
    journeys_df['skandia_time'] = pd.to_datetime(journeys_df['skandia_time'])
    journeys_df['sodertalje_time'] = pd.to_datetime(journeys_df['sodertalje_time'])
    print(f"  Loaded {len(journeys_df)} journeys")
    
    # Load station statistics
    station_stats = pd.read_csv(ANALYSIS_PATH / "station_statistics.csv")
    print(f"  Loaded {len(station_stats)} station statistics")
    
    # Load key stations
    key_stations = pd.read_csv(ANALYSIS_PATH / "key_stations.csv")['station'].tolist()
    print(f"  Loaded {len(key_stations)} key stations")
    
    # Load all delay data
    delay_files = list(DELAY_PATH.glob("*.parquet"))
    delay_dfs = []
    for f in delay_files:
        delay_dfs.append(pd.read_parquet(f))
    delay_df = pd.concat(delay_dfs, ignore_index=True)
    delay_df['plandatumtid'] = pd.to_datetime(delay_df['plandatumtid'])
    print(f"  Loaded {len(delay_df):,} delay records")
    
    # Filter to route journeys
    route_taglanks = set(journeys_df['taglank'].unique())
    route_delay_df = delay_df[delay_df['taglank'].isin(route_taglanks)].copy()
    print(f"  Filtered to {len(route_delay_df):,} route records")
    
    return journeys_df, station_stats, key_stations, route_delay_df


def calculate_travel_times(journeys_df, route_delay_df, key_stations):
    """Calculate travel times between key stations."""
    print("\nCalculating travel times between stations...")
    
    travel_times = {}
    
    for direction in ['TO_SODERTALJE', 'FROM_SODERTALJE']:
        dir_journeys = journeys_df[journeys_df['direction'] == direction]['taglank'].unique()
        dir_data = route_delay_df[route_delay_df['taglank'].isin(dir_journeys)]
        
        # Define origin/destination based on direction
        if direction == 'TO_SODERTALJE':
            origin = 'Göteborg Skandiahamnen'
            destination = 'Södertälje hamn'
        else:
            origin = 'Södertälje hamn'
            destination = 'Göteborg Skandiahamnen'
        
        # Calculate times from origin to each key station
        station_times = {}
        
        for taglank in dir_journeys:
            journey_data = dir_data[dir_data['taglank'] == taglank].sort_values('plandatumtid')
            
            if len(journey_data) == 0:
                continue
            
            # Find origin time
            origin_data = journey_data[journey_data['plats'] == origin]
            if len(origin_data) == 0:
                continue
            
            origin_time = origin_data['plandatumtid'].iloc[0]
            
            # Calculate time to each station
            for station in key_stations:
                station_data = journey_data[journey_data['plats'] == station]
                if len(station_data) > 0:
                    station_time = station_data['plandatumtid'].iloc[0]
                    minutes_from_origin = (station_time - origin_time).total_seconds() / 60
                    
                    if station not in station_times:
                        station_times[station] = []
                    station_times[station].append(minutes_from_origin)
        
        # Calculate median travel times
        travel_times[direction] = {}
        for station, times in station_times.items():
            if times:
                travel_times[direction][station] = {
                    'mean_minutes': np.mean(times),
                    'median_minutes': np.median(times),
                    'min_minutes': np.min(times),
                    'max_minutes': np.max(times)
                }
        
        print(f"  {direction}: calculated times for {len(travel_times[direction])} stations")
    
    return travel_times


def get_existing_trains_by_time(route_delay_df, journeys_df, key_stations):
    """Build a lookup of existing trains at each station by time of day."""
    print("\nBuilding existing train schedule lookup...")
    
    # Get all train events at key stations
    schedule = route_delay_df[route_delay_df['plats'].isin(key_stations)].copy()
    
    # Add journey direction
    journey_directions = journeys_df.set_index('taglank')['direction'].to_dict()
    schedule['direction'] = schedule['taglank'].map(journey_directions)
    
    # Extract time components
    schedule['hour'] = schedule['plandatumtid'].dt.hour
    schedule['minute'] = schedule['plandatumtid'].dt.minute
    schedule['time_minutes'] = schedule['hour'] * 60 + schedule['minute']
    schedule['day_of_week'] = schedule['plandatumtid'].dt.dayofweek
    schedule['is_weekday'] = schedule['day_of_week'] < 5
    
    print(f"  Built schedule with {len(schedule):,} events")
    
    return schedule


def find_available_slots(schedule, travel_times, direction, min_headway=DEFAULT_MIN_HEADWAY_MINUTES,
                        check_interval=15):
    """
    Find available time slots for a new train.
    
    Args:
        schedule: DataFrame with existing train schedule
        travel_times: Dict of travel times to each station
        direction: 'TO_SODERTALJE' or 'FROM_SODERTALJE'
        min_headway: Minimum minutes between trains
        check_interval: Check every N minutes
    
    Returns:
        DataFrame with slot analysis
    """
    print(f"\nFinding available slots for {direction}...")
    print(f"  Minimum headway: {min_headway} minutes")
    print(f"  Checking every {check_interval} minutes")
    
    # Get stations with travel times for this direction
    if direction not in travel_times:
        print(f"  No travel times for {direction}")
        return pd.DataFrame()
    
    stations = list(travel_times[direction].keys())
    print(f"  Checking {len(stations)} stations")
    
    # Filter schedule to this direction
    dir_schedule = schedule[schedule['direction'] == direction].copy()
    
    slots = []
    
    # Check each potential departure time (0-1440 minutes from midnight)
    for dep_minute in range(0, 1440, check_interval):
        dep_hour = dep_minute // 60
        dep_min = dep_minute % 60
        
        conflicts = []
        total_conflict_count = 0
        
        # Check each station
        for station in stations:
            if station not in travel_times[direction]:
                continue
            
            # Expected arrival time at this station
            travel_time = travel_times[direction][station]['median_minutes']
            arr_minute = (dep_minute + travel_time) % 1440
            
            # Check for conflicts (trains within ±headway window)
            station_schedule = dir_schedule[dir_schedule['plats'] == station]
            
            # Find trains within the time window
            window_start = (arr_minute - min_headway) % 1440
            window_end = (arr_minute + min_headway) % 1440
            
            # Handle wrap-around at midnight
            if window_start < window_end:
                conflicting = station_schedule[
                    (station_schedule['time_minutes'] >= window_start) & 
                    (station_schedule['time_minutes'] <= window_end)
                ]
            else:
                conflicting = station_schedule[
                    (station_schedule['time_minutes'] >= window_start) | 
                    (station_schedule['time_minutes'] <= window_end)
                ]
            
            # Count unique trains
            conflict_trains = conflicting['taglank'].nunique()
            
            if conflict_trains > 0:
                total_conflict_count += conflict_trains
                conflicts.append({
                    'station': station,
                    'expected_time': arr_minute,
                    'conflict_count': conflict_trains
                })
        
        # Determine slot status
        if total_conflict_count == 0:
            status = 'AVAILABLE'
        elif total_conflict_count <= 2:
            status = 'LIMITED'
        else:
            status = 'CONGESTED'
        
        slots.append({
            'departure_time': f"{dep_hour:02d}:{dep_min:02d}",
            'departure_minutes': dep_minute,
            'status': status,
            'total_conflicts': total_conflict_count,
            'conflict_stations': len(conflicts),
            'conflict_details': conflicts
        })
    
    slots_df = pd.DataFrame(slots)
    
    # Summary
    status_counts = slots_df['status'].value_counts()
    print(f"\n  Slot Summary:")
    for status, count in status_counts.items():
        print(f"    {status}: {count} slots ({count * check_interval / 60:.1f} hours)")
    
    return slots_df


def find_best_slots(slots_df, direction, n_best=20):
    """Find the best available slots."""
    print(f"\nBest available slots for {direction}:")
    
    # Sort by conflicts (ascending)
    best = slots_df[slots_df['status'] == 'AVAILABLE'].head(n_best)
    
    if len(best) == 0:
        best = slots_df.nsmallest(n_best, 'total_conflicts')
        print("  (No fully clear slots - showing least congested)")
    
    for _, row in best.iterrows():
        print(f"  {row['departure_time']} - {row['status']} ({row['total_conflicts']} conflicts)")
    
    return best


def analyze_daily_patterns(schedule, direction):
    """Analyze daily traffic patterns to identify quiet periods."""
    print(f"\nAnalyzing daily patterns for {direction}...")
    
    dir_schedule = schedule[schedule['direction'] == direction]
    
    # Count trains per hour (weekday average)
    weekday_schedule = dir_schedule[dir_schedule['is_weekday']]
    hourly_counts = weekday_schedule.groupby('hour')['taglank'].nunique() / weekday_schedule['plandatumtid'].dt.date.nunique()
    
    print("\n  Average trains per hour (weekdays):")
    for hour in range(24):
        count = hourly_counts.get(hour, 0)
        bar = '█' * int(count * 5)
        print(f"    {hour:02d}:00 - {count:.2f} {bar}")
    
    # Identify quiet periods (below average)
    avg_hourly = hourly_counts.mean()
    quiet_hours = [h for h in range(24) if hourly_counts.get(h, 0) < avg_hourly * 0.5]
    
    print(f"\n  Average hourly traffic: {avg_hourly:.2f}")
    print(f"  Quiet hours (<50% of average): {', '.join(f'{h:02d}:00' for h in quiet_hours)}")
    
    return hourly_counts, quiet_hours


def save_slot_analysis(slots_to, slots_from, travel_times):
    """Save slot analysis results."""
    print("\nSaving slot analysis...")
    
    # Save slots
    if len(slots_to) > 0:
        # Remove complex conflict_details column for CSV
        slots_to_save = slots_to.drop(columns=['conflict_details'])
        slots_to_save.to_csv(ANALYSIS_PATH / "available_slots_to_sodertalje.csv", index=False)
        print(f"  ✓ Saved available_slots_to_sodertalje.csv")
    
    if len(slots_from) > 0:
        slots_from_save = slots_from.drop(columns=['conflict_details'])
        slots_from_save.to_csv(ANALYSIS_PATH / "available_slots_from_sodertalje.csv", index=False)
        print(f"  ✓ Saved available_slots_from_sodertalje.csv")
    
    # Save travel times
    travel_data = []
    for direction, stations in travel_times.items():
        for station, times in stations.items():
            travel_data.append({
                'direction': direction,
                'station': station,
                'mean_minutes': times['mean_minutes'],
                'median_minutes': times['median_minutes'],
                'min_minutes': times['min_minutes'],
                'max_minutes': times['max_minutes']
            })
    
    travel_df = pd.DataFrame(travel_data)
    travel_df.to_csv(ANALYSIS_PATH / "station_travel_times.csv", index=False)
    print(f"  ✓ Saved station_travel_times.csv")
    
    # Save summary
    summary = {
        'to_sodertalje_available': (slots_to['status'] == 'AVAILABLE').sum() if len(slots_to) > 0 else 0,
        'to_sodertalje_limited': (slots_to['status'] == 'LIMITED').sum() if len(slots_to) > 0 else 0,
        'to_sodertalje_congested': (slots_to['status'] == 'CONGESTED').sum() if len(slots_to) > 0 else 0,
        'from_sodertalje_available': (slots_from['status'] == 'AVAILABLE').sum() if len(slots_from) > 0 else 0,
        'from_sodertalje_limited': (slots_from['status'] == 'LIMITED').sum() if len(slots_from) > 0 else 0,
        'from_sodertalje_congested': (slots_from['status'] == 'CONGESTED').sum() if len(slots_from) > 0 else 0,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(ANALYSIS_PATH / "slot_summary.csv", index=False)
    print(f"  ✓ Saved slot_summary.csv")


def create_slot_visualizations(slots_to, slots_from):
    """Create visualizations of available slots."""
    print("\nCreating slot visualizations...")
    
    for lang, path in [('en', VISUALS_EN_PATH), ('sv', VISUALS_SV_PATH)]:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Color mapping
        status_colors = {
            'AVAILABLE': '#2ca02c',   # Green
            'LIMITED': '#ff7f0e',     # Orange
            'CONGESTED': '#d62728'    # Red
        }
        
        for ax, (direction, slots_df, title) in zip(axes, [
            ('TO_SODERTALJE', slots_to, 'To Södertälje' if lang == 'en' else 'Till Södertälje'),
            ('FROM_SODERTALJE', slots_from, 'From Södertälje' if lang == 'en' else 'Från Södertälje')
        ]):
            if len(slots_df) == 0:
                continue
            
            # Create bar chart
            hours = slots_df['departure_minutes'] / 60
            colors = [status_colors.get(s, 'gray') for s in slots_df['status']]
            
            ax.bar(hours, slots_df['total_conflicts'] + 1, width=0.2, color=colors, alpha=0.7)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            if lang == 'en':
                ax.set_ylabel('Conflict Score')
            else:
                ax.set_ylabel('Konfliktpoäng')
            
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
        
        # X-axis label on bottom plot
        if lang == 'en':
            axes[1].set_xlabel('Departure Time (Hour)')
            fig.suptitle('Available Capacity Slots\nSkandiahamnen ↔ Södertälje Hamn',
                        fontsize=14, fontweight='bold', y=1.02)
        else:
            axes[1].set_xlabel('Avgångstid (timme)')
            fig.suptitle('Tillgängliga kapacitetsslottar\nSkandiahamnen ↔ Södertälje Hamn',
                        fontsize=14, fontweight='bold', y=1.02)
        
        # Legend
        from matplotlib.patches import Patch
        if lang == 'en':
            legend_elements = [
                Patch(facecolor=status_colors['AVAILABLE'], alpha=0.7, label='Available'),
                Patch(facecolor=status_colors['LIMITED'], alpha=0.7, label='Limited'),
                Patch(facecolor=status_colors['CONGESTED'], alpha=0.7, label='Congested')
            ]
        else:
            legend_elements = [
                Patch(facecolor=status_colors['AVAILABLE'], alpha=0.7, label='Tillgänglig'),
                Patch(facecolor=status_colors['LIMITED'], alpha=0.7, label='Begränsad'),
                Patch(facecolor=status_colors['CONGESTED'], alpha=0.7, label='Trångt')
            ]
        
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        filename = f"sodertalje__available_slots__{lang}__v1.png"
        plt.savefig(path / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save description
        desc_file = path / f"sodertalje__available_slots__{lang}__v1.txt"
        with open(desc_file, 'w', encoding='utf-8') as f:
            if lang == 'en':
                f.write(f"Source: Data/Delay/*.parquet, analysis/*.csv\n")
                f.write(f"Columns: plandatumtid, plats, taglank\n")
                f.write(f"Calculation: Check for conflicts at {DEFAULT_MIN_HEADWAY_MINUTES}min headway intervals\n")
                f.write(f"Filter: Journeys on Skandiahamnen-Södertälje route\n")
                f.write(f"Language: en\n")
            else:
                f.write(f"Källa: Data/Delay/*.parquet, analysis/*.csv\n")
                f.write(f"Kolumner: plandatumtid, plats, taglank\n")
                f.write(f"Beräkning: Kontrollera konflikter med {DEFAULT_MIN_HEADWAY_MINUTES}min intervall\n")
                f.write(f"Filter: Resor på Skandiahamnen-Södertälje rutten\n")
                f.write(f"Språk: sv\n")
        
        print(f"  ✓ Saved {filename}")
    
    # Create slot heatmap by hour
    for lang, path in [('en', VISUALS_EN_PATH), ('sv', VISUALS_SV_PATH)]:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Build hourly summary
        hourly_data = []
        for direction, slots_df, label in [
            ('TO', slots_to, 'To Södertälje' if lang == 'en' else 'Till Södertälje'),
            ('FROM', slots_from, 'From Södertälje' if lang == 'en' else 'Från Södertälje')
        ]:
            if len(slots_df) > 0:
                slots_df = slots_df.copy()
                slots_df['hour'] = slots_df['departure_minutes'] // 60
                hourly_summary = slots_df.groupby('hour').apply(
                    lambda x: (x['status'] == 'AVAILABLE').sum()
                )
                for hour in range(24):
                    hourly_data.append({
                        'direction': label,
                        'hour': hour,
                        'available_slots': hourly_summary.get(hour, 0)
                    })
        
        if hourly_data:
            hourly_df = pd.DataFrame(hourly_data)
            pivot = hourly_df.pivot(index='direction', columns='hour', values='available_slots')
            
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4)
            
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h:02d}" for h in range(24)])
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            
            if lang == 'en':
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Direction')
                ax.set_title('Available Slots by Hour\n(Green = More Available, Red = Limited)',
                           fontsize=14, fontweight='bold')
                cbar.set_label('Available Slots per Hour')
            else:
                ax.set_xlabel('Timme på dygnet')
                ax.set_ylabel('Riktning')
                ax.set_title('Tillgängliga slottar per timme\n(Grön = Fler tillgängliga, Röd = Begränsat)',
                           fontsize=14, fontweight='bold')
                cbar.set_label('Tillgängliga slottar per timme')
            
            plt.tight_layout()
            
            filename = f"sodertalje__slot_availability_heatmap__{lang}__v1.png"
            plt.savefig(path / filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save description
            desc_file = path / f"sodertalje__slot_availability_heatmap__{lang}__v1.txt"
            with open(desc_file, 'w', encoding='utf-8') as f:
                if lang == 'en':
                    f.write(f"Source: Data/Delay/*.parquet, analysis/*.csv\n")
                    f.write(f"Columns: plandatumtid, plats, taglank\n")
                    f.write(f"Calculation: Count available slots per hour by direction\n")
                    f.write(f"Filter: Journeys on Skandiahamnen-Södertälje route\n")
                    f.write(f"Language: en\n")
                else:
                    f.write(f"Källa: Data/Delay/*.parquet, analysis/*.csv\n")
                    f.write(f"Kolumner: plandatumtid, plats, taglank\n")
                    f.write(f"Beräkning: Räkna tillgängliga slottar per timme och riktning\n")
                    f.write(f"Filter: Resor på Skandiahamnen-Södertälje rutten\n")
                    f.write(f"Språk: sv\n")
            
            print(f"  ✓ Saved {filename}")


def main():
    print("="*60)
    print("PHASE 4: Slot Finding")
    print("="*60)
    
    # Load data
    journeys_df, station_stats, key_stations, route_delay_df = load_data()
    
    # Calculate travel times
    travel_times = calculate_travel_times(journeys_df, route_delay_df, key_stations)
    
    # Build schedule lookup
    schedule = get_existing_trains_by_time(route_delay_df, journeys_df, key_stations)
    
    # Find available slots for each direction
    slots_to = find_available_slots(schedule, travel_times, 'TO_SODERTALJE')
    slots_from = find_available_slots(schedule, travel_times, 'FROM_SODERTALJE')
    
    # Find best slots
    if len(slots_to) > 0:
        best_to = find_best_slots(slots_to, 'TO_SODERTALJE')
    
    if len(slots_from) > 0:
        best_from = find_best_slots(slots_from, 'FROM_SODERTALJE')
    
    # Analyze daily patterns
    if len(schedule) > 0:
        for direction in ['TO_SODERTALJE', 'FROM_SODERTALJE']:
            hourly_counts, quiet_hours = analyze_daily_patterns(schedule, direction)
    
    # Save results
    save_slot_analysis(slots_to, slots_from, travel_times)
    
    # Create visualizations
    create_slot_visualizations(slots_to, slots_from)
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE")
    print("="*60)
    
    # Final summary
    print("\n" + "="*60)
    print("CAPACITY ANALYSIS SUMMARY")
    print("="*60)
    
    if len(slots_to) > 0:
        to_available = (slots_to['status'] == 'AVAILABLE').sum()
        print(f"\nTO SÖDERTÄLJE:")
        print(f"  Available slots: {to_available} ({to_available * 15 / 60:.1f} hours)")
        print(f"  Best times: {', '.join(slots_to[slots_to['status'] == 'AVAILABLE']['departure_time'].head(5).tolist())}")
    
    if len(slots_from) > 0:
        from_available = (slots_from['status'] == 'AVAILABLE').sum()
        print(f"\nFROM SÖDERTÄLJE:")
        print(f"  Available slots: {from_available} ({from_available * 15 / 60:.1f} hours)")
        print(f"  Best times: {', '.join(slots_from[slots_from['status'] == 'AVAILABLE']['departure_time'].head(5).tolist())}")


if __name__ == "__main__":
    main()
