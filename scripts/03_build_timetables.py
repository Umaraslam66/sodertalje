"""
Script: 03_build_timetables.py
Purpose: Build hourly traffic matrices and analyze train spacing on the Skandiahamnen-Södertälje route
Author: AI Agent
Data sources: Data/Delay/*.parquet, analysis/sodertalje_capacity/analysis/*.csv
Date: 2024-12
Version: v1.0

This script:
1. Builds hourly traffic matrices (Station × Hour → train count)
2. Analyzes train spacing (headway) at key stations
3. Creates visualizations (heatmaps, timelines)
4. Identifies peak and off-peak periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DELAY_PATH = BASE_PATH / "Data" / "Delay"
ANALYSIS_PATH = Path(__file__).parent.parent / "analysis"
VISUALS_EN_PATH = Path(__file__).parent.parent / "visuals" / "en"
VISUALS_SV_PATH = Path(__file__).parent.parent / "visuals" / "sv"

# Colorblind-friendly palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load_data():
    """Load all required data."""
    print("Loading data...")
    
    # Load journeys
    journeys_file = ANALYSIS_PATH / "routes_all_journeys.csv"
    journeys_df = pd.read_csv(journeys_file)
    journeys_df['skandia_time'] = pd.to_datetime(journeys_df['skandia_time'])
    journeys_df['sodertalje_time'] = pd.to_datetime(journeys_df['sodertalje_time'])
    print(f"  Loaded {len(journeys_df)} journeys")
    
    # Load station statistics
    station_stats = pd.read_csv(ANALYSIS_PATH / "station_statistics.csv")
    print(f"  Loaded {len(station_stats)} station statistics")
    
    # Load all delay data for detailed timetable
    delay_files = list(DELAY_PATH.glob("*.parquet"))
    delay_dfs = []
    for f in delay_files:
        delay_dfs.append(pd.read_parquet(f))
    delay_df = pd.concat(delay_dfs, ignore_index=True)
    delay_df['plandatumtid'] = pd.to_datetime(delay_df['plandatumtid'])
    delay_df['utfdatumtid'] = pd.to_datetime(delay_df['utfdatumtid'])
    print(f"  Loaded {len(delay_df):,} delay records")
    
    # Filter to route journeys only
    route_taglanks = set(journeys_df['taglank'].unique())
    route_delay_df = delay_df[delay_df['taglank'].isin(route_taglanks)].copy()
    print(f"  Filtered to {len(route_delay_df):,} route records")
    
    return journeys_df, station_stats, route_delay_df


def build_hourly_traffic_matrix(journeys_df, route_delay_df, station_stats):
    """Build hourly traffic matrices for key stations."""
    print("\nBuilding hourly traffic matrices...")
    
    # Get key stop stations (>30% stop frequency)
    key_stations = station_stats[station_stats['stop_frequency'] > 0.3]['station'].tolist()
    
    # Add endpoints if not present
    for station in ['Göteborg Skandiahamnen', 'Södertälje hamn']:
        if station not in key_stations:
            key_stations.append(station)
    
    print(f"  Key stations: {len(key_stations)}")
    
    # Build matrix for each direction
    matrices = {}
    
    for direction in ['TO_SODERTALJE', 'FROM_SODERTALJE']:
        print(f"  Processing {direction}...")
        
        dir_journeys = journeys_df[journeys_df['direction'] == direction]['taglank'].unique()
        dir_data = route_delay_df[route_delay_df['taglank'].isin(dir_journeys)]
        
        # Filter to key stations
        dir_data = dir_data[dir_data['plats'].isin(key_stations)]
        
        # Extract hour
        dir_data['hour'] = dir_data['plandatumtid'].dt.hour
        dir_data['day_of_week'] = dir_data['plandatumtid'].dt.dayofweek
        
        # Build hourly count matrix
        hourly_counts = dir_data.groupby(['plats', 'hour'])['taglank'].nunique().unstack(fill_value=0)
        matrices[direction] = hourly_counts
        
        print(f"    Stations in matrix: {len(hourly_counts)}")
    
    return matrices, key_stations


def analyze_train_spacing(route_delay_df, journeys_df, station_stats):
    """Analyze headway/spacing between consecutive trains at key stations."""
    print("\nAnalyzing train spacing (headway)...")
    
    # Get key stations
    key_stations = station_stats[station_stats['stop_frequency'] > 0.3]['station'].tolist()
    
    # Add endpoints
    for station in ['Göteborg Skandiahamnen', 'Södertälje hamn', 'Hallsbergs pbg', 'Falköpings c']:
        if station not in key_stations:
            key_stations.append(station)
    
    headway_results = []
    
    for station in key_stations:
        station_data = route_delay_df[route_delay_df['plats'] == station].copy()
        
        if len(station_data) < 2:
            continue
        
        # Get departures only (to measure spacing)
        departures = station_data[station_data['riktningny'] == 'Avgång'].copy()
        if len(departures) < 2:
            departures = station_data.copy()  # Use all records if no departures
        
        departures = departures.sort_values('plandatumtid')
        
        # Calculate time gaps (by day)
        for date in departures['plandatumtid'].dt.date.unique():
            day_deps = departures[departures['plandatumtid'].dt.date == date]
            day_deps = day_deps.sort_values('plandatumtid')
            
            if len(day_deps) >= 2:
                times = day_deps['plandatumtid'].values
                gaps = np.diff(times).astype('timedelta64[m]').astype(float)
                
                for gap in gaps:
                    if 0 < gap < 1440:  # Exclude gaps > 24 hours
                        headway_results.append({
                            'station': station,
                            'date': date,
                            'headway_minutes': gap,
                            'hour': day_deps.iloc[0]['plandatumtid'].hour
                        })
    
    headway_df = pd.DataFrame(headway_results)
    
    if len(headway_df) > 0:
        print(f"  Total headway measurements: {len(headway_df):,}")
        
        # Summary by station
        station_headway = headway_df.groupby('station')['headway_minutes'].agg(['mean', 'median', 'min', 'max', 'count'])
        station_headway = station_headway.sort_values('mean')
        
        print("\n  Headway statistics by station (minutes):")
        print(station_headway.to_string())
    
    return headway_df


def build_detailed_timetable(route_delay_df, journeys_df):
    """Build a detailed timetable for a typical day."""
    print("\nBuilding detailed timetable...")
    
    # Find a typical weekday with most trains
    route_delay_df['date'] = route_delay_df['plandatumtid'].dt.date
    route_delay_df['day_of_week'] = route_delay_df['plandatumtid'].dt.dayofweek
    
    # Count trains per day
    trains_per_day = route_delay_df.groupby('date')['taglank'].nunique()
    
    # Filter to weekdays (Mon-Fri)
    weekday_dates = route_delay_df[route_delay_df['day_of_week'] < 5]['date'].unique()
    weekday_trains = trains_per_day[trains_per_day.index.isin(weekday_dates)]
    
    # Find median day
    median_train_count = weekday_trains.median()
    typical_day = weekday_trains[weekday_trains == weekday_trains.median()].index
    if len(typical_day) == 0:
        typical_day = weekday_trains.idxmax()
    else:
        typical_day = typical_day[0]
    
    print(f"  Typical day: {typical_day} with {trains_per_day[typical_day]} trains")
    
    # Get all trains for this day
    day_data = route_delay_df[route_delay_df['date'] == typical_day]
    
    # Build timetable
    timetable = []
    for taglank in day_data['taglank'].unique():
        train_data = day_data[day_data['taglank'] == taglank].sort_values('plandatumtid')
        
        # Get journey info
        journey_info = journeys_df[journeys_df['taglank'] == taglank]
        if len(journey_info) == 0:
            continue
        
        direction = journey_info.iloc[0]['direction']
        tagnr = train_data.iloc[0].get('tagnr', None)
        
        # Get first and last station times
        first_time = train_data['plandatumtid'].min()
        last_time = train_data['plandatumtid'].max()
        
        timetable.append({
            'taglank': taglank,
            'tagnr': tagnr,
            'direction': direction,
            'departure_time': first_time,
            'arrival_time': last_time,
            'stations': len(train_data['plats'].unique())
        })
    
    timetable_df = pd.DataFrame(timetable)
    timetable_df = timetable_df.sort_values('departure_time')
    
    print(f"  Timetable entries: {len(timetable_df)}")
    
    return timetable_df, typical_day


def save_analysis_outputs(hourly_matrices, headway_df, timetable_df, typical_day, key_stations):
    """Save analysis outputs to CSV files."""
    print("\nSaving analysis outputs...")
    
    # Save hourly traffic matrices
    for direction, matrix in hourly_matrices.items():
        filename = f"hourly_traffic_{direction.lower()}.csv"
        matrix.to_csv(ANALYSIS_PATH / filename)
        print(f"  ✓ Saved {filename}")
    
    # Save headway analysis
    if len(headway_df) > 0:
        headway_df.to_csv(ANALYSIS_PATH / "headway_analysis.csv", index=False)
        print(f"  ✓ Saved headway_analysis.csv")
        
        # Save headway summary
        headway_summary = headway_df.groupby('station')['headway_minutes'].agg(
            ['mean', 'median', 'min', 'max', 'std', 'count']
        ).round(2)
        headway_summary.to_csv(ANALYSIS_PATH / "headway_summary.csv")
        print(f"  ✓ Saved headway_summary.csv")
    
    # Save timetable
    timetable_df.to_csv(ANALYSIS_PATH / "typical_day_timetable.csv", index=False)
    print(f"  ✓ Saved typical_day_timetable.csv (for {typical_day})")
    
    # Save key stations list
    pd.DataFrame({'station': key_stations}).to_csv(ANALYSIS_PATH / "key_stations.csv", index=False)
    print(f"  ✓ Saved key_stations.csv")


def create_visualizations(hourly_matrices, headway_df, timetable_df, typical_day, key_stations):
    """Create visualizations following guidelines."""
    print("\nCreating visualizations...")
    
    # =========================================================================
    # 1. Hourly Traffic Heatmap
    # =========================================================================
    for direction, matrix in hourly_matrices.items():
        for lang, path in [('en', VISUALS_EN_PATH), ('sv', VISUALS_SV_PATH)]:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Reorder stations by average position
            if len(matrix) > 0:
                # Plot heatmap
                im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto')
                
                # Labels
                ax.set_xticks(range(24))
                ax.set_xticklabels(range(24))
                ax.set_yticks(range(len(matrix.index)))
                ax.set_yticklabels(matrix.index, fontsize=8)
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                
                if lang == 'en':
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Station')
                    title = f'Hourly Train Traffic - {"To Södertälje" if "TO" in direction else "From Södertälje"}'
                    cbar.set_label('Number of Trains')
                else:
                    ax.set_xlabel('Timme på dygnet')
                    ax.set_ylabel('Station')
                    title = f'Tågtrafik per timme - {"Till Södertälje" if "TO" in direction else "Från Södertälje"}'
                    cbar.set_label('Antal tåg')
                
                ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
                
                plt.tight_layout()
                
                # Save
                dir_suffix = 'to_sodertalje' if 'TO' in direction else 'from_sodertalje'
                filename = f"sodertalje__hourly_traffic_heatmap_{dir_suffix}__{lang}__v1.png"
                plt.savefig(path / filename, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                # Save description
                desc_file = path / f"sodertalje__hourly_traffic_heatmap_{dir_suffix}__{lang}__v1.txt"
                with open(desc_file, 'w', encoding='utf-8') as f:
                    if lang == 'en':
                        f.write(f"Source: Data/Delay/*.parquet\n")
                        f.write(f"Columns: plandatumtid, plats, taglank\n")
                        f.write(f"Calculation: Count unique trains per station per hour\n")
                        f.write(f"Filter: Journeys on Skandiahamnen-Södertälje route, direction {direction}\n")
                        f.write(f"Language: en\n")
                    else:
                        f.write(f"Källa: Data/Delay/*.parquet\n")
                        f.write(f"Kolumner: plandatumtid, plats, taglank\n")
                        f.write(f"Beräkning: Räkna unika tåg per station per timme\n")
                        f.write(f"Filter: Resor på Skandiahamnen-Södertälje rutten, riktning {direction}\n")
                        f.write(f"Språk: sv\n")
                
                print(f"  ✓ Saved {filename}")
    
    # =========================================================================
    # 2. Train Timeline for Typical Day
    # =========================================================================
    if len(timetable_df) > 0:
        for lang, path in [('en', VISUALS_EN_PATH), ('sv', VISUALS_SV_PATH)]:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Sort by departure time
            timetable_df = timetable_df.sort_values('departure_time')
            
            # Plot each train as a horizontal bar
            y_pos = 0
            colors = {'TO_SODERTALJE': COLORS[0], 'FROM_SODERTALJE': COLORS[1]}
            
            to_count = 0
            from_count = 0
            
            for _, row in timetable_df.iterrows():
                color = colors.get(row['direction'], COLORS[2])
                
                # Convert to hours from midnight
                dep_hour = row['departure_time'].hour + row['departure_time'].minute/60
                arr_hour = row['arrival_time'].hour + row['arrival_time'].minute/60
                
                # Handle overnight trains
                if arr_hour < dep_hour:
                    arr_hour += 24
                
                ax.barh(y_pos, arr_hour - dep_hour, left=dep_hour, height=0.8, 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Count
                if row['direction'] == 'TO_SODERTALJE':
                    to_count += 1
                else:
                    from_count += 1
                
                y_pos += 1
            
            # Format
            ax.set_xlim(0, 28)
            ax.set_xticks(range(0, 29, 2))
            ax.set_xticklabels([f"{h%24:02d}:00" for h in range(0, 29, 2)])
            
            ax.set_ylim(-0.5, y_pos + 0.5)
            ax.set_yticks([])
            
            if lang == 'en':
                ax.set_xlabel('Time of Day')
                ax.set_ylabel('Trains')
                ax.set_title(f'Train Timeline - Typical Day ({typical_day})\n'
                           f'To Södertälje: {to_count} trains | From Södertälje: {from_count} trains',
                           fontsize=14, fontweight='bold')
                legend_labels = ['To Södertälje', 'From Södertälje']
            else:
                ax.set_xlabel('Tid på dygnet')
                ax.set_ylabel('Tåg')
                ax.set_title(f'Tågtidtabell - Typisk dag ({typical_day})\n'
                           f'Till Södertälje: {to_count} tåg | Från Södertälje: {from_count} tåg',
                           fontsize=14, fontweight='bold')
                legend_labels = ['Till Södertälje', 'Från Södertälje']
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors['TO_SODERTALJE'], alpha=0.7, label=legend_labels[0]),
                             Patch(facecolor=colors['FROM_SODERTALJE'], alpha=0.7, label=legend_labels[1])]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            filename = f"sodertalje__train_timeline_typical_day__{lang}__v1.png"
            plt.savefig(path / filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save description
            desc_file = path / f"sodertalje__train_timeline_typical_day__{lang}__v1.txt"
            with open(desc_file, 'w', encoding='utf-8') as f:
                if lang == 'en':
                    f.write(f"Source: Data/Delay/*.parquet, analysis/routes_all_journeys.csv\n")
                    f.write(f"Columns: plandatumtid, taglank, direction\n")
                    f.write(f"Calculation: Plot each train as a bar from departure to arrival time\n")
                    f.write(f"Filter: Typical weekday ({typical_day}) on Skandiahamnen-Södertälje route\n")
                    f.write(f"Language: en\n")
                else:
                    f.write(f"Källa: Data/Delay/*.parquet, analysis/routes_all_journeys.csv\n")
                    f.write(f"Kolumner: plandatumtid, taglank, direction\n")
                    f.write(f"Beräkning: Rita varje tåg som en stapel från avgång till ankomst\n")
                    f.write(f"Filter: Typisk vardag ({typical_day}) på Skandiahamnen-Södertälje rutten\n")
                    f.write(f"Språk: sv\n")
            
            print(f"  ✓ Saved {filename}")
    
    # =========================================================================
    # 3. Headway Distribution
    # =========================================================================
    if len(headway_df) > 0:
        for lang, path in [('en', VISUALS_EN_PATH), ('sv', VISUALS_SV_PATH)]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Overall headway distribution
            ax1 = axes[0]
            headway_df['headway_minutes'].hist(bins=50, ax=ax1, color=COLORS[0], 
                                               alpha=0.7, edgecolor='black')
            
            mean_headway = headway_df['headway_minutes'].mean()
            median_headway = headway_df['headway_minutes'].median()
            
            ax1.axvline(mean_headway, color=COLORS[1], linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_headway:.0f} min' if lang == 'en' else f'Medel: {mean_headway:.0f} min')
            ax1.axvline(median_headway, color=COLORS[2], linestyle='--', linewidth=2,
                       label=f'Median: {median_headway:.0f} min')
            
            if lang == 'en':
                ax1.set_xlabel('Headway (minutes)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Train Headway')
            else:
                ax1.set_xlabel('Tidslucka (minuter)')
                ax1.set_ylabel('Frekvens')
                ax1.set_title('Fördelning av tidsluckor mellan tåg')
            
            ax1.legend(loc='upper right')
            
            # Headway by hour
            ax2 = axes[1]
            hourly_headway = headway_df.groupby('hour')['headway_minutes'].mean()
            ax2.bar(hourly_headway.index, hourly_headway.values, color=COLORS[0], 
                   alpha=0.7, edgecolor='black')
            
            if lang == 'en':
                ax2.set_xlabel('Hour of Day')
                ax2.set_ylabel('Average Headway (minutes)')
                ax2.set_title('Average Headway by Hour')
            else:
                ax2.set_xlabel('Timme på dygnet')
                ax2.set_ylabel('Genomsnittlig tidslucka (minuter)')
                ax2.set_title('Genomsnittlig tidslucka per timme')
            
            ax2.set_xticks(range(0, 24, 2))
            
            plt.tight_layout()
            
            filename = f"sodertalje__headway_analysis__{lang}__v1.png"
            plt.savefig(path / filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save description
            desc_file = path / f"sodertalje__headway_analysis__{lang}__v1.txt"
            with open(desc_file, 'w', encoding='utf-8') as f:
                if lang == 'en':
                    f.write(f"Source: Data/Delay/*.parquet\n")
                    f.write(f"Columns: plandatumtid, plats, taglank\n")
                    f.write(f"Calculation: Time gap between consecutive trains at each station\n")
                    f.write(f"Filter: Journeys on Skandiahamnen-Södertälje route, gaps < 24h\n")
                    f.write(f"Language: en\n")
                else:
                    f.write(f"Källa: Data/Delay/*.parquet\n")
                    f.write(f"Kolumner: plandatumtid, plats, taglank\n")
                    f.write(f"Beräkning: Tidslucka mellan på varandra följande tåg vid varje station\n")
                    f.write(f"Filter: Resor på Skandiahamnen-Södertälje rutten, luckor < 24h\n")
                    f.write(f"Språk: sv\n")
            
            print(f"  ✓ Saved {filename}")
    
    # =========================================================================
    # 4. Trains per Hour of Day
    # =========================================================================
    for lang, path in [('en', VISUALS_EN_PATH), ('sv', VISUALS_SV_PATH)]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Combine both directions
        hourly_data = pd.DataFrame(index=range(24))
        for direction, matrix in hourly_matrices.items():
            if len(matrix) > 0:
                # Sum across all stations
                hourly_total = matrix.sum(axis=0)
                # Reindex to ensure all hours 0-23
                hourly_total = hourly_total.reindex(range(24), fill_value=0)
                hourly_data[direction] = hourly_total.values
        
        if len(hourly_data.columns) > 0:
            # Average across directions
            combined = hourly_data.mean(axis=1)
            
            ax.bar(range(24), combined.values, color=COLORS[0], alpha=0.7, edgecolor='black')
            
            if lang == 'en':
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Average Train Passages')
                ax.set_title('Train Traffic by Hour of Day\nSkandiahamnen ↔ Södertälje Hamn Route',
                           fontsize=14, fontweight='bold')
            else:
                ax.set_xlabel('Timme på dygnet')
                ax.set_ylabel('Genomsnittligt antal tågpassager')
                ax.set_title('Tågtrafik per timme på dygnet\nSkandiahamnen ↔ Södertälje Hamn rutten',
                           fontsize=14, fontweight='bold')
            
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h:02d}" for h in range(24)])
            
            plt.tight_layout()
            
            filename = f"sodertalje__trains_per_hour__{lang}__v1.png"
            plt.savefig(path / filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save description
            desc_file = path / f"sodertalje__trains_per_hour__{lang}__v1.txt"
            with open(desc_file, 'w', encoding='utf-8') as f:
                if lang == 'en':
                    f.write(f"Source: Data/Delay/*.parquet\n")
                    f.write(f"Columns: plandatumtid, plats, taglank\n")
                    f.write(f"Calculation: Count train passages per hour across key stations\n")
                    f.write(f"Filter: Journeys on Skandiahamnen-Södertälje route\n")
                    f.write(f"Language: en\n")
                else:
                    f.write(f"Källa: Data/Delay/*.parquet\n")
                    f.write(f"Kolumner: plandatumtid, plats, taglank\n")
                    f.write(f"Beräkning: Räkna tågpassager per timme vid nyckelstationer\n")
                    f.write(f"Filter: Resor på Skandiahamnen-Södertälje rutten\n")
                    f.write(f"Språk: sv\n")
            
            print(f"  ✓ Saved {filename}")


def main():
    print("="*60)
    print("PHASE 3: Timetable Analysis")
    print("="*60)
    
    # Load data
    journeys_df, station_stats, route_delay_df = load_data()
    
    # Build hourly traffic matrices
    hourly_matrices, key_stations = build_hourly_traffic_matrix(journeys_df, route_delay_df, station_stats)
    
    # Analyze train spacing
    headway_df = analyze_train_spacing(route_delay_df, journeys_df, station_stats)
    
    # Build detailed timetable
    timetable_df, typical_day = build_detailed_timetable(route_delay_df, journeys_df)
    
    # Save analysis outputs
    save_analysis_outputs(hourly_matrices, headway_df, timetable_df, typical_day, key_stations)
    
    # Create visualizations
    create_visualizations(hourly_matrices, headway_df, timetable_df, typical_day, key_stations)
    
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)
    
    # Summary
    if len(headway_df) > 0:
        print(f"\nKey findings:")
        print(f"  • Mean headway: {headway_df['headway_minutes'].mean():.1f} minutes")
        print(f"  • Median headway: {headway_df['headway_minutes'].median():.1f} minutes")
        print(f"  • Min observed headway: {headway_df['headway_minutes'].min():.1f} minutes")
        print(f"  • Trains on typical day ({typical_day}): {len(timetable_df)}")
        
        # Peak hours
        hourly_counts = headway_df.groupby('hour').size()
        peak_hours = hourly_counts.nlargest(3).index.tolist()
        print(f"  • Peak traffic hours: {', '.join(f'{h:02d}:00' for h in sorted(peak_hours))}")


if __name__ == "__main__":
    main()
