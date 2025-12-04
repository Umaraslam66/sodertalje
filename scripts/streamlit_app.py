"""
Streamlit Application: Södertälje Capacity Slot Finder
Purpose: Interactive tool to visualize train traffic and find available capacity slots
Author: AI Agent
Date: 2024-12
Version: v2.0 - Added proper time-space diagrams (Grafischer Fahrplan)

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import ast

# Page config
st.set_page_config(
    page_title="Södertälje Capacity Analysis",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - use absolute path to handle different working directories
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_PATH = SCRIPT_DIR.parent / "analysis"

# Colors for different trains
TRAIN_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly


@st.cache_data
def load_base_data():
    """Load pre-processed analysis data."""
    data = {}
    
    # Load journeys
    data['journeys'] = pd.read_csv(ANALYSIS_PATH / "routes_all_journeys.csv")
    data['journeys']['skandia_time'] = pd.to_datetime(data['journeys']['skandia_time'])
    data['journeys']['sodertalje_time'] = pd.to_datetime(data['journeys']['sodertalje_time'])
    data['journeys']['journey_date'] = pd.to_datetime(data['journeys']['journey_date']).dt.date
    
    # Load summary
    data['summary'] = pd.read_csv(ANALYSIS_PATH / "routes_summary.csv")
    
    # Load station statistics
    data['station_stats'] = pd.read_csv(ANALYSIS_PATH / "station_statistics.csv")
    
    # Load station sequences (for path identification)
    data['sequences'] = pd.read_csv(ANALYSIS_PATH / "station_sequences.csv")
    
    # Load slots
    data['slots_to'] = pd.read_csv(ANALYSIS_PATH / "available_slots_to_sodertalje.csv")
    data['slots_from'] = pd.read_csv(ANALYSIS_PATH / "available_slots_from_sodertalje.csv")
    
    # Load slot summary
    data['slot_summary'] = pd.read_csv(ANALYSIS_PATH / "slot_summary.csv")
    
    # Load travel times
    data['travel_times'] = pd.read_csv(ANALYSIS_PATH / "station_travel_times.csv")
    
    # Load canonical station order
    data['station_order'] = pd.read_csv(ANALYSIS_PATH / "canonical_station_order.csv")
    
    return data


@st.cache_data
def load_route_delay_data():
    """Load pre-processed route delay data from CSV (much faster than parquet)."""
    route_delay = pd.read_csv(ANALYSIS_PATH / "route_delay_data.csv")
    route_delay['plandatumtid'] = pd.to_datetime(route_delay['plandatumtid'])
    route_delay['utfdatumtid'] = pd.to_datetime(route_delay['utfdatumtid'])
    route_delay['date'] = route_delay['plandatumtid'].dt.date
    return route_delay


@st.cache_data
def load_endpoint_movements():
    """Load all train movements at endpoint stations (Skandiahamnen and Södertälje hamn)."""
    endpoint_df = pd.read_csv(ANALYSIS_PATH / "endpoint_movements.csv")
    endpoint_df['plandatumtid'] = pd.to_datetime(endpoint_df['plandatumtid'])
    endpoint_df['utfdatumtid'] = pd.to_datetime(endpoint_df['utfdatumtid'])
    endpoint_df['date'] = endpoint_df['plandatumtid'].dt.date
    endpoint_df['hour'] = endpoint_df['plandatumtid'].dt.hour
    endpoint_df['weekday'] = endpoint_df['plandatumtid'].dt.dayofweek
    endpoint_df['weekday_name'] = endpoint_df['plandatumtid'].dt.day_name()
    return endpoint_df


@st.cache_data
def load_april10_traffic():
    """Load April 10 overnight traffic data (22:00 - 02:00 window)."""
    traffic_df = pd.read_csv(ANALYSIS_PATH / "april10_overnight_traffic.csv")
    traffic_df['plandatumtid'] = pd.to_datetime(traffic_df['plandatumtid'])
    station_order = pd.read_csv(ANALYSIS_PATH / "april10_station_order.csv")
    return traffic_df, station_order


def get_unique_paths(sequences_df):
    """Extract unique path variants from the sequences."""
    # Group by station_sequence and count
    path_counts = sequences_df.groupby('station_sequence').agg({
        'taglank': 'count',
        'direction': 'first',
        'station_count': 'first'
    }).reset_index()
    path_counts.columns = ['path', 'count', 'direction', 'station_count']
    path_counts = path_counts.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Create a short path ID using enumerate for proper numbering
    path_counts['path_id'] = [f"Path {idx+1} ({row['count']} trains, {int(row['station_count'])} stations)" 
                              for idx, row in enumerate(path_counts.to_dict('records'))]
    
    return path_counts


def get_operating_days(journeys_df):
    """Get all unique operating days."""
    dates = journeys_df['journey_date'].dropna().unique()
    dates = sorted(dates)
    return dates


def build_time_space_data_simple(route_delay_df, journeys_df, selected_date, direction, selected_taglanks=None):
    """Build time-space data for ONE direction with simple, clear logic.
    
    ROBUST approach - follow the data:
    1. Get taglanks for this direction that depart on selected_date
    2. Fetch ALL data for those taglanks 
    3. For each train, look at actual start/end times to determine if overnight
    4. Build station order from actual train path
    5. Calculate time relative to midnight of departure date
    """
    # Convert selected_date to string format for comparison
    if hasattr(selected_date, 'strftime'):
        date_str = selected_date.strftime('%Y-%m-%d')
    else:
        date_str = str(selected_date)
    
    # Get taglanks that DEPART on selected_date in this direction
    journeys_df = journeys_df.copy()
    journeys_df['journey_date_str'] = pd.to_datetime(journeys_df['journey_date']).dt.strftime('%Y-%m-%d')
    day_journeys = journeys_df[
        (journeys_df['journey_date_str'] == date_str) & 
        (journeys_df['direction'] == direction)
    ]
    
    if len(day_journeys) == 0:
        return pd.DataFrame(), [], {}
    
    # Get the taglanks for trains departing this day
    dir_taglanks = day_journeys['taglank'].tolist()
    
    if selected_taglanks is not None:
        dir_taglanks = [tl for tl in dir_taglanks if tl in selected_taglanks]
    
    if len(dir_taglanks) == 0:
        return pd.DataFrame(), [], {}
    
    # Get ALL delay data for these taglanks (including next-day data for overnight trains)
    day_data = route_delay_df[route_delay_df['taglank'].isin(dir_taglanks)].copy()
    
    if len(day_data) == 0:
        return pd.DataFrame(), [], {}
    
    # Base datetime = midnight of selected date
    base_date = pd.Timestamp(date_str)
    
    # First pass: analyze each train's actual journey times
    journey_info = {}
    for taglank in dir_taglanks:
        train_data = day_data[day_data['taglank'] == taglank].sort_values('plandatumtid')
        if len(train_data) == 0:
            continue
        
        first_time = train_data['plandatumtid'].iloc[0]
        last_time = train_data['plandatumtid'].iloc[-1]
        first_station = train_data['plats'].iloc[0]
        last_station = train_data['plats'].iloc[-1]
        
        # Calculate hours from midnight of departure date
        start_hours = (first_time - base_date).total_seconds() / 3600
        end_hours = (last_time - base_date).total_seconds() / 3600
        
        # Determine if overnight (ends after midnight = end_hours >= 24)
        is_overnight = end_hours >= 24 or last_time.date() > first_time.date()
        
        journey_info[taglank] = {
            'start_time': first_time,
            'end_time': last_time,
            'start_station': first_station,
            'end_station': last_station,
            'start_hours': start_hours,
            'end_hours': end_hours,
            'is_overnight': is_overnight,
            'duration_hours': end_hours - start_hours
        }
    
    # Build station order from the FIRST train's actual journey
    first_taglank = dir_taglanks[0]
    first_train = day_data[day_data['taglank'] == first_taglank].sort_values('plandatumtid')
    
    # Get unique stations in order of appearance (first occurrence only)
    seen_stations = set()
    station_order = []
    for station in first_train['plats']:
        if station not in seen_stations:
            station_order.append(station)
            seen_stations.add(station)
    
    # Create station to Y position mapping
    station_to_y = {station: i for i, station in enumerate(station_order)}
    
    # Now process all trains
    trains_data = []
    
    for taglank in dir_taglanks:
        train_data = day_data[day_data['taglank'] == taglank].sort_values('plandatumtid')
        
        if len(train_data) == 0:
            continue
        
        info = journey_info.get(taglank, {})
        
        for _, row in train_data.iterrows():
            station = row['plats']
            time = row['plandatumtid']
            event_type = row.get('riktningny', 'Pass')
            
            # Calculate time as hours from midnight of departure date
            # This naturally handles overnight - 23:44 = 23.73, 00:02 next day = 24.03
            time_hours = (time - base_date).total_seconds() / 3600
            
            # Get Y position - if station not in our order, add it
            if station not in station_to_y:
                station_to_y[station] = len(station_order)
                station_order.append(station)
            
            y_pos = station_to_y[station]
            
            trains_data.append({
                'taglank': taglank,
                'tagnr': taglank,  # Use taglank as identifier
                'direction': direction,
                'station': station,
                'y_pos': y_pos,
                'time': time,
                'time_hours': time_hours,
                'event_type': event_type,
                'is_overnight': info.get('is_overnight', False)
            })
    
    return pd.DataFrame(trains_data), station_order, journey_info


def create_direction_diagram(ts_data, station_order, direction, title, yaxis_side='left'):
    """Create a time-space diagram for a single direction.
    
    Args:
        yaxis_side: 'left' or 'right' - which side to put Y-axis labels
    """
    if len(ts_data) == 0 or len(station_order) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trains for this direction", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    fig = go.Figure()
    
    # Color for this direction
    color = '#2E86AB' if direction == 'TO_SODERTALJE' else '#E94F37'
    
    # Plot each train
    for taglank in ts_data['taglank'].unique():
        train = ts_data[ts_data['taglank'] == taglank].sort_values('time_hours')
        
        tagnr = str(train['tagnr'].iloc[0])  # Convert to string for Plotly
        x_vals = train['time_hours'].tolist()
        y_vals = train['y_pos'].tolist()
        stations = train['station'].tolist()
        
        # Hover text
        hover_texts = []
        for x, y, s in zip(x_vals, y_vals, stations):
            h = int(x) % 24
            m = int((x % 1) * 60)
            day_note = " (+1 day)" if x >= 24 else (" (prev day)" if x < 0 else "")
            hover_texts.append(f"<b>{tagnr}</b><br>{s}<br>{h:02d}:{m:02d}{day_note}")
        
        # Draw train line
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=tagnr,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>"
        ))
        
        # Draw dwell times (thicker horizontal lines)
        for j in range(len(stations) - 1):
            if stations[j] == stations[j + 1]:
                fig.add_trace(go.Scatter(
                    x=[x_vals[j], x_vals[j + 1]],
                    y=[y_vals[j], y_vals[j + 1]],
                    mode='lines',
                    line=dict(color=color, width=6),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Time range
    min_time = ts_data['time_hours'].min()
    max_time = ts_data['time_hours'].max()
    
    # Build time axis labels
    tick_start = int(min_time) - (int(min_time) % 2)  # Round down to even hour
    tick_end = int(max_time) + 2
    tick_vals = list(range(tick_start, tick_end, 2))
    tick_text = []
    for h in tick_vals:
        if h < 0:
            tick_text.append(f"{(h % 24):02d}:00 -1d")
        elif h >= 24:
            tick_text.append(f"{(h % 24):02d}:00 +1d")
        else:
            tick_text.append(f"{h:02d}:00")
    
    # Margins based on Y-axis side
    if yaxis_side == 'left':
        margins = dict(l=180, r=10, t=50, b=50)
    else:
        margins = dict(l=10, r=180, t=50, b=50)
    
    # For right-side Y-axis, flip the station order so both diagrams have
    # origin at bottom and destination at top (mirrored view)
    if yaxis_side == 'right':
        # Reverse the Y-axis range to flip the diagram
        y_range = [len(station_order) - 0.5, -0.5]
        # Reverse station labels too
        display_stations = station_order[::-1]
        display_ticks = list(range(len(station_order) - 1, -1, -1))
    else:
        y_range = [-0.5, len(station_order) - 0.5]
        display_stations = station_order
        display_ticks = list(range(len(station_order)))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            title="Time",
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            gridcolor='lightgray',
            range=[min_time - 0.5, max_time + 0.5]
        ),
        yaxis=dict(
            title="",
            tickmode='array',
            tickvals=display_ticks,
            ticktext=display_stations,
            gridcolor='lightgray',
            range=y_range,
            side=yaxis_side  # Put Y-axis on specified side
        ),
        height=max(600, len(station_order) * 16),
        margin=margins,
        showlegend=False,  # Hide legend for cleaner side-by-side view
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    # Add reference lines at 6-hour intervals
    for h in range(tick_start, tick_end, 6):
        if min_time - 1 <= h <= max_time + 1:
            fig.add_vline(x=h, line_dash="dash", line_color="lightgray", opacity=0.5)
    
    return fig


def show_time_space_diagram_page(base_data, route_delay_df):
    """Show the time-space diagram page with separate diagrams per direction."""
    st.header("📊 Time-Space Diagram (Grafischer Fahrplan)")
    
    st.write("""
    This diagram shows train movements over time. Stations are on the Y-axis, time on the X-axis.
    - **Diagonal lines** show trains moving between stations
    - **Horizontal segments** show dwell time (stops) at stations
    - Each direction has its own diagram with properly ordered stations
    """)
    
    journeys = base_data['journeys']
    sequences = base_data['sequences']
    
    # Get all operating days
    operating_days = get_operating_days(journeys)
    
    # Sidebar filters
    st.sidebar.subheader("📅 Filters")
    
    # Date selector
    selected_date = st.sidebar.selectbox(
        "Select Operating Day",
        options=operating_days,
        format_func=lambda x: x.strftime("%Y-%m-%d (%A)") if hasattr(x, 'strftime') else str(x),
        index=len(operating_days) // 2  # Default to middle of year
    )
    
    # Build time-space data for BOTH directions separately
    with st.spinner("Building time-space diagrams..."):
        ts_to, stations_to, info_to = build_time_space_data_simple(
            route_delay_df, journeys, selected_date, 'TO_SODERTALJE'
        )
        ts_from, stations_from, info_from = build_time_space_data_simple(
            route_delay_df, journeys, selected_date, 'FROM_SODERTALJE'
        )
    
    # Show statistics
    to_count = ts_to['taglank'].nunique() if len(ts_to) > 0 else 0
    from_count = ts_from['taglank'].nunique() if len(ts_from) > 0 else 0
    total_trains = to_count + from_count
    
    # Check for overnight trains
    overnight_to = sum(1 for v in info_to.values() if v.get('is_overnight', False))
    overnight_from = sum(1 for v in info_from.values() if v.get('is_overnight', False))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trains", total_trains)
    with col2:
        st.metric("🔵 To Södertälje", to_count)
    with col3:
        st.metric("🔴 From Södertälje", from_count)
    with col4:
        st.metric("🌙 Overnight", overnight_to + overnight_from)
    
    if total_trains == 0:
        st.warning(f"No trains found for {selected_date}")
        st.info("Try selecting a different date.")
        return
    
    st.divider()
    
    # Show diagrams SIDE BY SIDE with Y-axis on outer edges
    # Left diagram: To Södertälje (Y-axis on LEFT)
    # Right diagram: From Södertälje (Y-axis on RIGHT)
    # This creates a mirrored view effect
    
    if to_count > 0 and from_count > 0:
        # Both directions - show side by side
        st.markdown(f"### 🔵 To Södertälje ← | → From Södertälje 🔴")
        overnight_info = ""
        if overnight_to > 0 or overnight_from > 0:
            overnight_info = f" (🌙 {overnight_to + overnight_from} overnight trains)"
        st.caption(f"Skandiahamnen ↔ Södertälje Hamn{overnight_info}")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            fig_to = create_direction_diagram(
                ts_to, stations_to, 'TO_SODERTALJE', 
                f"→ To Södertälje",
                yaxis_side='left'
            )
            st.plotly_chart(fig_to, use_container_width=True)
        
        with col_right:
            fig_from = create_direction_diagram(
                ts_from, stations_from, 'FROM_SODERTALJE',
                f"← From Södertälje",
                yaxis_side='right'
            )
            st.plotly_chart(fig_from, use_container_width=True)
        
        # Train details below in two columns
        col_left, col_right = st.columns(2)
        
        with col_left:
            with st.expander(f"📋 To Södertälje ({to_count} trains)"):
                train_rows = []
                for taglank, info in info_to.items():
                    train_rows.append({
                        'ID': taglank[-8:],  # Short ID
                        'Dep': f"{int(info['start_hours'])%24:02d}:{int((info['start_hours']%1)*60):02d}",
                        'Arr': f"{int(info['end_hours'])%24:02d}:{int((info['end_hours']%1)*60):02d}" + ("+" if info['end_hours'] >= 24 else ""),
                        '🌙': '✓' if info['is_overnight'] else ''
                    })
                st.dataframe(pd.DataFrame(train_rows), use_container_width=True, hide_index=True)
        
        with col_right:
            with st.expander(f"📋 From Södertälje ({from_count} trains)"):
                train_rows = []
                for taglank, info in info_from.items():
                    train_rows.append({
                        'ID': taglank[-8:],  # Short ID
                        'Dep': f"{int(info['start_hours'])%24:02d}:{int((info['start_hours']%1)*60):02d}",
                        'Arr': f"{int(info['end_hours'])%24:02d}:{int((info['end_hours']%1)*60):02d}" + ("+" if info['end_hours'] >= 24 else ""),
                        '🌙': '✓' if info['is_overnight'] else ''
                    })
                st.dataframe(pd.DataFrame(train_rows), use_container_width=True, hide_index=True)
    
    elif to_count > 0:
        # Only To direction
        st.subheader(f"🔵 To Södertälje (Skandiahamnen → Södertälje Hamn)")
        fig_to = create_direction_diagram(
            ts_to, stations_to, 'TO_SODERTALJE', 
            f"Trains TO Södertälje - {selected_date}",
            yaxis_side='left'
        )
        st.plotly_chart(fig_to, use_container_width=True)
        
        with st.expander(f"📋 Train Details ({to_count} trains)"):
            train_rows = []
            for taglank, info in info_to.items():
                train_rows.append({
                    'Journey ID': taglank,
                    'Departure': f"{int(info['start_hours'])%24:02d}:{int((info['start_hours']%1)*60):02d}",
                    'Arrival': f"{int(info['end_hours'])%24:02d}:{int((info['end_hours']%1)*60):02d}" + (" +1d" if info['end_hours'] >= 24 else ""),
                    'Duration': f"{info['duration_hours']:.1f}h",
                    'Overnight': '🌙' if info['is_overnight'] else ''
                })
            st.dataframe(pd.DataFrame(train_rows), use_container_width=True)
    
    elif from_count > 0:
        # Only From direction
        st.subheader(f"🔴 From Södertälje (Södertälje Hamn → Skandiahamnen)")
        fig_from = create_direction_diagram(
            ts_from, stations_from, 'FROM_SODERTALJE',
            f"Trains FROM Södertälje - {selected_date}",
            yaxis_side='left'
        )
        st.plotly_chart(fig_from, use_container_width=True)
        
        with st.expander(f"📋 Train Details ({from_count} trains)"):
            train_rows = []
            for taglank, info in info_from.items():
                train_rows.append({
                    'Journey ID': taglank,
                    'Departure': f"{int(info['start_hours'])%24:02d}:{int((info['start_hours']%1)*60):02d}",
                    'Arrival': f"{int(info['end_hours'])%24:02d}:{int((info['end_hours']%1)*60):02d}" + (" +1d" if info['end_hours'] >= 24 else ""),
                    'Duration': f"{info['duration_hours']:.1f}h",
                    'Overnight': '🌙' if info['is_overnight'] else ''
                })
            st.dataframe(pd.DataFrame(train_rows), use_container_width=True)


def show_overview(data):
    """Show route overview page."""
    st.header("🚂 Route Overview")
    
    summary = data['summary'].iloc[0]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Journeys", f"{summary['total_journeys']:,}")
    
    with col2:
        st.metric("To Södertälje", f"{summary['to_sodertalje_count']:,}")
    
    with col3:
        st.metric("From Södertälje", f"{summary['from_sodertalje_count']:,}")
    
    with col4:
        st.metric("Avg Journey Time", f"{summary['mean_duration_hours']:.1f}h")
    
    st.divider()
    
    # Operators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Operators")
        operators = eval(summary['operators']) if isinstance(summary['operators'], str) else summary['operators']
        for op in operators:
            st.write(f"• {op}")
    
    with col2:
        st.subheader("Route Variants")
        sequences = data['sequences']
        path_counts = sequences.groupby('station_sequence').size().reset_index(name='count')
        st.write(f"**{len(path_counts)}** unique route variants identified")
        st.write(f"Most trains use {path_counts['count'].max()} common paths")
    
    # Journey distribution
    st.subheader("Journey Distribution")
    
    journeys = data['journeys']
    
    col1, col2 = st.columns(2)
    
    with col1:
        direction_counts = journeys['direction'].value_counts()
        fig = px.pie(
            values=direction_counts.values,
            names=['From Södertälje', 'To Södertälje'] if 'FROM' in direction_counts.index[0] else ['To Södertälje', 'From Södertälje'],
            color_discrete_sequence=['#ff7f0e', '#1f77b4'],
            title="By Direction"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly distribution
        journeys['month'] = pd.to_datetime(journeys['journey_date']).dt.month
        monthly = journeys.groupby('month').size()
        fig = px.bar(
            x=monthly.index,
            y=monthly.values,
            labels={'x': 'Month', 'y': 'Trains'},
            title="Trains per Month"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Station statistics
    st.subheader("Key Stop Stations")
    st.write("Stations where trains frequently make scheduled stops:")
    
    station_stats = data['station_stats']
    key_stations = station_stats[station_stats['stop_frequency'] > 0.3].sort_values('stop_frequency', ascending=False)
    
    if len(key_stations) > 0:
        fig = px.bar(
            key_stations.head(15),
            x='station',
            y='stop_frequency',
            title='Stop Frequency by Station',
            labels={'stop_frequency': 'Stop Frequency', 'station': 'Station'},
            color='stop_frequency',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)


def show_slot_finder(data):
    """Show slot finder page."""
    st.header("🔍 Capacity Slot Finder")
    
    st.write("""
    Find available time windows for scheduling new freight trains on the 
    Skandiahamnen ↔ Södertälje Hamn route.
    """)
    
    # Summary metrics
    slot_summary = data['slot_summary'].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Available Slots (To Södertälje)",
            f"{slot_summary['to_sodertalje_available']} slots",
            delta=f"{slot_summary['to_sodertalje_available'] * 15 / 60:.1f} hours"
        )
    
    with col2:
        st.metric(
            "Available Slots (From Södertälje)",
            f"{slot_summary['from_sodertalje_available']} slots",
            delta=f"{slot_summary['from_sodertalje_available'] * 15 / 60:.1f} hours"
        )
    
    with col3:
        total_available = slot_summary['to_sodertalje_available'] + slot_summary['from_sodertalje_available']
        st.metric("Total Available", f"{total_available} slots")
    
    st.divider()
    
    # Interactive slot finder
    st.subheader("Find Your Slot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        direction = st.selectbox(
            "Direction",
            options=['TO_SODERTALJE', 'FROM_SODERTALJE'],
            format_func=lambda x: '→ To Södertälje' if 'TO' in x else '← From Södertälje',
            key='slot_direction'
        )
    
    with col2:
        desired_hour = st.slider(
            "Desired Departure Hour",
            min_value=0, max_value=23, value=8
        )
    
    with col3:
        flexibility = st.slider(
            "Flexibility (± hours)",
            min_value=1, max_value=6, value=2
        )
    
    # Get slots for selected direction
    slots_df = data['slots_to'] if 'TO' in direction else data['slots_from']
    
    # Filter to time window
    start_minute = (desired_hour - flexibility) * 60
    end_minute = (desired_hour + flexibility) * 60
    
    window_slots = slots_df[
        (slots_df['departure_minutes'] >= start_minute) & 
        (slots_df['departure_minutes'] <= end_minute)
    ].copy()
    
    # Show results
    if len(window_slots) > 0:
        available = window_slots[window_slots['status'] == 'AVAILABLE']
        limited = window_slots[window_slots['status'] == 'LIMITED']
        congested = window_slots[window_slots['status'] == 'CONGESTED']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"✅ Available: {len(available)} slots")
        with col2:
            st.warning(f"⚠️ Limited: {len(limited)} slots")
        with col3:
            st.error(f"🚫 Congested: {len(congested)} slots")
        
        if len(available) > 0:
            st.subheader("✅ Recommended Departure Times")
            rec_times = available['departure_time'].tolist()
            
            cols = st.columns(min(5, len(rec_times)))
            for i, time in enumerate(rec_times[:5]):
                with cols[i]:
                    st.info(f"🕐 {time}")
            
            if len(rec_times) > 5:
                st.write(f"... and {len(rec_times) - 5} more available slots")
    
    # Visualization
    st.subheader("Slot Availability Overview")
    
    fig = go.Figure()
    
    colors = {'AVAILABLE': '#2ca02c', 'LIMITED': '#ff7f0e', 'CONGESTED': '#d62728'}
    
    for status in ['AVAILABLE', 'LIMITED', 'CONGESTED']:
        status_data = slots_df[slots_df['status'] == status]
        if len(status_data) > 0:
            fig.add_trace(go.Bar(
                x=status_data['departure_minutes'] / 60,
                y=status_data['total_conflicts'] + 0.5,
                name=status.capitalize(),
                marker_color=colors[status],
                opacity=0.7,
                width=0.2
            ))
    
    fig.add_vrect(
        x0=desired_hour - flexibility,
        x1=desired_hour + flexibility,
        fillcolor="rgba(0, 100, 255, 0.1)",
        layer="below",
        line_width=2,
        line_color="blue",
        line_dash="dash"
    )
    
    fig.update_layout(
        title=f"Slot Availability - {'To Södertälje' if 'TO' in direction else 'From Södertälje'}",
        xaxis_title="Departure Time (Hour)",
        yaxis_title="Conflict Score",
        barmode='overlay',
        height=400,
        xaxis=dict(range=[0, 24], dtick=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_path_analysis(base_data):
    """Show path analysis page."""
    st.header("🛤️ Route Path Analysis")
    
    st.write("""
    Analyze the different route variants that trains take between Skandiahamnen and Södertälje.
    Different trains may take slightly different paths depending on operational requirements.
    """)
    
    sequences = base_data['sequences']
    
    # Get unique paths
    path_data = get_unique_paths(sequences)
    
    # Direction filter
    direction = st.selectbox(
        "Filter by Direction",
        options=['All', 'TO_SODERTALJE', 'FROM_SODERTALJE'],
        format_func=lambda x: {
            'All': '↔ All Directions',
            'TO_SODERTALJE': '→ To Södertälje',
            'FROM_SODERTALJE': '← From Södertälje'
        }.get(x, x)
    )
    
    if direction != 'All':
        filtered_paths = path_data[path_data['direction'] == direction]
    else:
        filtered_paths = path_data
    
    st.subheader(f"Found {len(filtered_paths)} unique route variants")
    
    # Show top paths
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            filtered_paths.head(10),
            x='path_id',
            y='count',
            color='direction',
            title="Top 10 Most Common Routes",
            labels={'count': 'Number of Trains', 'path_id': 'Route'},
            color_discrete_map={'TO_SODERTALJE': '#1f77b4', 'FROM_SODERTALJE': '#ff7f0e'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Path Summary")
        st.write(f"**Total Variants:** {len(filtered_paths)}")
        st.write(f"**Most Common:** {filtered_paths.iloc[0]['count']} trains")
        st.write(f"**Station Range:** {filtered_paths['station_count'].min()}-{filtered_paths['station_count'].max()} stations")
    
    # Path details
    st.subheader("Path Details")
    
    selected_path_id = st.selectbox(
        "Select a route to view details",
        options=filtered_paths['path_id'].tolist()
    )
    
    if selected_path_id:
        path_row = filtered_paths[filtered_paths['path_id'] == selected_path_id].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"**Direction:** {'→ To Södertälje' if path_row['direction'] == 'TO_SODERTALJE' else '← From Södertälje'}")
            st.write(f"**Trains using this path:** {path_row['count']}")
            st.write(f"**Total stations:** {path_row['station_count']}")
        
        with col2:
            st.write("**Station Sequence:**")
            stations = path_row['path'].split(' → ')
            
            # Show as numbered list in columns
            n_cols = 3
            cols = st.columns(n_cols)
            per_col = len(stations) // n_cols + 1
            
            for i, col in enumerate(cols):
                with col:
                    start_idx = i * per_col
                    end_idx = min((i + 1) * per_col, len(stations))
                    for j, station in enumerate(stations[start_idx:end_idx], start=start_idx + 1):
                        st.write(f"{j}. {station}")


def show_endpoint_traffic(endpoint_df):
    """Show all traffic at endpoint stations - helps identify available capacity slots."""
    st.header("🚉 Endpoint Station Traffic")
    
    st.write("""
    View **ALL** train movements at Skandiahamnen and Södertälje Hamn - not just the route trains.
    This shows the full picture of station activity to identify available capacity windows.
    
    - **Arrivals (Ankomst):** Trains arriving at the station
    - **Departures (Avgång):** Trains departing from the station
    """)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    skandia_data = endpoint_df[endpoint_df['plats'] == 'Göteborg Skandiahamnen']
    sodertalje_data = endpoint_df[endpoint_df['plats'] == 'Södertälje hamn']
    
    with col1:
        st.metric("🚢 Skandiahamnen Total", len(skandia_data))
    with col2:
        st.metric("🚢 Skandia Unique Trains", skandia_data['taglank'].nunique())
    with col3:
        st.metric("⚓ Södertälje Total", len(sodertalje_data))
    with col4:
        st.metric("⚓ Södertälje Unique Trains", sodertalje_data['taglank'].nunique())
    
    st.divider()
    
    # Station selector
    station = st.selectbox(
        "Select Station",
        options=['Göteborg Skandiahamnen', 'Södertälje hamn'],
        format_func=lambda x: '🚢 Göteborg Skandiahamnen' if 'Skandia' in x else '⚓ Södertälje Hamn'
    )
    
    station_data = endpoint_df[endpoint_df['plats'] == station].copy()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Hourly Heatmap", "📈 Daily Pattern", "📅 Calendar View", "📋 Details"])
    
    with tab1:
        st.subheader(f"Hourly Traffic Heatmap - {station}")
        st.write("Shows train activity by hour and day of week. **Darker = more trains.**")
        
        # Create heatmap data: weekday x hour
        heatmap_data = station_data.groupby(['weekday', 'hour']).size().reset_index(name='count')
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(index='weekday', columns='hour', values='count').fillna(0)
        
        # Ensure all hours present
        for h in range(24):
            if h not in heatmap_pivot.columns:
                heatmap_pivot[h] = 0
        heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]
        
        # Ensure all weekdays present
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for wd in range(7):
            if wd not in heatmap_pivot.index:
                heatmap_pivot.loc[wd] = 0
        heatmap_pivot = heatmap_pivot.sort_index()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=weekday_names,
            colorscale='YlOrRd',
            hoverongaps=False,
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Trains: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Train Activity by Hour and Day - {station}",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight low-activity windows
        st.subheader("🟢 Low Activity Windows (Potential Capacity)")
        low_threshold = heatmap_pivot.values.mean() * 0.5
        
        low_windows = []
        for wd_idx, wd_name in enumerate(weekday_names):
            for h in range(24):
                if heatmap_pivot.iloc[wd_idx, h] <= low_threshold:
                    low_windows.append(f"{wd_name} {h:02d}:00")
        
        if low_windows:
            n_cols = 4
            cols = st.columns(n_cols)
            for i, window in enumerate(low_windows[:20]):
                cols[i % n_cols].write(f"✅ {window}")
            if len(low_windows) > 20:
                st.write(f"... and {len(low_windows) - 20} more low-activity windows")
    
    with tab2:
        st.subheader(f"Average Daily Traffic Pattern - {station}")
        
        # Separate arrivals and departures
        col1, col2 = st.columns(2)
        
        with col1:
            arrivals = station_data[station_data['riktningny'] == 'Ankomst']
            hourly_arrivals = arrivals.groupby('hour').size().reset_index(name='count')
            
            fig_arr = go.Figure()
            fig_arr.add_trace(go.Bar(
                x=hourly_arrivals['hour'],
                y=hourly_arrivals['count'],
                name='Arrivals',
                marker_color='#2E86AB'
            ))
            fig_arr.update_layout(
                title="Arrivals by Hour",
                xaxis_title="Hour",
                yaxis_title="Number of Trains",
                height=300,
                xaxis=dict(dtick=2)
            )
            st.plotly_chart(fig_arr, use_container_width=True)
        
        with col2:
            departures = station_data[station_data['riktningny'] == 'Avgång']
            hourly_departures = departures.groupby('hour').size().reset_index(name='count')
            
            fig_dep = go.Figure()
            fig_dep.add_trace(go.Bar(
                x=hourly_departures['hour'],
                y=hourly_departures['count'],
                name='Departures',
                marker_color='#E94F37'
            ))
            fig_dep.update_layout(
                title="Departures by Hour",
                xaxis_title="Hour",
                yaxis_title="Number of Trains",
                height=300,
                xaxis=dict(dtick=2)
            )
            st.plotly_chart(fig_dep, use_container_width=True)
        
        # Combined timeline
        st.subheader("Combined Hourly Pattern")
        
        hourly_combined = station_data.groupby(['hour', 'riktningny']).size().reset_index(name='count')
        
        fig_combined = go.Figure()
        for direction, color in [('Ankomst', '#2E86AB'), ('Avgång', '#E94F37')]:
            dir_data = hourly_combined[hourly_combined['riktningny'] == direction]
            fig_combined.add_trace(go.Scatter(
                x=dir_data['hour'],
                y=dir_data['count'],
                mode='lines+markers',
                name='Arrivals' if direction == 'Ankomst' else 'Departures',
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))
        
        fig_combined.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Trains",
            height=350,
            xaxis=dict(dtick=2, range=[-0.5, 23.5]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_combined, use_container_width=True)
    
    with tab3:
        st.subheader(f"Monthly Traffic - {station}")
        
        station_data['month'] = station_data['plandatumtid'].dt.to_period('M').astype(str)
        monthly = station_data.groupby(['month', 'riktningny']).size().reset_index(name='count')
        
        fig_monthly = go.Figure()
        for direction, color in [('Ankomst', '#2E86AB'), ('Avgång', '#E94F37')]:
            dir_data = monthly[monthly['riktningny'] == direction]
            fig_monthly.add_trace(go.Bar(
                x=dir_data['month'],
                y=dir_data['count'],
                name='Arrivals' if direction == 'Ankomst' else 'Departures',
                marker_color=color
            ))
        
        fig_monthly.update_layout(
            barmode='group',
            xaxis_title="Month",
            yaxis_title="Number of Trains",
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Daily distribution for selected month
        st.subheader("Daily Distribution")
        months = sorted(station_data['month'].unique())
        selected_month = st.selectbox("Select Month", options=months, index=len(months)//2)
        
        month_data = station_data[station_data['month'] == selected_month]
        month_data['day'] = month_data['plandatumtid'].dt.day
        
        daily = month_data.groupby(['day', 'riktningny']).size().reset_index(name='count')
        
        fig_daily = go.Figure()
        for direction, color in [('Ankomst', '#2E86AB'), ('Avgång', '#E94F37')]:
            dir_data = daily[daily['riktningny'] == direction]
            fig_daily.add_trace(go.Scatter(
                x=dir_data['day'],
                y=dir_data['count'],
                mode='lines+markers',
                name='Arrivals' if direction == 'Ankomst' else 'Departures',
                line=dict(color=color),
                marker=dict(size=6)
            ))
        
        fig_daily.update_layout(
            title=f"Daily Traffic - {selected_month}",
            xaxis_title="Day of Month",
            yaxis_title="Number of Trains",
            height=300
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab4:
        st.subheader(f"Recent Movements - {station}")
        
        # Show recent movements
        recent = station_data.sort_values('plandatumtid', ascending=False).head(100)
        
        display_df = recent[['plandatumtid', 'riktningny', 'tagnr', 'taglank', 'tidsavvikelse']].copy()
        display_df['plandatumtid'] = display_df['plandatumtid'].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Planned Time', 'Direction', 'Train Nr', 'Journey ID', 'Delay (min)']
        display_df['Direction'] = display_df['Direction'].map({'Ankomst': '🔵 Arrival', 'Avgång': '🔴 Departure'})
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Stats
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Direction:**")
            st.write(station_data.groupby('riktningny').size().to_frame('Count'))
        
        with col2:
            st.write("**By Operator (Avtalspart):**")
            operator_counts = station_data.groupby('avtalspart').size().sort_values(ascending=False)
            st.write(operator_counts.head(10).to_frame('Count'))


def show_april10_capacity(traffic_df, station_order):
    """Show April 10 overnight traffic - ALL trains at route stations."""
    st.header("🌙 April 10 Overnight - All Traffic")
    
    st.write("""
    **Time Window:** April 10, 2024 22:00 → April 11, 2024 02:00 (4 hours around midnight)
    
    Shows **ALL trains** passing through stations on our route during this window.
    - 🔵 **Blue lines** = Our route trains (202404108396 TO Södertälje)
    - ⚫ **Gray lines** = Other trains using the same infrastructure
    
    Look for **gaps/white space** = potential capacity for new train paths!
    """)
    
    # Create station to Y position mapping
    station_to_y = dict(zip(station_order['station'], station_order['order']))
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trains", traffic_df['taglank'].nunique())
    with col2:
        st.metric("Total Records", len(traffic_df))
    with col3:
        st.metric("Stations", len(station_order))
    with col4:
        # Our route train
        our_train = '202404108396'
        is_our_train = traffic_df['taglank'].astype(str) == our_train
        st.metric("Our Train Records", is_our_train.sum())
    
    st.divider()
    
    # Build the time-space diagram
    fig = go.Figure()
    
    # Base time for X-axis (midnight April 11)
    base_time = pd.Timestamp('2024-04-11 00:00:00')
    
    # Process each train
    our_train_id = '202404108396'
    train_colors = {}
    
    for taglank in traffic_df['taglank'].unique():
        train_data = traffic_df[traffic_df['taglank'] == taglank].sort_values('plandatumtid')
        
        if len(train_data) < 2:
            continue
        
        # Calculate time as hours from midnight (negative for before midnight)
        times = train_data['plandatumtid']
        x_vals = [(t - base_time).total_seconds() / 3600 for t in times]
        
        # Get Y positions
        y_vals = []
        stations = []
        for _, row in train_data.iterrows():
            station = row['plats']
            if station in station_to_y:
                y_vals.append(station_to_y[station])
                stations.append(station)
        
        if len(y_vals) < 2:
            continue
        
        # Determine if this is our route train
        is_our = str(taglank) == our_train_id
        
        if is_our:
            color = '#2E86AB'  # Blue for our train
            width = 4
            opacity = 1.0
            name = f"🔵 {taglank} (Our Route)"
        else:
            color = '#888888'  # Gray for other trains
            width = 1.5
            opacity = 0.5
            name = str(taglank)
        
        # Create hover text
        hover_texts = []
        for i, (x, y, s) in enumerate(zip(x_vals[:len(y_vals)], y_vals, stations)):
            h = int(x) if x >= 0 else int(x) + 24
            m = int(abs(x % 1) * 60)
            day = "Apr 11" if x >= 0 else "Apr 10"
            hover_texts.append(f"<b>{taglank}</b><br>{s}<br>{day} {h:02d}:{m:02d}")
        
        fig.add_trace(go.Scatter(
            x=x_vals[:len(y_vals)],
            y=y_vals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=width),
            marker=dict(size=4 if is_our else 2, color=color),
            opacity=opacity,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=is_our
        ))
    
    # Update layout
    fig.update_layout(
        title="All Train Traffic - April 10 22:00 to April 11 02:00",
        xaxis_title="Hours from Midnight (April 11)",
        yaxis_title="Station",
        height=800,
        xaxis=dict(
            range=[-2.5, 2.5],
            dtick=0.5,
            ticktext=[f"{int(h+24) if h < 0 else int(h):02d}:00" for h in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]],
            tickvals=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='red',
            zerolinewidth=2
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(station_order))),
            ticktext=station_order['station'].tolist(),
            autorange='reversed',  # Skandiahamnen at top
            gridcolor='lightgray'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='closest'
    )
    
    # Add midnight line annotation
    fig.add_annotation(
        x=0, y=-0.5,
        text="← Apr 10 | Midnight | Apr 11 →",
        showarrow=False,
        yref='paper',
        font=dict(color='red', size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show train list
    st.divider()
    st.subheader("📋 Trains in Window")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Our Route Train:**")
        our_data = traffic_df[traffic_df['taglank'].astype(str) == our_train_id]
        if len(our_data) > 0:
            st.write(f"- **{our_train_id}** - {len(our_data)} station records")
            first = our_data['plandatumtid'].min()
            last = our_data['plandatumtid'].max()
            st.write(f"  - First: {first.strftime('%H:%M')} at {our_data[our_data['plandatumtid'] == first]['plats'].iloc[0]}")
            st.write(f"  - Last: {last.strftime('%H:%M')} at {our_data[our_data['plandatumtid'] == last]['plats'].iloc[0]}")
    
    with col2:
        st.write("**Other Trains (sample):**")
        other_trains = traffic_df[traffic_df['taglank'].astype(str) != our_train_id]['taglank'].unique()[:10]
        for t in other_trains:
            count = len(traffic_df[traffic_df['taglank'] == t])
            st.write(f"- {t} ({count} records)")
        if len(traffic_df['taglank'].unique()) > 11:
            st.write(f"... and {len(traffic_df['taglank'].unique()) - 11} more trains")


def main():
    """Main application."""
    
    # Sidebar
    st.sidebar.title("🚂 Södertälje Capacity")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        options=['Overview', 'Time-Space Diagram', 'April 10 Capacity', 'Endpoint Traffic', 'Path Analysis', 'Slot Finder'],
        format_func=lambda x: {
            'Overview': '📊 Route Overview',
            'Time-Space Diagram': '📈 Time-Space Diagram',
            'April 10 Capacity': '🌙 April 10 Overnight',
            'Endpoint Traffic': '🚉 Endpoint Traffic',
            'Path Analysis': '🛤️ Path Analysis',
            'Slot Finder': '🔍 Find Slots'
        }.get(x, x)
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this tool**
    
    Analyze railway capacity between 
    Göteborg Skandiahamnen and 
    Södertälje Hamn.
    
    Find available timeslots for 
    scheduling new freight trains.
    
    **April 10 Overnight:**
    See ALL trains using our route 
    stations during the overnight 
    window to find capacity gaps.
    
    **Time-Space Diagram:**
    Standard railway planning view 
    with stations on Y-axis and 
    time on X-axis.
    """)
    
    # Load data
    try:
        base_data = load_base_data()
    except Exception as e:
        st.error(f"Error loading base data: {e}")
        st.info("Please run the analysis scripts first (01-04) to generate the required data files.")
        return
    
    # Show selected page
    if page == 'Overview':
        show_overview(base_data)
    elif page == 'Time-Space Diagram':
        try:
            with st.spinner("Loading train data..."):
                route_delay_df = load_route_delay_data()
            show_time_space_diagram_page(base_data, route_delay_df)
        except Exception as e:
            st.error(f"Error loading delay data: {e}")
            st.info("Make sure route_delay_data.csv exists in the analysis folder.")
    elif page == 'April 10 Capacity':
        try:
            with st.spinner("Loading April 10 traffic..."):
                traffic_df, station_order = load_april10_traffic()
            show_april10_capacity(traffic_df, station_order)
        except Exception as e:
            st.error(f"Error loading April 10 data: {e}")
            st.info("Make sure april10_overnight_traffic.csv exists in the analysis folder.")
    elif page == 'Endpoint Traffic':
        try:
            with st.spinner("Loading endpoint movements..."):
                endpoint_df = load_endpoint_movements()
            show_endpoint_traffic(endpoint_df)
        except Exception as e:
            st.error(f"Error loading endpoint data: {e}")
            st.info("Make sure endpoint_movements.csv exists in the analysis folder.")
    elif page == 'Path Analysis':
        show_path_analysis(base_data)
    elif page == 'Slot Finder':
        show_slot_finder(base_data)


if __name__ == "__main__":
    main()
