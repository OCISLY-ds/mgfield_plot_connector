import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hapiclient import hapi
import numpy as np
import math
import argparse
import os
from datetime import datetime
import json
import urllib.request
import csv
import textwrap
import map_plot

FILE_1 = 'rawmeasurementsstation1.csv.gz'
FILE_2 = 'rawmeasurementsstation2.csv.gz'  # Zweite Datei hier eintragen
DEFAULT_IAGA_CODES = ['WNG']        # Referenzstationen (z.B. Wingst)
VALUES_TO_AVERAGE = 120             # 120 Werte = 60 Sekunden bei 0.5s Intervall
ENABLE_AVERAGING = True
OUTPUT_DIR = 'output'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- KP INDEX FUNCTIONS ---
def __checkdate__(starttime, endtime):
    if starttime > endtime:
        raise NameError("Error! Start time must be before or equal to end time")
    return True

def __checkIndex__(index):
    if index not in ['Kp', 'ap', 'Ap', 'Cp', 'C9', 'Hp30', 'Hp60', 'ap30', 'ap60', 'SN', 'Fobs', 'Fadj']:
        raise IndexError("Error! Wrong index parameter! \nAllowed are only the string parameter: 'Kp', 'ap', 'Ap', 'Cp', 'C9', 'Hp30', 'Hp60', 'ap30', 'ap60', 'SN', 'Fobs', 'Fadj'")
    return True

def __checkstatus__(status):
    if status not in ['all', 'def']:
        raise IndexError("Error! Wrong option parameter! \nAllowed are only the string parameter: 'def'")
    return True

def __addstatus__(url, status):
    if status == 'def':
        url = url + '&status=def'
    return url

def getKpindex(starttime, endtime, index, status='all'):
    """
    Download 'Kp', 'ap', 'Ap', 'Cp', 'C9', 'Hp30', 'Hp60', 'ap30', 'ap60', 'SN', 'Fobs' or 'Fadj' index data from kp.gfz-potsdam.de
    date format for starttime and endtime is 'yyyy-mm-dd' or 'yyyy-mm-ddTHH:MM:SSZ'
    """
    result_t = 0
    result_index = 0
    result_s = 0

    # If only date is given, append time
    if len(starttime) == 10: starttime += 'T00:00:00Z'
    if len(endtime) == 10: endtime += 'T23:59:00Z'

    try:
        d1 = datetime.strptime(starttime, '%Y-%m-%dT%H:%M:%SZ')
        d2 = datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%SZ')

        __checkdate__(d1, d2)
        __checkIndex__(index)
        __checkstatus__(status)

        time_string = "start=" + d1.strftime('%Y-%m-%dT%H:%M:%SZ') + "&end=" + d2.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = 'https://kp.gfz-potsdam.de/app/json/?' + time_string + "&index=" + index
        if index not in ['Hp30', 'Hp60', 'ap30', 'ap60', 'Fobs', 'Fadj']:
            url = __addstatus__(url, status)

        print(f"Fetching Kp index from: {url}")
        webURL = urllib.request.urlopen(url)
        binary = webURL.read()
        text = binary.decode('utf-8')

        try:
            data = json.loads(text)
            result_t = tuple(data["datetime"])
            result_index = tuple(data[index])
            if index not in ['Hp30', 'Hp60', 'ap30', 'ap60', 'Fobs', 'Fadj']:
                result_s = tuple(data["status"])
        except:
            print("Error parsing JSON response:")
            print(text)

    except Exception as er:
        print(f"Error fetching Kp index: {er}")
    
    return result_t, result_index, result_s

def fetch_iaga_data(station_code, start_time, stop_time):
    """Fetches data from Intermagnet (HAPI) for a station."""
    print(f"\n--- Fetching data for {station_code} from Intermagnet ---")
    print(f"Time range: {start_time} to {stop_time}")
    
    server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
    dataset = f'{station_code.lower()}/best-avail/PT1M/xyzf'
    parameters = 'Field_Vector'
    opts = {'logging': True, 'usecache': True}
    
    try:
        data, meta = hapi(server, dataset, parameters, start_time, stop_time, **opts)
        
        if isinstance(data, np.ndarray) and data.ndim == 1:
            # Extract data
            timestamps = [pd.to_datetime(item[0].decode('utf-8')) for item in data]
            vectors = np.array([item[1] for item in data])
            
            # Calculate magnitude
            magnitudes = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in vectors]
            
            # Create DataFrame
            # Include X, Y, Z components for axis comparison
            df = pd.DataFrame({
                'time_utc': timestamps,
                'x': vectors[:, 0],
                'y': vectors[:, 1],
                'z': vectors[:, 2],
                'magnitude': magnitudes
            })

            # Fix: Ensure Timezone-Awareness (UTC)
            if df['time_utc'].dt.tz is None:
                df['time_utc'] = df['time_utc'].dt.tz_localize('UTC')
            else:
                df['time_utc'] = df['time_utc'].dt.tz_convert('UTC')
            
            # Filter (similar to IAGA_API.py)
            df = df[df['magnitude'] <= 160000]
            
            df = df.sort_values(by='time_utc')
            
            # Save as CSV
            csv_name = f'iaga_data_{station_code}.csv'
            df.to_csv(csv_name, index=False)
            print(f"  Data saved to {csv_name}")
            
            print(f"  Data points received: {len(df)}")
            return df
        else:
            print("Format error in HAPI data.")
            return None
            
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return None

def load_and_process(filename):
    """Loads the CSV, calculates magnitude and performs averaging."""
    print(f"\n--- Processing {filename} ---")
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found.")
        return None
    
    return process_dataframe(df)

def process_dataframe(df):
    """Processes a raw dataframe (Magnitude, Averaging)."""
    if df is None or df.empty:
        return None

    # Convert timestamps and sort
    # Check if time_utc is already datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_utc']):
        df['time_utc'] = pd.to_datetime(df['time_utc'])
    
    # Fix: Ensure Timezone-Awareness (UTC)
    if df['time_utc'].dt.tz is None:
        df['time_utc'] = df['time_utc'].dt.tz_localize('UTC')
    else:
        df['time_utc'] = df['time_utc'].dt.tz_convert('UTC')

    df = df.sort_values(by='time_utc')

    # Calculate Magnitude
    # Formula: sqrt((x-out)^2 + (y-out)^2 + (z-out)^2)
    # Ensure columns exist
    req_cols = ['x_value', 'y_value', 'z_value', 'out_value']
    if not all(col in df.columns for col in req_cols):
        print(f"Error: Missing columns in dataframe. Required: {req_cols}")
        return None

    df['corrected_x'] = df['x_value'] - df['out_value']
    df['corrected_y'] = df['y_value'] - df['out_value']
    df['corrected_z'] = df['z_value'] - df['out_value']
    df['magnitude'] = (df['corrected_x']**2 + df['corrected_y']**2 + df['corrected_z']**2)**0.5

    # Averaging
    if ENABLE_AVERAGING:
        print(f"  Calculating average over {VALUES_TO_AVERAGE} values...")
        # Reset index for clean grouping
        df = df.reset_index(drop=True)
        
        cols_to_mean = ['x_value', 'y_value', 'z_value', 'out_value', 'corrected_x', 'corrected_y', 'corrected_z', 'magnitude']
        
        # Group by index blocks
        # 1. Timestamp: First value of the block
        time_series = df['time_utc'].groupby(df.index // VALUES_TO_AVERAGE).first()
        # 2. Data: Mean of the block
        data_mean = df[cols_to_mean].groupby(df.index // VALUES_TO_AVERAGE).mean()
        
        df = data_mean
        df['time_utc'] = time_series
        print(f"  Data points after averaging: {len(df)}")
    
    return df

def run_plotting_pipeline(df1_raw, df2_raw, start_time_arg=None, end_time_arg=None, iaga_codes=None):
    """
    Main entry point for external calls (e.g. from main.py).
    df1_raw, df2_raw: Raw pandas DataFrames (like from DB).
    """
    print("\n=== Starting Plotting Pipeline ===")

    # Use default IAGA codes if none provided
    if iaga_codes is None:
        iaga_codes = DEFAULT_IAGA_CODES
    
    # 1. Process Data
    print("Pre-processing Station 1 data...")
    df1 = process_dataframe(df1_raw)
    print("Pre-processing Station 2 data...")
    df2 = process_dataframe(df2_raw)

    # Determine fixed time range from arguments for data fetching
    start_str = None
    stop_str = None
    if start_time_arg and end_time_arg:
         # Convert input strings to datetime to ensure correct format
        try:
             s_dt = pd.to_datetime(start_time_arg)
             e_dt = pd.to_datetime(end_time_arg)
             start_str = s_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
             stop_str = e_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except:
             pass

    # Fetch Hp30 Index (using Hp30 instead of Kp as requested)
    hp_df = None
    if start_str and stop_str:
        print("\n--- Fetching Hp30 Index ---")
        # Fetch Hp30 index
        # Note: getKpindex signature is (starttime, endtime, index, status)
        res_t, res_index, res_s = getKpindex(start_str, stop_str, 'Hp30')
        
        if res_t and res_index:
            hp_df = pd.DataFrame({'time_utc': res_t, 'Hp30': res_index})
            # if res_s: hp_df['status'] = res_s # Hp30 might not have status in same way, check API if needed
            
            # Convert time to datetime and localize to UTC
            hp_df['time_utc'] = pd.to_datetime(hp_df['time_utc'])
            if hp_df['time_utc'].dt.tz is None:
                hp_df['time_utc'] = hp_df['time_utc'].dt.tz_localize('UTC')
            else:
                hp_df['time_utc'] = hp_df['time_utc'].dt.tz_convert('UTC')
            
            # Save Hp data to CSV
            # Overwrite previous file
            hp_csv_name = os.path.join(OUTPUT_DIR, f'Hp30_index.csv')
            hp_df.to_csv(hp_csv_name, index=False)
            print(f"Hp30 index data saved to {hp_csv_name}")
        else:
            print("No Hp30 index data found.")

    # Loop over IAGA codes to fetch reference data
    ref_datasets = {}
    for iaga_code in iaga_codes:
         if start_str and stop_str:
            df_ref = fetch_iaga_data(iaga_code, start_str, stop_str)
            if df_ref is not None:
                ref_datasets[iaga_code] = df_ref

    # 3. Create Plots
    # User Requirement: "Fit finden ... immer zu der WNG Intermagnet station passen"
    # We use 'WNG' as the primary reference for calibration if available.
    primary_ref_code = 'WNG'
    primary_ref_df = ref_datasets.get(primary_ref_code)
    
    # If WNG not loaded but others are, take the first one?
    if primary_ref_df is None and ref_datasets:
        primary_ref_code = list(ref_datasets.keys())[0]
        primary_ref_df = ref_datasets[primary_ref_code]
        print(f"Warning: WNG not found. Using {primary_ref_code} for calibration.")

    # Plot for Station 1
    if df1 is not None:
        plot_station_combined(df1, "Station 1", hp_df, ref_datasets, calibration_ref_df=primary_ref_df, calibration_ref_name=primary_ref_code)
    
    # Plot for Station 2
    if df2 is not None:
        plot_station_combined(df2, "my BUE Station 2", hp_df, ref_datasets, calibration_ref_df=primary_ref_df, calibration_ref_name=primary_ref_code)

    # 3b. Generate XYZ Plots
    print("\n--- Generating individual XYZ component plots ---")
    if df1 is not None:
        plot_xyz_components(df1, "Station 1")
    if df2 is not None:
        plot_xyz_components(df2, "Station 2")
    for code, df in ref_datasets.items():
        if df is not None:
            plot_xyz_components(df, f"{code} (Intermagnet)")

    # 4. Generate Station Map
    if iaga_codes:
        map_plot.create_map(iaga_codes)

    print("\nDone! All plots have been saved.")

def calculate_linear_fit(station_df, ref_df):
    """
    Calculates the linear fit (y = m*x + c) to match station data to reference.
    Returns: m (slope), c (offset), correlation
    """
    # 1. Synchronize data (round to minutes)
    df_stat = station_df[['time_utc', 'magnitude']].copy()
    df_stat['time_utc'] = df_stat['time_utc'].dt.round('1min')
    df_stat = df_stat.groupby('time_utc').mean()
    
    df_ref = ref_df[['time_utc', 'magnitude']].copy()
    df_ref['time_utc'] = df_ref['time_utc'].dt.round('1min')
    df_ref = df_ref.groupby('time_utc').mean()
    
    # Inner Join for common time points
    merged = df_stat.join(df_ref, lsuffix='_stat', rsuffix='_ref', how='inner').dropna()
    
    if merged.empty:
        return 1.0, 0.0, 0.0 # Fallback

    x = merged['magnitude_stat']
    y = merged['magnitude_ref']
    
    # Linear Regression (y = m*x + c)
    m, c = np.polyfit(x, y, 1)
    
    # Calculate Correlation
    corr = x.corr(y)
    
    return m, c, corr

def plot_station_combined(station_df, station_name, hp_df, ref_datasets, calibration_ref_df=None, calibration_ref_name=None):
    """
    Creates a combined plot with:
    1. Station Magnitude (CALIBRATED/FITTED)
    2. Intermagnet Reference Data (WNG etc) - Plotted roughly on top to show match
    3. Hp30 Index 
    """
    print(f"\nCreating combined plot for {station_name}...")
    
    # Calibration Step
    calibrated_mag = station_df['magnitude']
    fit_info_str = "Raw Data (Uncalibrated)"
    
    if calibration_ref_df is not None:
        print(f"  -> Calibrating against {calibration_ref_name}...")
        m, c, corr = calculate_linear_fit(station_df, calibration_ref_df)
        
        if m != 1.0 or c != 0.0:
            calibrated_mag = station_df['magnitude'] * m + c
            fit_info_str = f"Calibrated to {calibration_ref_name}: y = {m:.4f}x + {c:.2f} nT | R = {corr:.4f}"
            print(f"     {fit_info_str}")
        else:
             print("     Calibration failed (no overlap), using raw data.")

    # Create figure with 2 subplots (Magnetometer Data and Hp30 Index)
    # Ratios: 3 for Data, 1 for Index
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    fig.suptitle(f'Analysis: G4 Severe Geomagnetic Storm 19 Jan, 2026\n({fit_info_str})', fontsize=14)

    # --- Plot 1: Magnetic Field Strength ---
    
    # 1. Station Data (Calibrated)
    # High contrast color: Blue-ish
    ax1.plot(station_df['time_utc'], calibrated_mag, label=f'{station_name} (Calibrated)', color='#0000FF', linewidth=1.5, alpha=0.9)
    
    # 2. Reference Data
    # High contrast color: distinct from Blue, e.g. Black or Deep Magenta
    for code, df_ref in ref_datasets.items():
        ax1.plot(df_ref['time_utc'], df_ref['magnitude'], label=f'{code} (Intermagnet)', color='#000000', linewidth=1.5, alpha=0.7, linestyle='-')
        
    ax1.set_ylabel('Magnetic Field strength (nT)')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Plot 2: Hp30 Index ---
    target_ax = ax1 # Fallback
    if hp_df is not None:
        # Color logic with custom RGB values (normalized to 0-1 for matplotlib)
        # Green: 117, 251, 76 -> (0.459, 0.984, 0.298)
        # Orange/Yellow: 255, 255, 85 -> (1.0, 1.0, 0.333)
        # Red: 234, 51, 35 -> (0.918, 0.2, 0.137)
        
        c_green  = (117/255, 251/255, 76/255)
        c_orange = (255/255, 255/255, 85/255)
        c_red    = (234/255, 51/255, 35/255)
        
        colors = []
        for val in hp_df['Hp30']:
            if val <= 3:
                colors.append(c_green)
            elif val <= 6:
                colors.append(c_orange)
            else:
                colors.append(c_red)
        
        # Hp30 - connect bars
        width = 1/48
        
        ax2.bar(hp_df['time_utc'], hp_df['Hp30'], width=width, color=colors, label='Hp30 Index', alpha=0.8, align='edge')
        
        ax2.set_ylabel('Hp30 Index')
        ax2.set_ylim(0, 9)
        ax2.set_yticks(range(10))
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Custom Legend for Colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c_green, label='0-3 (Quiet)'),
            Patch(facecolor=c_orange, label='3-6 (Moderate)'),
            Patch(facecolor=c_red, label='6-9 (Storm)')
        ]
        # Legend inside plot area
        ax2.legend(handles=legend_elements, loc='upper left')
        
        target_ax = ax2
    else:
        ax2.text(0.5, 0.5, 'No Hp30 Data available', ha='center', va='center')

    # Formatting Time Axis
    start_t = station_df['time_utc'].min()
    end_t = station_df['time_utc'].max()
    setup_date_axis(target_ax, start_t, end_t)
    
    # Add footer text with disclaimer (Year removed)
    if calibration_ref_name == 'WNG':
        disclaimer_body = ("The measurement data were adjusted to the baseline of the INTERMAGNET observatory Wingst (WNG) using linear regression. "
                           "Due to the spatial proximity (~25 km) and low model deviations (< 0.02%) by NASA, the values represent the local geomagnetic conditions with high accuracy, but are not independent absolute measurements.")
    else:
        # Fallback for other stations
        disclaimer_body = (f"The measurement data were adjusted to the baseline of the INTERMAGNET observatory {calibration_ref_name} using linear regression. "
                           "The values represent the geomagnetic variations, but are not independent absolute measurements.")

    full_footer = disclaimer_body
    # Increase width to reduce height, fits better in smaller margin
    wrapped_footer = textwrap.fill(full_footer, width=120)
    
    # Place at bottom with small margin
    fig.text(0.5, 0.01, wrapped_footer, ha='center', va='bottom', fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    # Adjust layout to make room for footer (bottom)
    # 0.09 reserves bottom 9% for footer
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    
    # Save with OVERWRITE (fixed filename)
    filename = f"{station_name.replace(' ', '_').lower()}_analysis.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path)
    print(f"Plot saved as: {out_path}")
    plt.close(fig)

def setup_date_axis(ax, start_time=None, end_time=None):
    """Formats the X-axis dynamically based on the time range."""
    
    if start_time is not None and end_time is not None:
        # Round to full hours for the axis limits
        s_floor = start_time.floor('h')
        e_ceil = end_time.ceil('h')
        
        # If the duration is very short, ensure we have at least 1 hour
        if s_floor == e_ceil:
             e_ceil = s_floor + pd.Timedelta(hours=1)

        ax.set_xlim(s_floor, e_ceil)
        
        duration_hours = (e_ceil - s_floor).total_seconds() / 3600
    else:
        duration_hours = 24 # Fallback

    # Dynamic intervals (more frequent)
    # User request: "more times printed", "minimum one hour", "adapt to width"
    if duration_hours <= 12:
        interval = 1 # Every hour
    elif duration_hours <= 24:
        interval = 2 # Every 2 hours
    elif duration_hours <= 48:
        interval = 3 # Every 3 hours
    elif duration_hours <= 72:
        interval = 6 # Every 6 hours
    else:
        interval = 12 # Every 12 hours

    # Generate ticks
    if start_time is not None and end_time is not None:
        freq = f'{interval}h'
        ticks = pd.date_range(start=s_floor, end=e_ceil, freq=freq)
        ax.set_xticks(ticks)
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))

    # Formatter: DD.MM. HH:MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M'))
    # Labels slightly rotated
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

def plot_individual(df, filename, kp_df=None):
    """Creates the detailed plot for a sensor, optionally with Kp index."""
    if df is None: return

    if kp_df is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18), sharex=True, gridspec_kw={'height_ratios': [3, 3, 3, 1]})
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    fig.suptitle(f'Analysis: G4 Severe Geomagnetic Storm 19 Jan 26', fontsize=16)

    # --- DIAGRAM 1: X, Y, Z (Centered) ---
    x_centered = df['x_value'] - df['x_value'].mean()
    y_centered = df['y_value'] - df['y_value'].mean()
    z_centered = df['z_value'] - df['z_value'].mean()

    ax1.plot(df['time_utc'], x_centered, label=f'X (Mean: {df["x_value"].mean():.2f})', color='red', linewidth=1, alpha=0.8)
    ax1.plot(df['time_utc'], y_centered, label=f'Y (Mean: {df["y_value"].mean():.2f})', color='green', linewidth=1, alpha=0.8)
    ax1.plot(df['time_utc'], z_centered, label=f'Z (Mean: {df["z_value"].mean():.2f})', color='blue', linewidth=1, alpha=0.8)
    
    ax1.set_ylabel('Deviation from Mean')
    ax1.set_title('Components X, Y, Z (Centered)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- DIAGRAM 2: Out Value ---
    ax2.plot(df['time_utc'], df['out_value'], label='Out Value', color='black', linewidth=1.5)
    ax2.set_ylabel('Output Value')
    ax2.set_title('Result Value (Out)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- DIAGRAM 3: Magnitude ---
    ax3.plot(df['time_utc'], df['magnitude'], label='Calculated Magnitude', color='purple', linewidth=1.5)
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Calculated Magnitude')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # --- DIAGRAM 4: Kp Index (Optional) ---
    target_ax = ax3
    if kp_df is not None:
        ax4.bar(kp_df['time_utc'], kp_df['Kp'], width=0.125, color='orange', label='Kp Index', alpha=0.7, align='edge')
        ax4.set_ylabel('Kp Index')
        ax4.set_ylim(0, 9)
        ax4.set_yticks(range(10))
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend(loc='upper right')
        target_ax = ax4

    # Formatting
    start_t = df['time_utc'].min()
    end_t = df['time_utc'].max()
    setup_date_axis(target_ax, start_t, end_t)
    plt.xlabel('Time (UTC)')
    plt.tight_layout()
    
    # Save
    base_name = filename.replace('.csv.gz', '').replace('.csv', '')
    out_name = os.path.join(OUTPUT_DIR, f'{base_name}_analysis_{TIMESTAMP}.png')
    print(f"Saving individual plot as: {out_name}")
    plt.savefig(out_name)

def plot_comparison(df1, name1, df2, name2, df_ref=None, name_ref=None, kp_df=None):
    """Creates a comparison plot only for the magnitude, optionally with Kp index."""
    if df1 is None and df2 is None:
        print("No data available for comparison.")
        return

    if kp_df is not None:
        fig, (ax, ax_kp) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    fig.suptitle('Comparison: Calculated Magnitude', fontsize=16)

    if df1 is not None:
        # Normalization: Set start value to 0
        mag1_norm = df1['magnitude'] - df1['magnitude'].iloc[0]
        ax.plot(df1['time_utc'], mag1_norm, label=f'{name1} (Start=0)', color='#1f77b4', linewidth=1.5, alpha=0.8)
    
    if df2 is not None:
        # Normalization: Set start value to 0
        mag2_norm = df2['magnitude'] - df2['magnitude'].iloc[0]
        ax.plot(df2['time_utc'], mag2_norm, label=f'{name2} (Start=0)', color='#d62728', linewidth=1.5, alpha=0.8)

    if df_ref is not None:
        # Normalization: Set start value to 0
        mag_ref_norm = df_ref['magnitude'] - df_ref['magnitude'].iloc[0]
        ax.plot(df_ref['time_utc'], mag_ref_norm, label=f'{name_ref} (Start=0)', color='black', linewidth=1.5, alpha=0.8)

    ax.set_ylabel('Magnitude Change (relative to start)')
    ax.set_title('Sensor Comparison (Normalized to start 0)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot Kp Index if available
    target_ax = ax
    if kp_df is not None:
        # Kp is 3-hour interval. Width should be approx 3 hours.
        # Convert width to days: 3h / 24h = 0.125
        ax_kp.bar(kp_df['time_utc'], kp_df['Kp'], width=0.125, color='orange', label='Kp Index', alpha=0.7, align='edge')
        ax_kp.set_ylabel('Kp Index')
        ax_kp.set_ylim(0, 9)
        ax_kp.set_yticks(range(10))
        ax_kp.grid(True, linestyle='--', alpha=0.6)
        ax_kp.legend(loc='upper right')
        target_ax = ax_kp

    # Determine time range
    all_times = []
    if df1 is not None: all_times.append(df1['time_utc'])
    if df2 is not None: all_times.append(df2['time_utc'])
    if df_ref is not None: all_times.append(df_ref['time_utc'])
    
    start_t = min(t.min() for t in all_times) if all_times else None
    end_t = max(t.max() for t in all_times) if all_times else None

    setup_date_axis(target_ax, start_t, end_t)
    plt.xlabel('Time (UTC)')
    plt.tight_layout()
    
    out_name = os.path.join(OUTPUT_DIR, f'comparison_magnitude_{name_ref}_{TIMESTAMP}.png')
    print(f"Saving comparison plot as: {out_name}")
    plt.savefig(out_name)

def plot_xyz_components(df, station_name):
    """Plots X, Y, Z components in one diagram for comparison (centered around mean)."""
    if df is None or df.empty:
        return

    print(f"Plotting XYZ components for {station_name}...")
    
    # Determine columns
    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        cols = {'x': 'x', 'y': 'y', 'z': 'z'}
    elif 'corrected_x' in df.columns:
        cols = {'x': 'corrected_x', 'y': 'corrected_y', 'z': 'corrected_z'}
    elif 'x_value' in df.columns:
        cols = {'x': 'x_value', 'y': 'y_value', 'z': 'z_value'}
    else:
        print(f"Could not find XYZ columns for {station_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_val = df[cols['x']]
    y_val = df[cols['y']]
    z_val = df[cols['z']]
    
    # Center around mean to allow visual comparison of variations
    x_mean = x_val.mean()
    y_mean = y_val.mean()
    z_mean = z_val.mean()
    
    ax.plot(df['time_utc'], x_val - x_mean, label=f'X (Mean={x_mean:.0f} nT)', color='#1f77b4', linewidth=1)
    ax.plot(df['time_utc'], y_val - y_mean, label=f'Y (Mean={y_mean:.0f} nT)', color='#ff7f0e', linewidth=1)
    ax.plot(df['time_utc'], z_val - z_mean, label=f'Z (Mean={z_mean:.0f} nT)', color='#2ca02c', linewidth=1)
    
    ax.set_title(f'{station_name} - XYZ Components (Centered around Mean)')
    ax.set_ylabel('Field Variation (nT)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    start_t = df['time_utc'].min()
    end_t = df['time_utc'].max()
    setup_date_axis(ax, start_t, end_t)
    
    plt.xlabel('Time (UTC)')
    plt.tight_layout()
    
    safe_name = station_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    out_path = os.path.join(OUTPUT_DIR, f'xyz_components_{safe_name}_{TIMESTAMP}.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved XYZ plot to {out_path}")
    plt.close(fig)

def calculate_and_print_correlation(datasets):
    """
    Calculates the correlation matrix between the datasets.
    datasets: Dictionary {'Name': DataFrame}
    """
    print("\n--- Calculating correlations ---")
    
    # Create a combined DataFrame for correlation
    combined_df = pd.DataFrame()
    
    for name, df in datasets.items():
        if df is None or df.empty:
            continue
            
        # Create copy to avoid modifying original
        temp_df = df[['time_utc', 'magnitude']].copy()
        
        # Round timestamps to full minutes for better matching
        temp_df['time_utc'] = temp_df['time_utc'].dt.round('1min')
        
        # Remove duplicates by rounding (take mean)
        temp_df = temp_df.groupby('time_utc').mean()
        
        # Rename column
        temp_df = temp_df.rename(columns={'magnitude': name})
        
        if combined_df.empty:
            combined_df = temp_df
        else:
            # Outer join to keep as much data as possible
            combined_df = combined_df.join(temp_df, how='outer')
    
    if combined_df.empty:
        print("No common data found for correlation.")
        return

    # Calculate correlation matrix
    corr_matrix = combined_df.corr()
    
    print("Correlation Matrix (Pearson):")
    print(corr_matrix)
    
    # Save as CSV
    csv_name = os.path.join(OUTPUT_DIR, f'correlation_matrix_{TIMESTAMP}.csv')
    corr_matrix.to_csv(csv_name)
    print(f"Table saved as '{csv_name}'")
    
    # Optional: Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.title('Correlation Matrix')
    
    # Create mask for diagonal (so it stays white)
    mask = np.eye(len(corr_matrix), dtype=bool)
    plot_data = corr_matrix.copy()
    plot_data[mask] = np.nan  # NaN is not drawn by default (white/transparent)
    
    # Set background color of axes to white
    plt.gca().set_facecolor('white')
    
    plt.imshow(plot_data, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Write values into cells
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i == j: continue # Leave diagonal empty
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}', 
                     ha='center', va='center', color='black')
            
    plt.tight_layout()
    heatmap_name = os.path.join(OUTPUT_DIR, f'correlation_heatmap_{TIMESTAMP}.png')
    plt.savefig(heatmap_name)
    print(f"Heatmap saved as '{heatmap_name}'")

def optimize_and_plot_fit(station_df, station_name, ref_df, ref_name, color, kp_df=None):
    """
    Calculates the optimal scaling factor and offset (Linear Regression),
    to fit the station to the reference.
    """
    print(f"\n--- Optimizing {station_name} vs {ref_name} ---")
    
    if station_df is None or ref_df is None:
        print("Data missing for optimization.")
        return

    # 1. Synchronize data (round to minutes)
    # Use copies to avoid modifying original DataFrames
    df_stat = station_df[['time_utc', 'magnitude']].copy()
    df_stat['time_utc'] = df_stat['time_utc'].dt.round('1min')
    # Mean per minute
    df_stat = df_stat.groupby('time_utc').mean()
    
    df_ref = ref_df[['time_utc', 'magnitude']].copy()
    df_ref['time_utc'] = df_ref['time_utc'].dt.round('1min')
    df_ref = df_ref.groupby('time_utc').mean()
    
    # Inner Join for common time points
    merged = df_stat.join(df_ref, lsuffix='_stat', rsuffix='_ref', how='inner').dropna()
    
    if merged.empty:
        print("No overlapping data points found.")
        return

    x = merged['magnitude_stat']
    y = merged['magnitude_ref']
    
    # Calculate correlation
    correlation = x.corr(y)
    
    # 2. Linear Regression (y = m*x + c)
    # m = Slope (Scaling Factor), c = Offset
    m, c = np.polyfit(x, y, 1)
    
    print(f"Optimal fit found:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Factor (Multiplier): {m:.5f}")
    print(f"  Offset (Shift):  {c:.5f} nT")
    print(f"  Formula: {ref_name} ≈ {m:.5f} * {station_name} + {c:.5f}")
    
    # 3. Plot
    if kp_df is not None:
        fig, (ax, ax_kp) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        
    fig.suptitle(f'Comparison {station_name} and {ref_name}', fontsize=16)
    
    # Apply formula to original data (high resolution)
    calibrated_mag = station_df['magnitude'] * m + c
    
    # Plot reference
    ax.plot(ref_df['time_utc'], ref_df['magnitude'], label=f'{ref_name} (Original)', color='#1f77b4', linewidth=2, alpha=0.8)
    
    # Plot calibrated station
    ax.plot(station_df['time_utc'], calibrated_mag, label=f'{station_name} (Calibrated)', color=color, linewidth=1, alpha=0.8)
    
    ax.set_ylabel('Magnetic Field Strength (nT)')
    ax.set_title(f'Fit: y = {m:.4f}x + {c:.2f} | Correlation: {correlation:.4f}')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Kp Index if available
    target_ax = ax
    if kp_df is not None:
        ax_kp.bar(kp_df['time_utc'], kp_df['Kp'], width=0.125, color='orange', label='Kp Index', alpha=0.7, align='edge')
        ax_kp.set_ylabel('Kp Index')
        ax_kp.set_ylim(0, 9)
        ax_kp.set_yticks(range(10))
        ax_kp.grid(True, linestyle='--', alpha=0.6)
        ax_kp.legend(loc='upper right')
        target_ax = ax_kp
    
    start_t = min(station_df['time_utc'].min(), ref_df['time_utc'].min())
    end_t = max(station_df['time_utc'].max(), ref_df['time_utc'].max())
    setup_date_axis(target_ax, start_t, end_t)

    # --- ZOOM INSET ---
    # Define zoom range: 2025-11-13 00:00:00 to 15:00:00
    z_start = pd.Timestamp('2025-11-13 00:00:00').tz_localize('UTC')
    z_end = pd.Timestamp('2025-11-13 10:00:00').tz_localize('UTC')

    # Check if we have data in this range
    if end_t > z_start and start_t < z_end:
        # Position: x, y, width, height (Bottom Right, raised to avoid overlap)
        axins = ax.inset_axes([0.625, 0.1, 0.35, 0.35]) 
        axins.set_facecolor('white')
        
        # Plot on inset
        axins.plot(ref_df['time_utc'], ref_df['magnitude'], color='#1f77b4', linewidth=2, alpha=0.8)
        axins.plot(station_df['time_utc'], calibrated_mag, color=color, linewidth=1, alpha=0.8)
        
        # Set limits
        axins.set_xlim(z_start, z_end)
        
        # Determine Y limits for the zoom window
        mask_ref = (ref_df['time_utc'] >= z_start) & (ref_df['time_utc'] <= z_end)
        mask_stat = (station_df['time_utc'] >= z_start) & (station_df['time_utc'] <= z_end)
        
        y_vals = []
        if mask_ref.any(): y_vals.extend(ref_df.loc[mask_ref, 'magnitude'])
        if mask_stat.any(): y_vals.extend(calibrated_mag[mask_stat])
        
        if y_vals:
            y_min_z = min(y_vals)
            y_max_z = max(y_vals)
            margin = (y_max_z - y_min_z) * 0.1 if y_max_z != y_min_z else 10
            axins.set_ylim(y_min_z - margin, y_max_z + margin)
            
        # Setup date axis for inset (simplified)
        axins.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(axins.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
        axins.grid(True, linestyle=':', alpha=0.5)
        
        ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, f'calibration_{station_name.replace(" ", "")}_{ref_name}_{TIMESTAMP}.png')
    plt.savefig(filename)
    print(f"Calibration plot saved as: {filename}")

    # --- DETAIL VIEW (Zoom) ---
    # Fixed range: Mean +/- 150 nT
    # Use mean of calibrated data as center
    mean_val = calibrated_mag.mean()
    y_min = mean_val - 350
    y_max = mean_val + 100
    
    ax.set_ylim(y_min, y_max)
    
    ax.set_title(f'Fit (Detail): y = {m:.4f}x + {c:.2f} | Correlation: {correlation:.4f}')
    
    filename_detail = f'calibration_detailed_{station_name.replace(" ", "")}.png'
    plt.savefig(filename_detail)
    print(f"Detail plot saved as: {filename_detail}")
    
    # Close figure to avoid display
    plt.close(fig)

# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    # Argumente parsen
    parser = argparse.ArgumentParser(description='Plot magnetic field data.')
    parser.add_argument('--start', type=str, help='Startzeit (z.B. 2023-10-27T00:00:00)')
    parser.add_argument('--end', type=str, help='Endzeit (z.B. 2023-10-28T00:00:00)')
    parser.add_argument('--iaga', nargs='+', default=DEFAULT_IAGA_CODES, help='Liste der IAGA Codes (z.B. WNG NGK)')
    args = parser.parse_args()

    # Update DEFAULT_IAGA_CODES if provided via args
    if args.iaga:
        DEFAULT_IAGA_CODES = args.iaga

    # 1. Daten laden
    df1 = load_and_process(FILE_1)
    df2 = load_and_process(FILE_2)

    # Apply user defined filtering (if provided)
    user_start = pd.to_datetime(args.start).tz_localize('UTC') if args.start else None
    user_end = pd.to_datetime(args.end).tz_localize('UTC') if args.end else None

    if user_start:
        print(f"Filtering data from: {user_start}")
        if df1 is not None: df1 = df1[df1['time_utc'] >= user_start]
        if df2 is not None: df2 = df2[df2['time_utc'] >= user_start]
    
    if user_end:
        print(f"Filtering data to: {user_end}")
        if df1 is not None: df1 = df1[df1['time_utc'] <= user_end]
        if df2 is not None: df2 = df2[df2['time_utc'] <= user_end]

    # NEW: Synchronize time ranges (Intersection)
    if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
        print("\n--- Synchronizing time ranges (Intersection) ---")
        # Start: The later of the two start dates
        start_common = max(df1['time_utc'].min(), df2['time_utc'].min())
        # End: The earlier of the two end dates
        end_common = min(df1['time_utc'].max(), df2['time_utc'].max())
        
        print(f"Common time range: {start_common} to {end_common}")
        
        if start_common < end_common:
            # Filter to common range
            df1 = df1[(df1['time_utc'] >= start_common) & (df1['time_utc'] <= end_common)]
            df2 = df2[(df2['time_utc'] >= start_common) & (df2['time_utc'] <= end_common)]
            print(f"Data points after synchronization -> Station 1: {len(df1)}, Station 2: {len(df2)}")
        else:
            print("Warning: No time overlap between files!")
    elif (df1 is None or df1.empty) and (df2 is None or df2.empty):
        print("No data available after filtering.")
        exit()

    # Determine time range across all loaded files
    all_timestamps = []
    if df1 is not None and not df1.empty: all_timestamps.append(df1['time_utc'])
    if df2 is not None and not df2.empty: all_timestamps.append(df2['time_utc'])
    
    start_str = None
    stop_str = None

    if all_timestamps:
        # We take the minimum of start and maximum of end of all files
        start_t = min(t.min() for t in all_timestamps)
        stop_t = max(t.max() for t in all_timestamps)
        
        # Format for HAPI: YYYY-MM-DDTHH:MM:SSZ
        start_str = start_t.strftime('%Y-%m-%dT%H:%M:%SZ')
        stop_str = stop_t.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Fetch Kp Index
    kp_df = None
    if start_str and stop_str:
        print("\n--- Fetching Kp Index ---")
        # Fetch Kp index
        res_t, res_index, res_s = getKpindex(start_str, stop_str, 'Kp')
        
        if res_t and res_index:
            kp_df = pd.DataFrame({'time_utc': res_t, 'Kp': res_index})
            if res_s:
                kp_df['status'] = res_s
            
            # Convert time to datetime and localize to UTC if not already
            # The API returns strings like '2024-12-01T00:00:00Z'
            kp_df['time_utc'] = pd.to_datetime(kp_df['time_utc'])
            if kp_df['time_utc'].dt.tz is None:
                kp_df['time_utc'] = kp_df['time_utc'].dt.tz_localize('UTC')
            else:
                kp_df['time_utc'] = kp_df['time_utc'].dt.tz_convert('UTC')
            
            # Save Kp data to CSV
            kp_csv_name = os.path.join(OUTPUT_DIR, f'Kp_index_{TIMESTAMP}.csv')
            kp_df.to_csv(kp_csv_name, index=False)
            print(f"Kp index data saved to {kp_csv_name}")
        else:
            print("No Kp index data found.")

    # 2. Create individual plots
    plot_individual(df1, FILE_1, kp_df=kp_df)
    plot_individual(df2, FILE_2, kp_df=kp_df)

    # Initialize datasets for correlation
    all_datasets = {}
    if df1 is not None: all_datasets[FILE_1] = df1
    if df2 is not None: all_datasets[FILE_2] = df2

    # Loop over IAGA codes
    for iaga_code in DEFAULT_IAGA_CODES:
        print(f"\n=== Processing IAGA Code: {iaga_code} ===")
        
        df_ref = None
        if start_str and stop_str:
            df_ref = fetch_iaga_data(iaga_code, start_str, stop_str)
            if df_ref is not None:
                all_datasets[f'{iaga_code} (Intermagnet)'] = df_ref

        # 4. Create comparison plot
        plot_comparison(df1, FILE_1, df2, FILE_2, df_ref, f'{iaga_code} (Intermagnet)', kp_df=kp_df)

        # 6. Calculate calibration (Fit vs Reference)
        if df_ref is not None:
            # Use different colors or filenames to distinguish? 
            # The optimize_and_plot_fit function uses iaga_code in filename, so it's safe.
            if df1 is not None:
                optimize_and_plot_fit(df1, "Station 1", df_ref, iaga_code, '#1f77b4', kp_df=kp_df)
            if df2 is not None:
                optimize_and_plot_fit(df2, "Station 2", df_ref, iaga_code, '#d62728', kp_df=kp_df)

    # 5. Calculate correlation for ALL datasets
    calculate_and_print_correlation(all_datasets)

    # 5b. New Step: Plot XYZ components for all datasets
    for name, df in all_datasets.items():
        plot_xyz_components(df, name)

    # --- NEW: Generate Master CSV with ALL data ---
    print("\n--- Generating Master CSV ---")
    master_df = pd.DataFrame()

    # 1. Merge all magnetic data (Magnitude)
    for name, df in all_datasets.items():
        if df is None or df.empty: continue
        
        # Prepare temp dataframe
        temp = df[['time_utc', 'magnitude']].copy()
        temp['time_utc'] = temp['time_utc'].dt.round('1min')
        temp = temp.groupby('time_utc').mean() # Handle duplicates
        temp = temp.rename(columns={'magnitude': name})
        
        if master_df.empty:
            master_df = temp
        else:
            master_df = master_df.join(temp, how='outer')

    # 2. Merge Kp Index
    if kp_df is not None and not master_df.empty:
        # Prepare Kp data
        kp_temp = kp_df[['time_utc', 'Kp']].copy()
        kp_temp = kp_temp.set_index('time_utc')
        
        # Sort both to be sure
        master_df = master_df.sort_index()
        kp_temp = kp_temp.sort_index()
        
        # Merge using asof (backward) or reindex with ffill
        # Since Kp is valid for the 3h interval starting at timestamp, we want to forward fill
        # But join/merge usually matches exact keys.
        # Strategy: Reindex Kp to master_df index using ffill
        
        # Combine indices to ensure we cover the range
        # Actually, we only care about the timestamps where we have magnetic data
        
        # We use merge_asof to find the latest Kp value for each magnetic timestamp
        # merge_asof requires sorted dataframes
        
        # Reset index to use merge_asof
        master_df = master_df.reset_index()
        kp_temp = kp_temp.reset_index()
        
        merged_master = pd.merge_asof(master_df, kp_temp, on='time_utc', direction='backward')
        
        # Set index back
        master_df = merged_master.set_index('time_utc')

    # Save Master CSV
    if not master_df.empty:
        master_csv_name = os.path.join(OUTPUT_DIR, f'master_data_{TIMESTAMP}.csv')
        master_df.to_csv(master_csv_name)
        print(f"Master CSV saved as: {master_csv_name}")
    else:
        print("Master DataFrame is empty. Nothing to save.")

    print("\nDone! All plots have been saved.")
    # plt.show() # Do not open window anymore
