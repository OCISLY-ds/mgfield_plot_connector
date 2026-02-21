import db_connect
import plot
import time
from datetime import datetime
import pandas as pd

def parse_german_date(date_str):
    """
    Parses DD.MM.YYYY or DD.MM.YYYY HH:MM:SS into YYYY-MM-DD HH:MM:SS format.
    Returns None if parsing fails.
    """
    if not date_str: return None
    
    # Try different formats
    formats = [
        "%d.%m.%Y %H:%M:%S", 
        "%d.%m.%Y %H:%M", 
        "%d.%m.%Y"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Force seconds to 00
            dt = dt.replace(second=0, microsecond=0)
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            continue
            
    # Fallback: Try pandas parsing which is very robust
    try:
        dt = pd.to_datetime(date_str, dayfirst=True)
        # Force seconds to 00
        dt = dt.replace(second=0, microsecond=0)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    except:
        return None

def main():
    print(f"=== M G F I E L D   C O N N E C T O R ===")
    
    # 1. User Input for Time Range
    print("Bitte Zeitraum eingeben (Gewünschtes Format: DD.MM.YYYY HH:MM)")
    print("Beispiel: 13.07.2024 12:30")
    
    default_start_show = "13.07.2024 12:34"
    default_end_show   = "13.07.2024 14:34"
    
    # Internal defaults (SQL format)
    default_start_sql = "2024-07-13 12:34:00.000"
    default_end_sql   = "2024-07-13 14:34:00.000"
    
    start_in = input(f"Startzeit (Enter für '{default_start_show}'): ").strip()
    end_in   = input(f"Endzeit   (Enter für '{default_end_show}'): ").strip()
    
    # Parse inputs
    start_time_str = parse_german_date(start_in) if start_in else default_start_sql
    end_time_str   = parse_german_date(end_in)   if end_in   else default_end_sql
    
    if not start_time_str or not end_time_str:
        print("❌ Ungültiges Datumsformat. Bitte DD.MM.YYYY verwenden.")
        return
    
    print("-" * 40)
    print(f"Gewählter Zeitraum (SQL): {start_time_str} bis {end_time_str}")
    print("-" * 40)
    
    # 1.5 Referenzstationen
    iaga_in = input(f"Intermagnet Station Code (Default 'WNG'): ").strip()
    iaga_codes = [code.strip().upper() for code in iaga_in.split(',')] if iaga_in else ['WNG']
    print(f"Gewählte Referenzstationen: {iaga_codes}")

    # 2. Daten Abrufen (Beide Sensoren)
    # db_connect.get_all_sensor_data kümmert sich um den SSH Tunnel und beide Queries
    sensor_data = db_connect.get_all_sensor_data(start_time_str, end_time_str)
    
    if not sensor_data:
        print("❌ Keine Daten erhalten. Programm wird beendet.")
        return

    df1 = sensor_data.get('Station 1')
    df2 = sensor_data.get('Station 2')

    if df1 is None and df2 is None:
        print("❌ Beide Datensätze leer.")
        return

    # 3. Daten an Plotter übergeben
    # plot.run_plotting_pipeline übernimmt die Verarbeitung (Magnitude berechnen, etc.) und das Plotten
    print("\n>>> Starte Visualisierung...")
    plot.run_plotting_pipeline(
        df1, 
        df2, 
        start_time_arg=start_time_str, 
        end_time_arg=end_time_str,
        iaga_codes=iaga_codes
    )
    
    print("\n✅ Ausführung abgeschlossen.")

if __name__ == "__main__":
    main()