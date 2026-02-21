import mysql.connector
import os
import time
import pandas as pd
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
import logging
import warnings

# Unterdrücke UserWarnung von pandas bzgl. DB-Connection
warnings.filterwarnings('ignore', category=UserWarning)

# Logging config
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("sshtunnel").setLevel(logging.WARNING)

load_dotenv()

def get_raw_measurements(start_time, end_time):
    # Diese Funktion wird erweitert, um beide Sensoren abzufragen
    return get_all_sensor_data(start_time, end_time)

def get_all_sensor_data(start_time, end_time):
    SSH_HOST = os.getenv("SSH_HOST")         
    SSH_USER = os.getenv("SSH_USER")
    SSH_PASS = os.getenv("SSH_PASSWORD")
    
    # DB Konstanten
    TARGET_DB_IP = '10.99.0.101'             
    TARGET_DB_PORT = 3306
    
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_DATABASE")

    print(f"⏱️  [{time.strftime('%X')}] Starte Datenabfrage für beide Sensoren... ({start_time} - {end_time})")
    
    tunnel = None
    connection = None
    results = {}
    
    try:
        # 1. Tunnel aufbauen
        print(f"🔨 [{time.strftime('%X')}] Baue SSH-Tunnel auf...")
        
        tunnel = SSHTunnelForwarder(
            (SSH_HOST, 22),
            ssh_username=SSH_USER,
            ssh_password=SSH_PASS,
            remote_bind_address=(TARGET_DB_IP, TARGET_DB_PORT),
            ssh_pkey=None,
            allow_agent=False
        )
        
        tunnel.start()
        print(f"✅ [{time.strftime('%X')}] SSH-Tunnel aktiv (Lokal: {tunnel.local_bind_port})")

        # 2. Datenbank Abfragen
        print(f"🔌 [{time.strftime('%X')}] Starte Abfragen in Datenbanken...")

        # Definition der Datenquellen: (Datenbankname, Tabellenname, Label)
        # Datenbank 2 hat '2' am Ende, Tabelle heißt in beiden gleich 'rawmeasurements'
        sources = [
            (DB_NAME, 'rawmeasurements', 'Station 1'), 
            (f"{DB_NAME}2", 'rawmeasurements', 'Station 2')
        ]
        
        base_config = {
            'user': DB_USER,
            'password': DB_PASS,
            'host': '127.0.0.1',
            'port': tunnel.local_bind_port,
            'connection_timeout': 20,
            'use_pure': True,
            'ssl_disabled': True
        }

        for db_name, table_name, label in sources:
            print(f"🔄 [{time.strftime('%X')}] Verbinde mit DB '{db_name}' für {label}...")
            
            current_config = base_config.copy()
            current_config['database'] = db_name
            
            conn = None
            try:
                conn = mysql.connector.connect(**current_config)
                if conn.is_connected():
                    print(f"   🔍 [{time.strftime('%X')}] Lade Daten aus {table_name}...")
                    query = f"SELECT * FROM {table_name} WHERE time_utc BETWEEN %s AND %s"
                    
                    df = pd.read_sql(query, conn, params=(start_time, end_time))
                    if not df.empty:
                        results[label] = df
                        print(f"   ✅ {len(df)} Zeilen geladen.")
                    else:
                        print(f"   ⚠️  Keine Daten gefunden.")
                        results[label] = None
            
            except mysql.connector.Error as err:
                print(f"   ❌ Datenbank-Fehler bei {label}: {err}")
                results[label] = None
            except Exception as e:
                print(f"   ❌ Fehler bei {label}: {e}")
                results[label] = None
            finally:
                if conn and conn.is_connected():
                    conn.close()

        print(f"🎉 [{time.strftime('%X')}] Alle Abfragen abgeschlossen.")
        return results

    except Exception as e:
        print(f"❌ [{time.strftime('%X')}] Kritischer Fehler (Tunnel/Setup): {e}")
    finally:
        if tunnel:
            tunnel.stop()
            print(f"🔒 [{time.strftime('%X')}] SSH-Tunnel geschlossen.")

    return None

if __name__ == "__main__":
    # Konfigurierbare Zeiträume
    start_ts = "2024-07-13 12:34:35.160"
    end_ts   = "2024-07-13 14:34:35.160"
    
    print(f"--- Start Abfrage ---")
    df = get_raw_measurements(start_ts, end_ts)
    
    if df is not None:
        print("\nErgebnis Vorschau:")
        print(df.head())
        print(f"\nSpalten: {list(df.columns)}")
        # df.to_csv('output.csv', index=False)