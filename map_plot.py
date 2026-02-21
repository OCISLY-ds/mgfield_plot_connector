import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hapiclient import hapi
import os
import json
import urllib.request
import math
import io
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

OUTPUT_DIR = 'output'
STATION_1_2_COORDS = (53.65084605271855, 9.424179775258073) # Lat, Lon

# --- OpenStreetMap Tile Functions ---
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_tile_image(xtile, ytile, zoom):
    url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    # User-Agent is required by OSM tile usage policy
    req = urllib.request.Request(url, headers={'User-Agent': 'MGFieldConnector/1.0 (Education Project)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
            return plt.imread(io.BytesIO(data), format='png')
    except Exception as e:
        print(f"Failed to fetch tile {url}: {e}")
        return None

def get_station_coords(station_code):
    """Fetches metadata for a station to get coordinates."""
    server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
    dataset = f'{station_code.lower()}/best-avail/PT1M/xyzf'
    parameters = 'Field_Vector' # We just need metadata
    
    try:
        # We assume metadata is returned in the 'meta' object from hapi call
        # But hapi() function returns (data, meta). 
        # If we request a tiny time range, we get metadata without downloading tons of data.
        # OR we can use hapi() with specific params.
        # Actually in my check script, meta was just the dictionary.
        # Using a dummy time range to get metadata.
        opts = {'logging': False}
        data, meta = hapi(server, dataset, parameters, '2024-01-01T00:00:00Z', '2024-01-01T00:01:00Z', **opts)
        
        # Check for x_latitude / x_longitude keys
        lat = meta.get('x_latitude')
        lon = meta.get('x_longitude')
        
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        else:
            print(f"Warning: No coordinates found for {station_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching metadata for {station_code}: {e}")
        return None

def fetch_geojson_background():
    """
    Fetches a simple GeoJSON for context. 
    Using a public raw URL for Germany Bundeslaender (States).
    """
    # Try different URLs if one fails
    urls = [
        "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/3_mittel.geojson",
        "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geojson",
        "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json" # Fallback to world
    ]
    
    for url in urls:
        try:
            print(f"Fetching background map data from {url}...")
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
            return data
        except Exception:
            continue
            
    print("Could not fetch any background map.")
    return None

def plot_geojson(ax, geojson_data, linewidth=0.5):
    """Plots GeoJSON polygons on the axes."""
    if not geojson_data: return

    patches = []
    
    for feature in geojson_data.get('features', []):
        geometry = feature.get('geometry', {})
        t = geometry.get('type')
        coords = geometry.get('coordinates')
        
        if t == 'Polygon':
            for poly_coords in coords:
                # poly_coords is list of [lon, lat]
                # matplotlib expects (N, 2) array
                poly = Polygon(poly_coords, closed=True)
                patches.append(poly)
        elif t == 'MultiPolygon':
            for multi in coords:
                for poly_coords in multi:
                    poly = Polygon(poly_coords, closed=True)
                    patches.append(poly)
    
    # Increased alpha from 0.2 to 0.4 for better visibility
    p = PatchCollection(patches, alpha=0.4, facecolor='lightgray', edgecolor='gray', linewidth=linewidth, zorder=0)
    ax.add_collection(p)


def create_map(reference_codes):
    print("\n=== Generating Map Plot ===")
    
    # 1. Collect Coordinates
    stations = []
    
    # Add Fixed Station 1/2
    stations.append({
        'name': 'My Stations',
        'lat': STATION_1_2_COORDS[0],
        'lon': STATION_1_2_COORDS[1],
        'color': 'blue',
        'marker': 'o'
    })
    
    # Add Reference Stations
    for code in reference_codes:
        coords = get_station_coords(code)
        if coords:
            stations.append({
                'name': f'{code} (Intermagnet)',
                'lat': coords[0],
                'lon': coords[1],
                'color': 'black',
                'marker': '^'
            })
    
    if not stations:
        print("No stations to plot.")
        return

    # 3. Calculate Map Extents and Fetch Tiles
    lats = [s['lat'] for s in stations]
    lons = [s['lon'] for s in stations]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Add Margin (approx 100% to zoom out further)
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Avoid zero range
    if lat_range == 0: lat_range = 0.1
    if lon_range == 0: lon_range = 0.1
    
    # Zoom: Increased margin_factor from 1.0 to 3.0 (Zoom out on main map)
    margin_factor = 3.0  
    lat_min_view = min_lat - lat_range * margin_factor
    lat_max_view = max_lat + lat_range * margin_factor
    lon_min_view = min_lon - lon_range * margin_factor
    lon_max_view = max_lon + lon_range * margin_factor
    
    # 2. Setup Plot with Dynamic Figure Size to remove whitespace
    # Calculate aspect ratio of the map view
    # Aspect Ratio = Height / Width = (dLat * deg_len) / (dLon * deg_len * cos(lat))
    view_center_lat = (lat_min_view + lat_max_view) / 2
    aspect_ratio = (lat_max_view - lat_min_view) / ((lon_max_view - lon_min_view) * np.cos(np.radians(view_center_lat)))
    
    # Set width to 12 inches, calculate height based on aspect ratio
    fig_width = 12
    # Add extra vertical height (1.5 inches) for titles and labels so it's not cramped
    fig_height = (fig_width * aspect_ratio) + 1.5
    
    # Cap height if needed
    if fig_height > 15: 
        fig_height = 15
        # Recalculate width to fit
        available_data_height = fig_height - 1.5
        fig_width = available_data_height / aspect_ratio
        
    print(f"Creating figure with size: {fig_width:.1f}x{fig_height:.1f} inches (Aspect: {aspect_ratio:.2f})")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Adjust layout to give more breathing room top/bottom
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
    
    # Determine Zoom Level
    # Rough estimation: 360 degrees / lon_range * map_width_pixels / 256
    # Assuming standard map width ~800px
    if lon_range > 0:
        zoom = int(np.log2(360.0 / (lon_max_view - lon_min_view) * 3))
    else:
        zoom = 10
    
    # Limit Zoom to reasonable range for OSM (0-19)
    zoom = max(0, min(14, zoom)) # Don't zoom in too crazy
    print(f"Fetching OSM tiles at zoom level {zoom}...")
    
    # Get Tile Ranges
    xtile_min, ytile_min = deg2num(lat_max_view, lon_min_view, zoom) # Top Left
    xtile_max, ytile_max = deg2num(lat_min_view, lon_max_view, zoom) # Bottom Right
    
    # Fetch and Plot Tiles
    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            tile_img = get_tile_image(x, y, zoom)
            if tile_img is not None:
                # Calculate extent of this tile in Lat/Lon
                lat_n, lon_w = num2deg(x, y, zoom)
                lat_s, lon_e = num2deg(x + 1, y + 1, zoom)
                
                # Plot Tile
                # Extent: [left, right, bottom, top]
                ax.imshow(tile_img, extent=[lon_w, lon_e, lat_s, lat_n], zorder=0)

    # 4. Plot Stations
    for s in stations:
        ax.scatter(s['lon'], s['lat'], color=s['color'], marker=s['marker'], s=100, label=s['name'], zorder=10)
        
        # Add label with white background for readability
        # Using annotate with 'offset points' moves the label to the right reliably
        ax.annotate(s['name'], 
                    xy=(s['lon'], s['lat']), 
                    xytext=(10, 0),  # 10 points to the right
                    textcoords='offset points',
                    verticalalignment='center', fontsize=9, fontweight='bold', zorder=11,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1))

    # 5. Framing
    ax.set_ylim(lat_min_view, lat_max_view)
    ax.set_xlim(lon_min_view, lon_max_view)
    
    # Fix Aspect Ratio (Mercator-ish approximation)
    mean_lat = np.mean(lats)
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Station Locations')
    
    # Add attribution
    ax.text(1, 0, '© OpenStreetMap contributors', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray', zorder=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # --- Inset Map (Overview) ---
    # Create inset axes in Top Right corner
    # Position: [x, y, width, height]
    # Adjusted to [0.64, 0.64, 0.36, 0.36] to be fully in top-right corner
    ax_inset = ax.inset_axes([0.72, 0.60, 0.36, 0.36])
    
    # Fetch and plot GeoJSON (Germany/World) for context
    geojson_data = fetch_geojson_background()
    if geojson_data:
        # Pass linewidth=2.0 for bolder borders
        plot_geojson(ax_inset, geojson_data, linewidth=2.0)
        
        # Plot stations as simple dots on inset instead of rectangle
        for s in stations:
            ax_inset.scatter(s['lon'], s['lat'], color=s['color'],  marker='o', s=30, zorder=30)
        
        # Set inset limits larger than main view to show context
        # Center around main view center
        center_lon = (lon_min_view + lon_max_view) / 2
        center_lat = (lat_min_view + lat_max_view) / 2
        
        # Inset span: Zoom out further (increased min span to 8/12)
        inset_span_lat = max(8, (lat_max_view - lat_min_view) * 3)
        inset_span_lon = max(12, (lon_max_view - lon_min_view) * 3)
        
        ax_inset.set_xlim(center_lon - inset_span_lon/2, center_lon + inset_span_lon/2)
        ax_inset.set_ylim(center_lat - inset_span_lat/2, center_lat + inset_span_lat/2)
        
        # Fix Aspect Ratio for Inset (prevents "tilted" look)
        ax_inset.set_aspect(1.0 / np.cos(np.radians(center_lat)))
        
        # Add "Germany" label
        ax_inset.text(0.05, 0.95, 'Germany', transform=ax_inset.transAxes, ha='left', va='top', fontsize=10, fontweight='bold', zorder=40)

        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.grid(False)
        ax_inset.set_facecolor('white') # White background
        
        # Add border to inset
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_path = os.path.join(OUTPUT_DIR, 'station_map.png')
    plt.savefig(out_path, dpi=150)
    print(f"Map saved to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    # Test run
    create_map(['WNG'])
