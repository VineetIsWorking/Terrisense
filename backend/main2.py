from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from geopy.distance import geodesic
import io
import folium
import os
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. SMART READER ---
def read_file_smartly(content, filename):
    filename = filename.lower()
    if filename.endswith('.csv'):
        try:
            df_temp = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None, nrows=20)
            read_func = lambda skip: pd.read_csv(io.StringIO(content.decode('utf-8')), skiprows=skip)
        except:
            raise ValueError("Could not read CSV.")
    elif filename.endswith(('.xls', '.xlsx')):
        df_temp = pd.read_excel(io.BytesIO(content), header=None, nrows=20)
        read_func = lambda skip: pd.read_excel(io.BytesIO(content), skiprows=skip)
    else:
        raise ValueError("Invalid file format")

    header_idx = 0
    found = False
    # Detect Sales Data Headers
    for idx, row in df_temp.iterrows():
        row_str = row.astype(str).str.lower().values
        if any("row labels" in s for s in row_str):
            header_idx = idx
            found = True
            break
    
    # Detect Rep File Headers (Fallback)
    if not found:
        for idx, row in df_temp.iterrows():
            row_str = row.astype(str).str.lower().values
            if any("employee" in s for s in row_str): # Looking for employee_id
                header_idx = idx
                found = True
                break

    if found: df = read_func(header_idx)
    else:
        if filename.endswith('.csv'): df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        else: df = pd.read_excel(io.BytesIO(content))
    return df

# --- 2. LOGIC: REP-CENTERED CLUSTERING ---
def process_with_reps(df_sales, df_reps):
    # --- A. CLEAN SALES DATA ---
    df_sales.columns = df_sales.columns.str.strip()
    rename_sales = {
        'Row Labels': 'zip_code',
        'Sum of Final Weighatge': 'weight',
        'Count of NPI': 'npi_count',
        'Max of Latitude': 'latitude',
        'Max of Longitute': 'longitude'
    }
    for col in df_sales.columns:
        for key in rename_sales:
            if key.lower() == col.lower():
                df_sales.rename(columns={col: rename_sales[key]}, inplace=True)

    # --- B. CLEAN REP DATA ---
    df_reps.columns = df_reps.columns.str.strip()
    # Expecting: employee_id, sample_storage_city, sample_storage_state, sample_storage_zip
    rename_reps = {
        'sample_storage_zip': 'rep_zip',
        'employee_id': 'rep_id'
    }
    for col in df_reps.columns:
        for key in rename_reps:
            if key.lower() == col.lower():
                df_reps.rename(columns={col: rename_reps[key]}, inplace=True)

    # --- C. FIND REP COORDINATES ---
    # We need to find where the Reps live by looking up their Zip in the Sales Data
    # Create a reference map of Zip -> Lat/Lon from the main sales file
    zip_coords = df_sales.set_index('zip_code')[['latitude', 'longitude']].to_dict('index')
    
    rep_initial_centers = []
    valid_reps = []

    for _, rep in df_reps.iterrows():
        r_zip = rep['rep_zip']
        if r_zip in zip_coords:
            coords = zip_coords[r_zip]
            # Ensure coordinates are valid
            if not np.isnan(coords['latitude']) and not np.isnan(coords['longitude']):
                rep_initial_centers.append([coords['latitude'], coords['longitude']])
                valid_reps.append(rep['rep_id'])
    
    # Convert to numpy array for K-Means
    init_matrix = np.array(rep_initial_centers)
    num_clusters = len(init_matrix)
    
    if num_clusters == 0:
        raise ValueError("Could not match any Rep Zips to the Sales Data Zips. Check your Rep File.")

    # --- D. RUN SEED-BASED K-MEANS ---
    sales_coords = df_sales[['latitude', 'longitude']].fillna(0)
    sales_weights = df_sales['weight'].fillna(0)

    # CRITICAL: We pass 'init=init_matrix' to force clusters to start at Rep locations
    # n_init=1 ensures it doesn't try to move them randomly
    kmeans = KMeans(n_clusters=num_clusters, init=init_matrix, n_init=1, random_state=42)
    kmeans.fit(sales_coords, sample_weight=sales_weights)
    
    df_sales['Territory_ID'] = kmeans.labels_
    
    # --- E. ANALYZE (The 1000 Rule) ---
    territory_status = {}
    
    for t_id in range(num_clusters):
        cluster_data = df_sales[df_sales['Territory_ID'] == t_id]
        total_weight = cluster_data['weight'].sum()
        
        # Calculate Diameter
        if len(cluster_data) < 2: diameter = 0
        else:
            lat_min, lat_max = cluster_data['latitude'].min(), cluster_data['latitude'].max()
            lon_min, lon_max = cluster_data['longitude'].min(), cluster_data['longitude'].max()
            diameter = geodesic((lat_min, lon_min), (lat_max, lon_max)).miles

        # Rules (Target 1000 +/- 25% => 750 to 1250)
        status, message = "Green", "Optimal"
        
        if total_weight > 1250:
             status, message = "Red", f"Overloaded ({int(total_weight)})"
        elif total_weight < 750:
             status, message = "Yellow", f"Underutilized ({int(total_weight)})"
        
        # Map the Territory ID back to the Employee ID for display
        rep_name = str(valid_reps[t_id])

        territory_status[t_id] = {
            "Rep_ID": rep_name,
            "Status": status,
            "Message": message,
            "Weight": int(total_weight),
            "Diameter": int(diameter),
            "Center_Lat": init_matrix[t_id][0], # Rep's Home Lat
            "Center_Lon": init_matrix[t_id][1]  # Rep's Home Lon
        }

    return df_sales, territory_status, num_clusters
# --- 2. LOGIC: REP-CENTERED CLUSTERING (FIXED) ---
# def process_with_reps(df_sales, df_reps):
#     # --- A. CLEAN SALES DATA ---
#     df_sales.columns = df_sales.columns.str.strip()
#     rename_sales = {
#         'Row Labels': 'zip_code',
#         'Sum of Final Weighatge': 'weight',
#         'Count of NPI': 'npi_count',
#         'Max of Latitude': 'latitude',
#         'Max of Longitute': 'longitude'
#     }
#     for col in df_sales.columns:
#         for key in rename_sales:
#             if key.lower() == col.lower():
#                 df_sales.rename(columns={col: rename_sales[key]}, inplace=True)

#     # --- B. CLEAN REP DATA ---
#     df_reps.columns = df_reps.columns.str.strip()
#     rename_reps = {
#         'sample_storage_zip': 'rep_zip',
#         'employee_id': 'rep_id'
#     }
#     for col in df_reps.columns:
#         for key in rename_reps:
#             if key.lower() == col.lower():
#                 df_reps.rename(columns={col: rename_reps[key]}, inplace=True)

#     # --- C. ZIP NORMALIZER (THE FIX) ---
#     # Convert everything to String, remove decimals (.0), and add leading zeros
#     df_sales['zip_code'] = df_sales['zip_code'].astype(str).str.split('.').str[0].str.zfill(5)
#     df_reps['rep_zip'] = df_reps['rep_zip'].astype(str).str.split('.').str[0].str.zfill(5)

#     # --- D. FIND REP COORDINATES ---
#     # Create reference map: Zip -> Lat/Lon
#     # We drop duplicates just in case multiple rows have same zip
#     zip_coords = df_sales.drop_duplicates('zip_code').set_index('zip_code')[['latitude', 'longitude']].to_dict('index')
    
#     rep_initial_centers = []
#     valid_reps = []
#     missing_reps = []

#     for _, rep in df_reps.iterrows():
#         r_zip = rep['rep_zip']
#         if r_zip in zip_coords:
#             coords = zip_coords[r_zip]
#             if not np.isnan(coords['latitude']) and not np.isnan(coords['longitude']):
#                 rep_initial_centers.append([coords['latitude'], coords['longitude']])
#                 valid_reps.append(rep['rep_id'])
#         else:
#             missing_reps.append(r_zip)
    
#     # Debugging: Print how many matched
#     print(f"Matched {len(valid_reps)} Reps. Missing: {len(missing_reps)}")
    
#     if len(rep_initial_centers) == 0:
#         raise ValueError(f"CRITICAL ERROR: No Rep Zips matched. Example Sales Zip: {df_sales['zip_code'].iloc[0]}, Example Rep Zip: {df_reps['rep_zip'].iloc[0]}")

#     # Convert to numpy array for K-Means
#     init_matrix = np.array(rep_initial_centers)
#     num_clusters = len(init_matrix)

#     # --- E. RUN SEED-BASED K-MEANS ---
#     sales_coords = df_sales[['latitude', 'longitude']].fillna(0)
#     sales_weights = df_sales['weight'].fillna(0)

#     # Force start at Rep locations
#     kmeans = KMeans(n_clusters=num_clusters, init=init_matrix, n_init=1, random_state=42)
#     kmeans.fit(sales_coords, sample_weight=sales_weights)
    
#     df_sales['Territory_ID'] = kmeans.labels_
    
#     # --- F. ANALYZE ---
#     territory_status = {}
    
#     for t_id in range(num_clusters):
#         cluster_data = df_sales[df_sales['Territory_ID'] == t_id]
#         total_weight = cluster_data['weight'].sum()
        
#         if len(cluster_data) < 2: diameter = 0
#         else:
#             lat_min, lat_max = cluster_data['latitude'].min(), cluster_data['latitude'].max()
#             lon_min, lon_max = cluster_data['longitude'].min(), cluster_data['longitude'].max()
#             diameter = geodesic((lat_min, lon_min), (lat_max, lon_max)).miles

#         status, message = "Green", "Optimal"
#         if total_weight > 1250:
#              status, message = "Red", f"Overloaded ({int(total_weight)})"
#         elif total_weight < 750:
#              status, message = "Yellow", f"Underutilized ({int(total_weight)})"
        
#         rep_name = str(valid_reps[t_id])

#         territory_status[t_id] = {
#             "Rep_ID": rep_name,
#             "Status": status,
#             "Message": message,
#             "Weight": int(total_weight),
#             "Diameter": int(diameter),
#             "Center_Lat": init_matrix[t_id][0], 
#             "Center_Lon": init_matrix[t_id][1]
#         }

#     return df_sales, territory_status, num_clusters
# --- 3. MAP SAVER ---
def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def save_map_to_file(df, territory_status, num_clusters, filename="territory_map.html"):
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, prefer_canvas=True)
    
    cluster_colors = [get_random_color() for _ in range(num_clusters)]

    # 1. Draw Polygons (Territories)
    for t_id in range(num_clusters):
        cluster_data = df[df['Territory_ID'] == t_id]
        if len(cluster_data) < 3: continue 
        
        points = cluster_data[['latitude', 'longitude']].values
        stats = territory_status[t_id]
        fill_col = cluster_colors[t_id]
        border_col = "red" if stats['Status'] == 'Red' else "black"
        
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            popup_text = f"<b>Rep: {stats['Rep_ID']}</b><br>Weight: {stats['Weight']}<br>Status: {stats['Message']}"

            folium.Polygon(
                locations=hull_points, color=border_col, weight=2,
                fill=True, fill_color=fill_col, fill_opacity=0.4,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        except: pass

    # 2. Draw Rep Home Base (Star Marker)
    for t_id in range(num_clusters):
        stats = territory_status[t_id]
        folium.Marker(
            [stats['Center_Lat'], stats['Center_Lon']],
            popup=f"<b>Home Base: {stats['Rep_ID']}</b>",
            icon=folium.Icon(color='black', icon='home', prefix='fa')
        ).add_to(m)

    # 3. Draw Zips (Dots)
    for _, row in df.iterrows():
        t_id = int(row['Territory_ID'])
        fill_col = cluster_colors[t_id]
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2, color=fill_col, fill=True, fill_opacity=1.0,
            popup=f"Zip: {row['zip_code']}"
        ).add_to(m)

    m.save(filename)
    return filename

# --- 4. NEW ENDPOINT (TWO FILES) ---
@app.post("/optimize_with_reps")
async def optimize_with_reps(
    sales_file: UploadFile = File(...),
    rep_file: UploadFile = File(...)
):
    try:
        # Read Sales File
        content_sales = await sales_file.read()
        df_sales = read_file_smartly(content_sales, sales_file.filename)
        
        # Read Rep File
        content_reps = await rep_file.read()
        df_reps = read_file_smartly(content_reps, rep_file.filename)
        
        # Process
        df_processed, territory_status, k = process_with_reps(df_sales, df_reps)
        
        # Generate Map
        output_file = "rep_alignment_map.html"
        save_map_to_file(df_processed, territory_status, k, output_file)
        
        return FileResponse(output_file, media_type='text/html', filename="rep_alignment_map.html")
        
    except Exception as e:
        return {"error": str(e)}