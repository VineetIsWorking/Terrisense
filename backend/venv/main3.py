
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import io
import folium
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
TARGET_CAPACITY = 1250 # 1000 + 25% Variance. We stop filling after this.
MAX_RADIUS_MILES = 150 # Absolute outlier limit.

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
    keywords = ['row labels', 'zip_code', 'employee_id']
    for idx, row in df_temp.iterrows():
        row_str = row.astype(str).str.lower().values
        if any(k in s for s in row_str for k in keywords):
            header_idx = idx
            found = True
            break
            
    if found: df = read_func(header_idx)
    else:
        if filename.endswith('.csv'): df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        else: df = pd.read_excel(io.BytesIO(content))
    return df

# --- 2. THE GREEDY FILLER ALGORITHM ---
def run_greedy_capacity_assignment(df_sales, df_reps):
    # --- A. PREP DATA ---
    df_sales.columns = df_sales.columns.str.strip()
    rename_sales = {
        'Row Labels': 'zip_code', 'Sum of Final Weighatge': 'weight',
        'Count of NPI': 'npi_count', 'Max of Latitude': 'latitude', 'Max of Longitute': 'longitude'
    }
    for col in df_sales.columns:
        for key in rename_sales:
            if key.lower() == col.lower(): df_sales.rename(columns={col: rename_sales[key]}, inplace=True)

    df_reps.columns = df_reps.columns.str.strip()
    rename_reps = { 'sample_storage_zip': 'rep_zip', 'employee_id': 'rep_id' }
    for col in df_reps.columns:
        for key in rename_reps:
            if key.lower() == col.lower(): df_reps.rename(columns={col: rename_reps[key]}, inplace=True)

    # Normalize Zips
    df_sales['zip_code'] = df_sales['zip_code'].astype(str).str.split('.').str[0].str.zfill(5)
    df_reps['rep_zip'] = df_reps['rep_zip'].astype(str).str.split('.').str[0].str.zfill(5)

    # --- B. SETUP REPS ---
    zip_coords = df_sales.drop_duplicates('zip_code').set_index('zip_code')[['latitude', 'longitude']].to_dict('index')
    
    reps = []
    for _, r in df_reps.iterrows():
        if r['rep_zip'] in zip_coords:
            c = zip_coords[r['rep_zip']]
            if not np.isnan(c['latitude']):
                reps.append({
                    'id': str(r['rep_id']),
                    'lat': c['latitude'],
                    'lon': c['longitude'],
                    'current_weight': 0,
                    'count': 0
                })
    
    if not reps: raise ValueError("No Reps found.")
    
    # --- C. CALCULATE ALL DISTANCES (THE MATRIX) ---
    # We need distance from EVERY Zip to EVERY Rep.
    # For 40k zips and 64 reps, that is 2.5 million pairs. Python handles this easily.
    
    sales_data = df_sales[['zip_code', 'latitude', 'longitude', 'weight']].to_dict('records')
    
    potential_assignments = []
    
    for zip_idx, z in enumerate(sales_data):
        if z['weight'] <= 0: continue # Skip zero value zips
        if np.isnan(z['latitude']): continue

        # Calculate dist to all 64 reps
        for rep_idx, r in enumerate(reps):
            # Euclidian approx for speed (deg * 69)
            dist_miles = np.sqrt((z['latitude'] - r['lat'])**2 + (z['longitude'] - r['lon'])**2) * 69
            
            if dist_miles <= MAX_RADIUS_MILES:
                potential_assignments.append({
                    'zip_idx': zip_idx,
                    'rep_idx': rep_idx,
                    'dist': dist_miles
                })
    
    # --- D. SORT BY DISTANCE (Closest First) ---
    # This is the "Priority" part. We process the shortest lines first.
    potential_assignments.sort(key=lambda x: x['dist'])
    
    # --- E. GREEDY ASSIGNMENT LOOP ---
    assigned_zips = set()
    final_assignments = {} # zip_idx -> rep_idx
    
    for p in potential_assignments:
        z_idx = p['zip_idx']
        r_idx = p['rep_idx']
        
        # 1. Is Zip already taken?
        if z_idx in assigned_zips:
            continue
            
        # 2. Is Rep full?
        rep_current = reps[r_idx]['current_weight']
        zip_weight = sales_data[z_idx]['weight']
        
        if rep_current + zip_weight <= TARGET_CAPACITY:
            # ASSIGN IT!
            final_assignments[z_idx] = r_idx
            reps[r_idx]['current_weight'] += zip_weight
            reps[r_idx]['count'] += 1
            assigned_zips.add(z_idx)
            
    # --- F. BUILD RESULT DATAFRAME ---
    # Only keep zips that got assigned. The rest are outliers.
    
    results = []
    for z_idx, r_idx in final_assignments.items():
        row = sales_data[z_idx].copy()
        row['Territory_ID'] = r_idx
        row['Rep_ID'] = reps[r_idx]['id']
        results.append(row)
        
    df_result = pd.DataFrame(results)
    
    # Stats for Map
    territory_status = {}
    for i, r in enumerate(reps):
        status = "Green" if r['current_weight'] > 750 else "Yellow"
        territory_status[i] = {
            "Rep_ID": r['id'],
            "Weight": int(r['current_weight']),
            "Status": status,
            "Home_Lat": r['lat'],
            "Home_Lon": r['lon']
        }
        
    system_msg = f"Greedy Fill Complete. Assigned {len(df_result)} Zips. Ignored outliers/overflow."
    
    return df_result, territory_status, len(reps), system_msg

# --- 3. MAP GENERATOR (CLEAN DOTS) ---
def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def save_map_to_file(df, territory_status, num_clusters, sys_msg, filename="territory_map.html"):
    if df.empty: return "error.html"
    
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, prefer_canvas=True)
    
    folium.map.Marker(
        [center_lat + 5, center_lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 16px; background: white; padding: 5px; border: 2px solid black;"><b>{sys_msg}</b></div>')
    ).add_to(m)

    cluster_colors = [get_random_color() for _ in range(num_clusters)]

    # Draw Zips (Dots)
    for _, row in df.iterrows():
        t_id = int(row['Territory_ID'])
        fill_col = cluster_colors[t_id]
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3, 
            color=fill_col,
            fill=True,
            fill_opacity=0.8,
            weight=0,
            popup=f"Zip: {row['zip_code']}<br>Owner: {row['Rep_ID']}"
        ).add_to(m)

    # Draw Reps (Anchors)
    for t_id in range(num_clusters):
        stats = territory_status[t_id]
        if stats['Weight'] == 0: continue # Skip empty reps

        folium.Marker(
            [stats['Home_Lat'], stats['Home_Lon']],
            popup=f"<b>{stats['Rep_ID']}</b><br>Weight: {stats['Weight']}",
            icon=folium.Icon(color='black', icon='user', prefix='fa')
        ).add_to(m)

    m.save(filename)
    return filename

# --- 4. ENDPOINT ---
@app.post("/optimize_professional")
async def optimize_professional(sales_file: UploadFile = File(...), rep_file: UploadFile = File(...)):
    try:
        content_s = await sales_file.read()
        df_s = read_file_smartly(content_s, sales_file.filename)
        content_r = await rep_file.read()
        df_r = read_file_smartly(content_r, rep_file.filename)
        
        df_processed, report, k, sys_msg = run_greedy_capacity_assignment(df_s, df_r)
        
        output = "greedy_map.html"
        save_map_to_file(df_processed, report, k, sys_msg, output)
        return FileResponse(output, media_type='text/html', filename="greedy_map.html")
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/optimize_professional_excel")
async def optimize_professional_excel(sales_file: UploadFile = File(...), rep_file: UploadFile = File(...)):
    try:
        # 1. Read Files
        content_s = await sales_file.read()
        df_s = read_file_smartly(content_s, sales_file.filename)
        
        content_r = await rep_file.read()
        df_r = read_file_smartly(content_r, rep_file.filename)
        
        # 2. Run the Greedy Algorithm
        df_processed, _, _, _ = run_greedy_capacity_assignment(df_s, df_r)

        # 3. Rename and Reorder for Excel
        # Rename 'Territory_ID' to 'Unique_Cluster_ID' as requested
        df_processed.rename(columns={'Territory_ID': 'Unique_Cluster_ID'}, inplace=True)

        # Reorder columns: Put Cluster ID and Rep ID at the front
        priority_cols = ['Unique_Cluster_ID', 'Rep_ID', 'zip_code', 'weight']
        other_cols = [c for c in df_processed.columns if c not in priority_cols]
        df_processed = df_processed[priority_cols + other_cols]

        # 4. Sort by Cluster ID so groups are together
        df_processed = df_processed.sort_values(by=['Unique_Cluster_ID'])

        # 5. Create Excel in Memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_processed.to_excel(writer, index=False, sheet_name='Assigned_Territories')
        
        output.seek(0)

        # 6. Return File
        headers = {
            'Content-Disposition': 'attachment; filename="professional_assignments.xlsx"'
        }
        return Response(
            content=output.getvalue(), 
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
            headers=headers
        )

    except Exception as e:
        return {"error": str(e)}