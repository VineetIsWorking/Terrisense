from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from geopy.distance import geodesic
import io
import folium
import os
import random
import json
import zipfile
from typing import Dict, Tuple, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL: MASTER GEO DATA ---
MASTER_ZIP_DF: pd.DataFrame = pd.DataFrame()

def load_master_geodata(filepath: str = "us_zips.csv"):
    global MASTER_ZIP_DF
    try:
        if os.path.exists(filepath):
            print("Loading Master Zip Data...")
            MASTER_ZIP_DF = pd.read_csv(filepath, dtype=str)
            MASTER_ZIP_DF.columns = MASTER_ZIP_DF.columns.str.strip().str.lower()
            MASTER_ZIP_DF.rename(columns={
                'zip': 'zip_code', 'zipcode': 'zip_code', 'zip_code': 'zip_code', 'zip code': 'zip_code',
                'postal': 'zip_code', 'postalcode': 'zip_code',
                'lat': 'latitude', 'latitude': 'latitude',
                'lng': 'longitude', 'long': 'longitude', 'longitude': 'longitude'
            }, inplace=True)

            required = {'zip_code', 'latitude', 'longitude'}
            if required.issubset(MASTER_ZIP_DF.columns):
                MASTER_ZIP_DF['zip_code'] = MASTER_ZIP_DF['zip_code'].astype(str).str.strip().str.zfill(5)
                MASTER_ZIP_DF['latitude'] = pd.to_numeric(MASTER_ZIP_DF['latitude'], errors='coerce')
                MASTER_ZIP_DF['longitude'] = pd.to_numeric(MASTER_ZIP_DF['longitude'], errors='coerce')
                MASTER_ZIP_DF.dropna(subset=['latitude', 'longitude', 'zip_code'], inplace=True)
                print("Master Data Loaded.")
            else:
                print(f"Missing columns in {filepath}. Found: {MASTER_ZIP_DF.columns}")
        else:
            print(f"WARNING: '{filepath}' not found.")
    except Exception as e:
        print(f"Error loading master geo data: {e}")

load_master_geodata()

# --- HELPER: FILE READING ---
def read_file_smartly(content: bytes, filename: str) -> pd.DataFrame:
    filename = filename.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')), dtype=str)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content), dtype=str)
        else:
            raise ValueError("Invalid file. Please upload CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")
    df.columns = df.columns.str.strip()
    return df

# --- CORE LOGIC SPLIT INTO 2 PARTS ---

# PART 1: PRE-PROCESSING (Run Once)
# Merges Geo Data and calculates the 'Base Weight Ratio' (Value / Total * %)
def preprocess_data(df: pd.DataFrame, column_config: Dict[str, float]) -> pd.DataFrame:
    global MASTER_ZIP_DF
    
    # 1. Find Zip Column
    zip_col = next((c for c in df.columns if c.lower().replace('_', '').replace(' ', '') in ['zip', 'zipcode', 'postalcode', 'rowlabels']), None)
    if not zip_col:
        raise ValueError(f"No Zip column found. Columns: {df.columns.tolist()}")

    # 2. Clean Zips
    df[zip_col] = df[zip_col].astype(str).str.split('.').str[0].str.strip()
    df['clean_zip'] = df[zip_col].apply(lambda x: x.zfill(5))

    # 3. Merge Geo
    if MASTER_ZIP_DF.empty: raise ValueError("Master Geo Data not loaded.")
    
    merged = pd.merge(df, MASTER_ZIP_DF[['zip_code', 'latitude', 'longitude']], 
                      left_on='clean_zip', right_on='zip_code', how='inner')
    
    if merged.empty: raise ValueError("No matching Zip Codes found in Master DB.")

    # 4. Calculate Base Ratio Sum (Before multiplying by K)
    # Formula: Sum of [ (RowVal / ColTotal) * (UserPct / 100) ]
    merged['base_ratio_sum'] = 0.0

    for col, pct in column_config.items():
        if col not in merged.columns: raise ValueError(f"Column '{col}' missing.")
        series = pd.to_numeric(merged[col], errors='coerce').fillna(0)
        total = series.sum()
        if total > 0:
            merged['base_ratio_sum'] += (series / total) * (float(pct) / 100.0)

    # Ensure Lat/Long are floats
    merged['latitude'] = merged['latitude'].astype(float)
    merged['longitude'] = merged['longitude'].astype(float)
    
    return merged

# PART 2: SCENARIO RUNNER (Run for each K)
def run_scenario(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, dict]:
    # Work on a copy so we don't mess up other scenarios
    scenario_df = df.copy()
    
    # 1. Calculate Final Weight for this K
    # Formula: BaseRatio * (K * 1000)
    target_factor = k * 1000
    scenario_df['final_weight'] = scenario_df['base_ratio_sum'] * target_factor
    
    # 2. Clustering
    coords = scenario_df[['latitude', 'longitude']].fillna(0).values
    weights = scenario_df['final_weight'].fillna(0).values
    
    # Handle edge case where weights are zero
    sample_weight = weights if weights.sum() > 0 else None

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords, sample_weight=sample_weight)
    scenario_df['Territory_ID'] = kmeans.labels_

    # 3. Stats & Analysis
    status_map = {}
    for t_id in range(k):
        cluster_data = scenario_df[scenario_df['Territory_ID'] == t_id]
        total_w = cluster_data['final_weight'].sum()
        
        # Diameter
        diameter = 0
        if len(cluster_data) >= 2:
            try:
                mins = cluster_data[['latitude', 'longitude']].min()
                maxs = cluster_data[['latitude', 'longitude']].max()
                diameter = geodesic((mins.latitude, mins.longitude), (maxs.latitude, maxs.longitude)).miles
            except: pass

        # Status Logic (Target = 1000, +/- 25%)
        if total_w > 1250:
            st, msg = "Red", "Overloaded"
        elif total_w < 750:
            st, msg = "Yellow", "Underutilized"
        else:
            st, msg = "Green", "Optimal"

        status_map[t_id] = {
            "Status": st, "Message": msg, 
            "Weight": int(total_w), "ZipCount": len(cluster_data), "Diameter": int(diameter)
        }
        
    return scenario_df, status_map

# --- MAP GENERATOR (Returns HTML String) ---
def generate_map_html(df, stats, k, title):
    center_lat = df['latitude'].mean() if not df.empty else 39.8
    center_lon = df['longitude'].mean() if not df.empty else -98.5
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, prefer_canvas=True)
    
    # Add Title
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))

    colors = ["#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(k)]

    # Polygons
    for t_id in range(k):
        c_data = df[df['Territory_ID'] == t_id]
        if len(c_data) < 3: continue
        
        try:
            pts = c_data[['latitude', 'longitude']].values
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            
            s = stats.get(t_id, {})
            border = "red" if s.get('Status') == 'Red' else "black"
            
            folium.Polygon(
                locations=hull_pts, color=border, weight=2,
                fill=True, fill_color=colors[t_id], fill_opacity=0.4,
                popup=f"ID: {t_id}<br>W: {s.get('Weight')}<br>{s.get('Message')}"
            ).add_to(m)
            
            # Label
            folium.Marker(
                [c_data['latitude'].mean(), c_data['longitude'].mean()],
                icon=folium.DivIcon(html=f'<div style="font-size:10pt;font-weight:bold;">{t_id}</div>')
            ).add_to(m)
        except: pass

    # Dots
    for _, row in df.iterrows():
        folium.CircleMarker(
            [row['latitude'], row['longitude']], radius=2,
            color=colors[int(row['Territory_ID'])], fill=True, fill_opacity=1,
            popup=f"Zip: {row['clean_zip']} (W:{int(row['final_weight'])})"
        ).add_to(m)

    return m._repr_html_()

# --- ENDPOINTS ---

@app.post("/optimize_map")
async def get_visual_map_scenarios(
    file: UploadFile = File(...),
    num_clusters: int = Form(...),
    column_config: str = Form(...)
):
    try:
        # 1. Parse & Preprocess
        config = json.loads(column_config)
        content = await file.read()
        raw_df = read_file_smartly(content, file.filename)
        base_df = preprocess_data(raw_df, config)

        # 2. Define Scenarios
        k_original = int(num_clusters)
        k_low = max(2, k_original - 5) # Ensure K is never < 2
        k_high = k_original + 5
        
        scenarios = [
            ("Original", k_original),
            ("Minus_5", k_low),
            ("Plus_5", k_high)
        ]

        # 3. Generate Maps & Zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
            for name, k in scenarios:
                # Run Logic
                df_res, stats_res = run_scenario(base_df, k)
                # Generate HTML
                html_str = generate_map_html(df_res, stats_res, k, f"Scenario: {name} (K={k})")
                # Write to Zip
                zf.writestr(f"map_{name}_k{k}.html", html_str)

        zip_buffer.seek(0)
        headers = {'Content-Disposition': 'attachment; filename="territory_maps_comparison.zip"'}
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

    except Exception as e:
        return {"error": str(e)}

# @app.post("/optimize_excel")
# async def get_excel_scenarios(
#     file: UploadFile = File(...),
#     num_clusters: int = Form(...),
#     column_config: str = Form(...)
# ):
#     try:
#         config = json.loads(column_config)
#         content = await file.read()
#         raw_df = read_file_smartly(content, file.filename)
#         base_df = preprocess_data(raw_df, config)

#         k_original = int(num_clusters)
#         k_low = max(2, k_original - 5)
#         k_high = k_original + 5
        
#         scenarios = [
#             ("Original_K", k_original),
#             ("Low_K_Minus_5", k_low),
#             ("High_K_Plus_5", k_high)
#         ]

#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             for name, k in scenarios:
#                 df_res, _ = run_scenario(base_df, k)
                
#                 # Cleanup for friendly Excel
#                 export = df_res.copy()
#                 export.rename(columns={'Territory_ID': 'Cluster_ID', 'final_weight': 'Calculated_Weight'}, inplace=True)
#                 # Sort
#                 export.sort_values('Cluster_ID', inplace=True)
                
#                 # Write Sheet
#                 export.to_excel(writer, index=False, sheet_name=f"{name} (K={k})")

#         output.seek(0)
#         headers = {'Content-Disposition': 'attachment; filename="territory_scenarios.xlsx"'}
#         return Response(content=output.getvalue(), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)

#     except Exception as e:
#         return {"error": str(e)}

@app.post("/optimize_excel")
async def get_excel_scenarios(
    file: UploadFile = File(...),
    num_clusters: int = Form(...),
    column_config: str = Form(...)
):
    try:
        # 1. Parse & Preprocess
        config = json.loads(column_config)
        content = await file.read()
        raw_df = read_file_smartly(content, file.filename)
        base_df = preprocess_data(raw_df, config)

        # 2. Define Scenarios
        k_original = int(num_clusters)
        k_low = max(2, k_original - 5)
        k_high = k_original + 5
        
        scenarios = [
            ("Original", k_original),
            ("Minus_5", k_low),
            ("Plus_5", k_high)
        ]

        # 3. Build Consolidated DataFrame
        final_export_df = base_df.copy()

        # Loop through scenarios to append BOTH Cluster ID and Weight
        for name, k in scenarios:
            df_res, _ = run_scenario(base_df, k)
            
            # 1. Append Cluster ID Column
            id_col_name = f"Cluster_ID_{name}_(K={k})"
            final_export_df[id_col_name] = df_res['Territory_ID']
            
            # 2. Append Weight Column (New addition)
            weight_col_name = f"Weight_{name}_(K={k})"
            final_export_df[weight_col_name] = df_res['final_weight']

        # 4. Sorting & Cleanup
        # Sort by the Original Cluster ID for readability
        sort_col = f"Cluster_ID_Original_(K={k_original})"
        if sort_col in final_export_df.columns:
            final_export_df.sort_values(sort_col, inplace=True)

        # 5. Export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_export_df.to_excel(writer, index=False, sheet_name="Consolidated_Scenarios")

        output.seek(0)
        headers = {'Content-Disposition': 'attachment; filename="territory_scenarios_consolidated.xlsx"'}
        return Response(content=output.getvalue(), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)

    except Exception as e:
        return {"error": str(e)}