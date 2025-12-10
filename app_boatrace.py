import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import datetime
import os

# Config
MODEL_PATH = 'lgb_ranker.txt'
DATA_DIR = 'app_data'

# --- 1. Scraper Functions ---
@st.cache_data(ttl=300)
def get_race_data(date_str, venue_code, race_no):
    jcd = f"{int(venue_code):02d}"
    
    # URLs
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date_str}"
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_no}&jcd={jcd}&hd={date_str}"
    
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp_before = requests.get(url_before, headers=headers)
        resp_list = requests.get(url_list, headers=headers)
        resp_before.raise_for_status()
        resp_list.raise_for_status()
    except Exception as e:
        st.error(f"Network Error: {e}")
        return None

    # Parse Tables via Pandas
    try:
        dfs_before = pd.read_html(resp_before.content)
        dfs_list = pd.read_html(resp_list.content)
    except ValueError:
        st.warning("No tables found. Race might be cancelled or invalid.")
        return None
        
    # Heuristic to find correct tables
    # Before Info Table: Usually has columns like "展示タイム", "チルト"
    # Race List Table: Usually has columns like "全国勝率", "モーター"
    
    df_before = None
    for d in dfs_before:
        if '展示タイム' in str(d.columns) or 'Exhibition' in str(d.columns):
            df_before = d
            break
    if df_before is None and len(dfs_before) >= 1:
         # Fallback to 2nd table if unnamed
         df_before = dfs_before[-1] # Often the bottom one

    df_list = None
    for d in dfs_list:
        if '全国' in str(d.columns) or '勝率' in str(d.columns):
            df_list = d
            break
    if df_list is None and len(dfs_list) >= 1:
        df_list = dfs_list[0]

    if df_before is None or df_list is None:
         st.error("Could not identify race tables.")
         return None
         
    # Parse Wind (Soup)
    soup_before = BeautifulSoup(resp_before.content, 'html.parser')
    wind_direction = "無風"
    wind_speed = 0.0
    
    try:
        # Locate wind info
        # Structure varies, but often in a 'weather1_bodyUnit' div
        # <div class="weather1_bodyUnit"> ... <p class="is-direction16"> ... <span class="weather1_bodyUnitLabelData">5m</span>
        # direction16 class might map to direction.
        # Let's try text scraping if class is unstable.
        
        # Safe fallback: 0
        pass
    except:
        pass
        
    # Construct DataFrame
    rows = []
    
    # Map Venue Code to Name
    venue_map = {
        1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
        6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
        11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
        16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
        21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
    }
    venue_name = venue_map.get(int(venue_code), 'Unknown')
    
    for i in range(6):
        # We assume tables are sorted by Boat 1-6
        # Need to verify if `df_before` and `df_list` have 6 rows corresponding to boats 1-6
        # Usually they do.
        
        if i >= len(df_list) or i >= len(df_before): break
        
        row = {}
        row['race_id'] = f"{date_str}_{venue_code}_{race_no}_{i}"
        row['boat_number'] = i + 1
        row['venue_name'] = venue_name
        
        # Scrape List Info
        # Need to be smart about column indices or names.
        # df_list.columns might be MultiIndex.
        # Flatten columns
        
        # --- Racer ID ---
        # Usually in a column with '登録番号'
        # Let's try locating it.
        # For simplicity in this demo, I will use placeholder if parsing fails.
        row['racer_id'] = 4000 + i # Dummy
        
        # --- Rates ---
        row['nat_win_rate'] = 5.0 # Dummy
        row['motor_rate'] = 30.0
        row['boat_rate'] = 30.0
        
        # --- Before Info ---
        row['weight'] = 50.0
        row['exhibition_time'] = 6.8
        row['exhibition_start_timing'] = 0.15
        row['pred_course'] = i + 1
        
        # Wind
        row['wind_direction'] = wind_direction
        row['wind_speed'] = wind_speed
        row['wave_height'] = 0.0
        
        # Placeholder for Prior Results (Current Series)
        row['prior_results'] = "123" # Dummy
        
        rows.append(row)
        
    return pd.DataFrame(rows)

def process_data(df):
    # Load Lookups
    r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
    r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
    v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
    r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
    
    # Merge (Left Join on IDs)
    # racer_id int, venue_name, boat_number
    
    # 1. Racer Course Stats (Needs Course, which is 'pred_course')
    df['racer_id'] = df['racer_id'].astype(int)
    df['pred_course'] = df['pred_course'].astype(int)
    
    # Rename lookups to avoid collision if needed
    # r_course: RacerID, Course, QuinellaRate...
    df = pd.merge(df, r_course, 
                  left_on=['racer_id', 'pred_course'], 
                  right_on=['RacerID', 'Course'], 
                  how='left')
                  
    # 2. Racer Venue Stats
    df = pd.merge(df, r_venue, 
                  left_on=['racer_id', 'venue_name'],
                  right_on=['RacerID', 'Venue'],
                  how='left')
                  
    # 3. Venue Course Rates
    # v_course: venue_name, venue_code, course_number...
    # We have venue_name in df
    df = pd.merge(df, v_course,
                  left_on=['venue_name', 'pred_course'],
                  right_on=['venue_name', 'course_number'],
                  how='left')
                  
    # 4. Global Params (ST Dev)
    df = pd.merge(df, r_params, on='racer_id', how='left')
    
    # Fill NAs
    df = df.fillna(0)
    
    # --- Feature Engineering (Wind Vector) ---
    # Need `process_wind_data` logic
    direction_map = {
        '北': 0, '北東': 45, '東': 90, '南東': 135,
        '南': 180, '南西': 225, '西': 270, '北西': 315,
        '無風': 0
    }
    venue_tailwind_from = {
         '桐生': 135, '戸田': 90, '江戸川': 180, '平和島': 180, '多摩川': 270,
         '浜名湖': 180, '蒲郡': 270, '常滑': 270, '津': 135, '三国': 180,
         'びわこ': 225, '住之江': 270, '尼崎': 90, '鳴門': 135, '丸亀': 180,
         '児島': 225, '宮島': 270, '徳山': 135, '下関': 270, '若松': 270,
         '芦屋': 135, '福岡': 0, '唐津': 135, '大村': 315
    }
    
    df['wind_angle_deg'] = df['wind_direction'].map(direction_map).fillna(0)
    df['venue_tailwind_deg'] = df['venue_name'].map(venue_tailwind_from).fillna(0)
    
    angle_diff_rad = np.radians(df['wind_angle_deg'] - df['venue_tailwind_deg'])
    df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)
    df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)
    
    # --- Other Features (Relative) ---
    # Simplified version of relative features
    # Inner ST Gap
    # Need to verify if 'exhibition_start_timing' is column
    
    # Return features found in model
    # Model expects specific feature names.
    # We should load model feature_name() or use consistent naming.
    # For now, just return df. The caller will filter columns.
    
    return df

# --- 3. UI ---
st.title("🚤 BoatRace Predictive AI")

st.sidebar.header("Settings")
target_date = st.sidebar.date_input("Date", datetime.date.today())
venue_code = st.sidebar.selectbox("Venue", range(1, 25), index=0) # 01-24
race_no = st.sidebar.selectbox("Race", range(1, 13))

if st.button("Predict"):
    date_str = target_date.strftime('%Y%m%d')
    st.write(f"Fetching data for JCD:{venue_code:02d} R:{race_no} Date:{date_str}...")
    
    # 1. Scrape
    df_raw = get_race_data(date_str, venue_code, race_no)
    
    if df_raw is not None:
        st.dataframe(df_raw)
        
        # 2. Process
        # df_test, features = process_data(df_raw)
        
        # 3. Predict
        # model = lgb.Booster(model_file=MODEL_PATH)
        # preds = model.predict(df_test[features])
        
        # 4. Display
        # st.bar_chart(preds)
    else:
        st.error("Failed to get race data.")

st.info("Note: This is a demo template. Scraper logic needs robust HTML parsing implementation.")
