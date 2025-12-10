import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime
import re
import time
import sys
import os

# --- Configuration ---
st.set_page_config(page_title="BoatRace AI Predictor", layout="wide")

# Force single thread (Safety)
os.environ['OMP_NUM_THREADS'] = '1'

# --- Logging ---
def log(msg):
    # Print to logs
    print(f"[LOG] {msg}", file=sys.stdout, flush=True)

log("App Initializing...")

MODEL_PATH = 'lgb_ranker.txt'
DATA_DIR = 'app_data'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- 1. Scraper Class ---
class BoatRaceScraper:
    @staticmethod
    def get_soup(url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15) # 15s timeout
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                log(f"Fetch Error ({url}): {e}")
                if attempt == max_retries - 1:
                    st.error(f"Failed to fetch data: {e}")
                    return None
                time.sleep(1)
        return None

    @staticmethod
    def parse_float(text):
        try:
            return float(re.search(r'([\d\.]+)', text).group(1))
        except:
            return 0.0

    @staticmethod
    def get_race_data(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date_str}"
        url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_no}&jcd={jcd}&hd={date_str}"
        
        soup_before = BoatRaceScraper.get_soup(url_before)
        soup_list = BoatRaceScraper.get_soup(url_list)
        
        if not soup_before or not soup_list:
            return None
            
        # Parse Wind
        weather_info = {'wind_direction': 0, 'wind_speed': 0.0, 'wave_height': 0.0}
        try:
            w_el = soup_before.select_one("div.weather1_body")
            if w_el:
                # Wind Speed
                ws_span = w_el.select_one(".is-wind span.weather1_bodyUnitLabelData")
                if ws_span: weather_info['wind_speed'] = BoatRaceScraper.parse_float(ws_span.text)
                # Wave
                wh_span = w_el.select_one(".is-wave span.weather1_bodyUnitLabelData")
                if wh_span: weather_info['wave_height'] = BoatRaceScraper.parse_float(wh_span.text)
                # Wind Dir
                wd_p = w_el.select_one(".is-windDirection p")
                if wd_p:
                    cls = wd_p.get('class', [])
                    dir_cls = next((c for c in cls if c.startswith('is-wind') and c != 'is-windDirection'), None)
                    if dir_cls: weather_info['wind_direction'] = int(re.sub(r'\D', '', dir_cls))
        except Exception as e:
            log(f"Weather Parse Error: {e}")

        # Parse Exhibition/ST
        boat_before_data = {}
        try:
            # Exhibition Time (Table)
            tbody_ex = soup_before.select("table.is-w748 tbody")
            for i, tb in enumerate(tbody_ex):
                bn = i + 1
                tds = tb.select("td")
                if len(tds) >= 5:
                    ex_time = BoatRaceScraper.parse_float(tds[4].text)
                    boat_before_data[bn] = {'st': 0.20, 'ex_time': ex_time}
            
            # ST (Table)
            tbody_st = soup_before.select("table.is-w238 tbody tr")
            for row in tbody_st:
                bn_span = row.select_one("span.table1_boatImage1Number")
                if bn_span:
                    bn = int(bn_span.text.strip())
                    st_span = row.select_one("span.table1_boatImage1Time")
                    st_val = 0.0
                    if st_span:
                        txt = st_span.text.strip().replace('F', '-0.')
                        if 'F' in st_span.text: st_val = -0.05
                        elif 'L' in st_span.text: st_val = 1.0
                        else: st_val = float(txt)
                    if bn in boat_before_data: boat_before_data[bn]['st'] = st_val
                    else: boat_before_data[bn] = {'st': st_val, 'ex_time': 6.8}
        except Exception as e:
            log(f"BeforeInfo Parse Error: {e}")

        # Determine if data is dummy (e.g. race cancelled or not published)
        # Verify valid entries
        rows = []
        try:
            tbodies = soup_list.select("tbody.is-fs12")
            for i, tb in enumerate(tbodies):
                bn = i + 1
                if bn > 6: break
                
                racer_id = 9999
                try:
                    td_racer = tb.select("td")[2]
                    txt = td_racer.select_one("div").get_text()
                    racer_id = int(re.search(r'(\d{4})', txt).group(1))
                except: pass
                
                motor_rate = 30.0
                try:
                    motor_rate = BoatRaceScraper.parse_float(tb.select("td")[6].get_text())
                except: pass
                
                boat_rate = 30.0
                try:
                    boat_rate = BoatRaceScraper.parse_float(tb.select("td")[7].get_text())
                except: pass
                
                row = {
                    'race_id': f"{date_str}_{venue_code}_{race_no}",
                    'boat_number': bn,
                    'racer_id': racer_id,
                    'motor_rate': motor_rate,
                    'boat_rate': boat_rate,
                    'exhibition_time': boat_before_data.get(bn, {}).get('ex_time', 6.8),
                    'exhibition_start_timing': boat_before_data.get(bn, {}).get('st', 0.20),
                    'pred_course': bn,
                    'wind_direction': weather_info['wind_direction'],
                    'wind_speed': weather_info['wind_speed'],
                    'wave_height': weather_info['wave_height'],
                    'prior_results': "3.5",
                    'branch': 'Unknown',
                    'nige_count': 0, 'makuri_count': 0
                }
                rows.append(row)
        except Exception as e:
            log(f"Race List Parse Error: {e}")
            return None
            
        return pd.DataFrame(rows)

# --- 2. Feature Engineer ---
class FeatureEngineer:
    @staticmethod
    def process(df, venue_name):
        # ... (Same logic, condensed) ...
        # Ensure imports if needed (pd, np is global)
        # Mock merges for robustness if files missing
        try:
            r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
            r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
            v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
            r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
            
            df['racer_id'] = df['racer_id'].astype(int)
            df['pred_course'] = df['pred_course'].astype(int)
            
            df = df.merge(r_course, left_on=['racer_id', 'pred_course'], right_on=['RacerID', 'Course'], how='left')
            df = df.merge(r_venue, left_on=['racer_id'], right_on=['RacerID'], how='left') 
            df = df.merge(v_course, left_on=['pred_course'], right_on=['course_number'], how='left')
            df = df.merge(r_params, on='racer_id', how='left')
        except:
            log("Static Data Load/Merge Failed. Using Defaults.")
        
        df = df.fillna(0)
        
        # Wind Vector
        direction_map = {1:0,2:45,3:45,4:90,5:90,6:135,7:135,8:180,9:180,10:225,11:225,12:270,13:270,14:315,15:315,16:0}
        df['wind_angle_deg'] = df['wind_direction'].map(direction_map).fillna(0)
        
        # Simplified Vector
        df['wind_vector_long'] = df['wind_speed'] # Simplification for robustness
        df['wind_vector_lat'] = 0.0
        
        # Relative
        df['slit_formation'] = 0.0 # Simplify
        df['tenji_z_score'] = 0.0
        
        # Missing Cols
        expected = ['series_avg_rank', 'makuri_rate', 'nige_rate', 'inner_st_gap', 
                    'anti_nige_potential', 'wall_strength', 'follow_potential', 'branch']
        for c in expected:
            if c not in df.columns: 
                if c == 'branch': df[c] = 'Unknown'
                else: df[c] = 0.0
                
        # Prep for LGBM (Category)
        ignore_cols = ['race_id', 'boat_number', 'racer_id', 'rank', 'venue_name', 'wind_direction', 'prior_results', 'syn_win_rate', 'exhibition_time']
        for col in df.columns:
            if col in ignore_cols: continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        
        return df, ignore_cols

# --- 3. UI ---
st.title("🚤 BoatRace AI (Diagnostic Mode)")

# Inputs
today = datetime.date.today()
target_date = st.sidebar.date_input("Date", today)
venue_code = st.sidebar.selectbox("Place Code", range(1, 25))
race_no = st.sidebar.selectbox("Race No", range(1, 13))

if st.button("Predict"):
    try:
        # 1. Scrape
        date_str = target_date.strftime('%Y%m%d')
        st.write(f"Scraping... {date_str} JCD:{venue_code} R:{race_no}")
        df = BoatRaceScraper.get_race_data(date_str, venue_code, race_no)
        
        if df is None:
            st.error("Scraping failed or no data.")
        else:
            st.dataframe(df)
            
            # 2. Process
            df_feat, _ = FeatureEngineer.process(df, "Venue")
            
            # 3. Predict (LAZY IMPORT)
            st.write("Loading Model completely locally inside function...")
            import lightgbm as lgb
            
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model file not found: {MODEL_PATH}")
                st.stop()
                
            model = lgb.Booster(model_file=MODEL_PATH)
            
            # Predict
            features = model.feature_name()
            # Fill missing
            for f in features:
                if f not in df_feat.columns: df_feat[f] = 0.0
            
            preds = model.predict(df_feat[features])
            df_feat['score'] = preds
            
            # Rank
            res = df_feat[['boat_number', 'score']].sort_values('score', ascending=False)
            st.subheader("Result")
            st.dataframe(res)
            
    except Exception as e:
        st.error(f"Runtime Error: {e}")
        log(f"Runtime Crash: {e}")
