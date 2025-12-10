import os
# Force single thread to prevent Streamlit Cloud crashes (OpenMP)
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import datetime
import re
import time
import sys

# --- Configuration ---
st.set_page_config(page_title="BoatRace AI Predictor", layout="wide")

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
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Data Fetch Error: {e}")
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
            
        # Parse Wind & Weather
        weather = {'wind_direction': 0, 'wind_speed': 0.0, 'wave_height': 0.0}
        try:
            w = soup_before.select_one("div.weather1_body")
            if w:
                ws = w.select_one(".is-wind span.weather1_bodyUnitLabelData")
                if ws: weather['wind_speed'] = BoatRaceScraper.parse_float(ws.text)
                wh = w.select_one(".is-wave span.weather1_bodyUnitLabelData")
                if wh: weather['wave_height'] = BoatRaceScraper.parse_float(wh.text)
                wd = w.select_one(".is-windDirection p")
                if wd:
                    cls = wd.get('class', [])
                    d = next((c for c in cls if c.startswith('is-wind') and c != 'is-windDirection'), None)
                    if d: weather['wind_direction'] = int(re.sub(r'\D', '', d))
        except: pass

        # Parse Exhibition/ST
        boat_before = {}
        try:
            # Exhibition Time
            for i, tb in enumerate(soup_before.select("table.is-w748 tbody")):
                tds = tb.select("td")
                if len(tds) >= 5:
                    boat_before[i+1] = {'ex_time': BoatRaceScraper.parse_float(tds[4].text), 'st': 0.20}
            # ST
            for row in soup_before.select("table.is-w238 tbody tr"):
                bn = row.select_one("span.table1_boatImage1Number")
                if bn:
                    b = int(bn.text.strip())
                    st_span = row.select_one("span.table1_boatImage1Time")
                    val = 0.20
                    if st_span:
                        txt = st_span.text.strip().replace('F', '-0.')
                        if 'L' in txt: val = 1.0 # Late
                        elif 'F' in txt: val = -0.05 # Flying
                        else: val = float(txt)
                    if b in boat_before: boat_before[b]['st'] = val
                    else: boat_before[b] = {'st': val, 'ex_time': 6.8}
        except: pass

        # Parse List
        rows = []
        try:
            for i, tb in enumerate(soup_list.select("tbody.is-fs12")):
                bn = i + 1
                if bn > 6: break
                
                # Racer ID
                racer_id = 9999
                try: 
                    txt = tb.select("td")[2].select_one("div").get_text()
                    racer_id = int(re.search(r'(\d{4})', txt).group(1))
                except: pass

                # Branch (Prefecture)
                branch = 'Unknown'
                try:
                    txt = tb.select("td")[2].get_text(separator=' ')
                    # Look for text like "群馬" or "東京"
                    # Simple heuristic: regex for Japanese chars
                    pass
                except: pass

                # Rates
                motor = 30.0
                try: motor = BoatRaceScraper.parse_float(tb.select("td")[6].get_text())
                except: pass
                boat = 30.0
                try: boat = BoatRaceScraper.parse_float(tb.select("td")[7].get_text())
                except: pass
                
                row = {
                    'race_id': f"{date_str}_{venue_code}_{race_no}",
                    'boat_number': bn,
                    'racer_id': racer_id,
                    'motor_rate': motor,
                    'boat_rate': boat,
                    'exhibition_time': boat_before.get(bn, {}).get('ex_time', 6.8),
                    'exhibition_start_timing': boat_before.get(bn, {}).get('st', 0.20),
                    'pred_course': bn,
                    'wind_direction': weather['wind_direction'],
                    'wind_speed': weather['wind_speed'],
                    'wave_height': weather['wave_height'],
                    'prior_results': "3.5",
                    'branch': branch,
                    'makuri_count': 0, 'nige_count': 0
                }
                rows.append(row)
        except Exception as e:
            st.error(f"List Parse Error: {e}")
            return None
            
        return pd.DataFrame(rows)

# --- 2. Feature Engineer ---
class FeatureEngineer:
    @staticmethod
    def process(df, venue_name):
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
        except: pass
        
        df = df.fillna(0)
        
        # Wind Vector
        direction_map = {1:0,2:45,3:45,4:90,5:90,6:135,7:135,8:180,9:180,10:225,11:225,12:270,13:270,14:315,15:315,16:0}
        df['wind_angle_deg'] = df['wind_direction'].map(direction_map).fillna(0)
        df['wind_vector_long'] = df['wind_speed'] # Simplified
        df['wind_vector_lat'] = 0.0
        
        # Relative
        df['slit_formation'] = 0.0
        df['tenji_z_score'] = 0.0
        
        # Missing
        expected = ['series_avg_rank', 'makuri_rate', 'nige_rate', 'inner_st_gap', 
                    'anti_nige_potential', 'wall_strength', 'follow_potential', 'branch']
        for c in expected:
            if c not in df.columns: 
                if c == 'branch': df[c] = 'Unknown'
                else: df[c] = 0.0

        # Type Conversion for LightGBM
        # IMPORTANT: Model expects 'category' for object columns.
        ignore_cols = ['race_id', 'boat_number', 'racer_id', 'rank', 'venue_name', 'wind_direction', 'prior_results', 'syn_win_rate', 'exhibition_time']
        for col in df.columns:
            if col in ignore_cols: continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
                
        return df

# --- 3. Main App ---
st.title("🚤 BoatRace AI Strategy: 'Structure & Value'")
st.markdown("Returns-Focused AI Prediction System")

# Sidebar
today = datetime.date.today()
target_date = st.sidebar.date_input("Date", today)
venue_map = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
}
venue_code = st.sidebar.selectbox("Venue", list(venue_map.keys()), format_func=lambda x: f"{x:02d}: {venue_map[x]}")
venue_name = venue_map[venue_code]
race_no = st.sidebar.selectbox("Race No", range(1, 13))

if st.button("Analyze Race", type="primary"):
    date_str = target_date.strftime('%Y%m%d')
    st.info(f"Fetching Data: {venue_name} {race_no}R ({date_str})")
    
    # 1. Scrape
    with st.spinner("Scraping..."):
        df_race = BoatRaceScraper.get_race_data(date_str, venue_code, race_no)
    
    if df_race is not None:
        st.subheader("Live Race Data")
        cols = ['boat_number', 'racer_id', 'motor_rate', 'exhibition_time', 'exhibition_start_timing', 'wind_speed']
        st.dataframe(df_race[cols])
        
        # 2. Features
        with st.spinner("Processing..."):
            df_feat = FeatureEngineer.process(df_race, venue_name)
            
        # 3. Predict
        if os.path.exists(MODEL_PATH):
            try:
                model = lgb.Booster(model_file=MODEL_PATH)
                
                # Align columns
                model_feats = model.feature_name()
                X_pred = df_feat[model_feats]
                
                preds = model.predict(X_pred)
                df_feat['score'] = preds
                
                # 4. Result
                rank_df = df_feat[['boat_number', 'score']].sort_values('score', ascending=False)
                rank_df['rank'] = range(1, len(rank_df) + 1)
                
                st.divider()
                st.subheader("🤖 AI Prediction Ranking")
                st.dataframe(rank_df.set_index('rank'))
                
                # Top 5
                st.subheader("🎯 Top 5 Strategy")
                # Simple permutation of top 3 boats
                boats = rank_df['boat_number'].tolist()
                import itertools
                scores = dict(zip(rank_df['boat_number'], rank_df['score']))
                
                if len(boats) >= 3:
                    combos = list(itertools.permutations(boats, 3))
                    c_list = []
                    for c in combos:
                        # Product of scores as metric
                        s = scores[c[0]] * scores[c[1]] * scores[c[2]]
                        c_list.append({'combo': f"{c[0]}-{c[1]}-{c[2]}", 'val': s})
                    
                    df_c = pd.DataFrame(c_list).sort_values('val', ascending=False).head(5)
                    for i, row in df_c.iterrows():
                        st.metric(f"Rank {i+1}", row['combo'])
            except Exception as e:
                st.error(f"AI Model Error: {e}")
        else:
            st.warning("Model file (lgb_ranker.txt) not found.")
    else:
        st.error("Failed to load race data.")
