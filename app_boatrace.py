import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import datetime
import os
import re
import time

# --- Configuration ---
MODEL_PATH = 'lgb_ranker.txt'
DATA_DIR = 'app_data'

# Default Headers for Scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- 1. Scraper Class (Synchronous Adaptation of data_fetcher.py) ---
class BoatRaceScraper:
    @staticmethod
    def get_soup(url):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return BeautifulSoup(resp.text, 'html.parser')
        except Exception as e:
            st.error(f"Error fetching URL {url}: {e}")
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
        
        # URLs
        url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date_str}"
        url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_no}&jcd={jcd}&hd={date_str}"
        
        soup_before = BoatRaceScraper.get_soup(url_before)
        soup_list = BoatRaceScraper.get_soup(url_list)
        
        if not soup_before or not soup_list:
            return None
            
        # --- Parse Before Info (Weather & ST) ---
        weather_info = {
            'wind_direction': 0, 'wind_speed': 0.0, 'wave_height': 0.0, 
            'temperature': 0.0, 'water_temp': 0.0
        }
        
        try:
            w_el = soup_before.select_one("div.weather1_body")
            if w_el:
                # Wind Dir
                wd_p = w_el.select_one(".is-windDirection p")
                if wd_p:
                    cls = wd_p.get('class', [])
                    dir_cls = next((c for c in cls if c.startswith('is-wind') and c != 'is-windDirection'), None)
                    if dir_cls:
                        weather_info['wind_direction'] = int(re.sub(r'\D', '', dir_cls))
                
                # Wind Speed
                ws_span = w_el.select_one(".is-wind span.weather1_bodyUnitLabelData")
                if ws_span: weather_info['wind_speed'] = BoatRaceScraper.parse_float(ws_span.text)
                
                # Wave
                wh_span = w_el.select_one(".is-wave span.weather1_bodyUnitLabelData")
                if wh_span: weather_info['wave_height'] = BoatRaceScraper.parse_float(wh_span.text)
                
        except Exception as e:
            st.warning(f"Weather parse error: {e}")

        # Parse ST & Exhibition
        boat_before_data = {}
        try:
            tbody_st = soup_before.select("table.is-w238 tbody tr") # ST Table
            for row in tbody_st:
                bn_span = row.select_one("span.table1_boatImage1Number")
                if bn_span:
                    bn = int(bn_span.text.strip())
                    st_span = row.select_one("span.table1_boatImage1Time")
                    st_val = 0.0
                    if st_span:
                        txt = st_span.text.strip().replace('F', '-0.') # F.01 -> -0.01
                        # Simple F handling: just negative
                        if 'F' in st_span.text:
                             st_val = -0.05 # Penalty assumption
                        elif 'L' in st_span.text:
                             st_val = 1.0 # Late
                        else:
                             st_val = float(txt)
                    boat_before_data[bn] = {'st': st_val}
            
            # Exhibition Time (Main Table)
            tbody_ex = soup_before.select("table.is-w748 tbody")
            for i, tb in enumerate(tbody_ex):
                bn = i + 1
                # Exhibition time is usually 5th col
                tds = tb.select("td")
                if len(tds) >= 5:
                    ex_time = BoatRaceScraper.parse_float(tds[4].text)
                    if bn in boat_before_data:
                        boat_before_data[bn]['ex_time'] = ex_time
                    else:
                        boat_before_data[bn] = {'st': 0.20, 'ex_time': ex_time}
                        
        except Exception as e:
            st.warning(f"Before Info parse error: {e}")
            # Fallback
            for i in range(1, 7):
                if i not in boat_before_data: boat_before_data[i] = {'st': 0.20, 'ex_time': 6.8}

        # --- Parse Race List (Racer Info) ---
        rows = []
        try:
            tbodies = soup_list.select("tbody.is-fs12")
            for i, tb in enumerate(tbodies):
                bn = i + 1
                if bn > 6: break
                
                # Racer ID
                racer_id = 9999
                try:
                    # 3rd td usually has racer info
                    td_racer = tb.select("td")[2] 
                    div_top = td_racer.select_one("div")
                    if div_top:
                        txt = div_top.get_text()
                        match = re.search(r'(\d{4})', txt)
                        if match: racer_id = int(match.group(1))
                except: pass
                
                # Motor Rate
                motor_rate = 30.0
                try:
                    td_motor = tb.select("td")[6]
                    txt = td_motor.get_text(separator='|')
                    parts = txt.split('|')
                    if len(parts) >= 2:
                        motor_rate = BoatRaceScraper.parse_float(parts[1])
                except: pass
                
                # Boat Rate
                boat_rate = 30.0
                try:
                    td_boat = tb.select("td")[7]
                    txt = td_boat.get_text(separator='|')
                    parts = txt.split('|')
                    if len(parts) >= 2:
                        boat_rate = BoatRaceScraper.parse_float(parts[1])
                except: pass
                
                row = {
                    'race_id': f"{date_str}_{venue_code}_{race_no}",
                    'boat_number': bn,
                    'racer_id': racer_id,
                    'motor_rate': motor_rate,
                    'boat_rate': boat_rate,
                    'exhibition_time': boat_before_data.get(bn, {}).get('ex_time', 6.8),
                    'exhibition_start_timing': boat_before_data.get(bn, {}).get('st', 0.20),
                    'pred_course': bn, # Default to frame
                    # Weather
                    'wind_direction': weather_info['wind_direction'],
                    'wind_speed': weather_info['wind_speed'],
                    'wave_height': weather_info['wave_height'],
                    # Dummy for now (Needs robust parsing or API)
                    'prior_results': "3.5",
                    'branch': '', # Will fill later or parse
                    'nige_count': 0, 'makuri_count': 0 # Will be filled by static data if available
                }
                
                # Try Parse Branch from text
                try:
                   txt = tb.get_text()
                   # Simple heuristic: Branch is usually Prefecture or region name
                   # But creating a full valid list is hard. 
                   # data_fetcher uses regex on specific div.
                   pass
                except: pass

                rows.append(row)
        except Exception as e:
            st.error(f"Race List parse error: {e}")
            return None
            
        return pd.DataFrame(rows)

# --- 2. Feature Engineer ---
class FeatureEngineer:
    @staticmethod
    def process(df, venue_name):
        # Load Static Data
        try:
            r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
            r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
            v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
            r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
            
            # Type Casting
            df['racer_id'] = df['racer_id'].astype(int)
            df['pred_course'] = df['pred_course'].astype(int)
            
            # Merges
            df = df.merge(r_course, left_on=['racer_id', 'pred_course'], right_on=['RacerID', 'Course'], how='left')
            df = df.merge(r_venue, left_on=['racer_id'], right_on=['RacerID'], how='left') 
            df = df.merge(v_course, left_on=['pred_course'], right_on=['course_number'], how='left')
            df = df.merge(r_params, on='racer_id', how='left')
            
            df = df.fillna(0)
            
        except Exception as e:
            st.warning(f"Static data merge error: {e}. Using raw features only.")
        
        # --- Wind Vector ---
        direction_map = {
            1: 0, 2: 45, 3: 45, 4: 90, 5: 90, 6: 135,
            7: 135, 8: 180, 9: 180, 10: 225, 11: 225, 12: 270,
            13: 270, 14: 315, 15: 315, 16: 0
        }
        
        venue_tailwind_from = {
             '桐生': 135, '戸田': 90, '江戸川': 180, '平和島': 180, '多摩川': 270,
             '浜名湖': 180, '蒲郡': 270, '常滑': 270, '津': 135, '三国': 180,
             'びわこ': 225, '住之江': 270, '尼崎': 90, '鳴門': 135, '丸亀': 180,
             '児島': 225, '宮島': 270, '徳山': 135, '下関': 270, '若松': 270,
             '芦屋': 135, '福岡': 0, '唐津': 135, '大村': 315
        }
        
        df['wind_angle_deg'] = df['wind_direction'].map(direction_map).fillna(0)
        tailwind_deg = venue_tailwind_from.get(venue_name, 0)
        
        angle_diff_rad = np.radians(df['wind_angle_deg'] - tailwind_deg)
        df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)
        df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)
        
        # --- Relative Stats ---
        df = df.sort_values('boat_number')
        df['inner_st'] = df['exhibition_start_timing'].shift(1).fillna(0)
        df['outer_st'] = df['exhibition_start_timing'].shift(-1).fillna(0)
        df['slit_formation'] = df['exhibition_start_timing'] - ((df['inner_st']+df['outer_st'])/2)
        
        mean_t = df['exhibition_time'].mean()
        std_t = df['exhibition_time'].std()
        if std_t == 0: std_t = 1.0
        df['tenji_z_score'] = (mean_t - df['exhibition_time']) / std_t
        
        # --- Mock Missing Columns (Matching Training Logic) ---
        expected_cols = [
            'series_avg_rank', 'makuri_rate', 'nige_rate', 'inner_st_gap', 
            'anti_nige_potential', 'wall_strength', 'follow_potential',
            'branch' # Ensure branch exists
        ]
        for c in expected_cols:
            if c not in df.columns: 
                if c == 'branch': df[c] = 'Unknown'
                else: df[c] = 0.0
                
        # --- LightGBM Prep: Category & Ignore Cols ---
        ignore_cols = [
            'race_id', 'boat_number', 'racer_id', 'rank',
            'venue_name', 'wind_direction', 'prior_results',
            'syn_win_rate', 'exhibition_time' # exhibition_time might be kept? Check train_model. 
            # train_model ignore_cols: race_id, boat_number, racer_id, rank, venue_name, wind_direction, prior_results, syn_win_rate.
            # exhibition_time IS kept (it's numeric).
        ]
        
        # Explicit drop of ignored cols
        # Also need to ensure object cols are category
        for col in df.columns:
            if col in ignore_cols: continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
                
        return df, ignore_cols

# --- 3. Main App ---
st.set_page_config(page_title="BoatRace AI Predictor", layout="wide")

st.title("🚤 BoatRace AI Strategy: 'Structure & Value'")
st.markdown("Returns-Focused AI Prediction System")

# Sidebar
st.sidebar.header("Race Selection")
target_date = st.sidebar.date_input("Date", datetime.date.today())
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
    with st.spinner("Scraping official website..."):
        df_race = BoatRaceScraper.get_race_data(date_str, venue_code, race_no)
    
    if df_race is not None:
        # Show Live Data
        st.subheader("Live Race Data")
        cols = ['boat_number', 'racer_id', 'motor_rate', 'boat_rate', 'exhibition_time', 'exhibition_start_timing', 'wind_speed', 'wave_height']
        st.dataframe(df_race[cols])
        
        # 2. Feature Engineering
        with st.spinner("Processing AI Features..."):
            df_features, ignore_list = FeatureEngineer.process(df_race, venue_name)
            
        # 3. Prediction
        with st.spinner("Running LightGBM Inference..."):
            try:
                model = lgb.Booster(model_file=MODEL_PATH)
                # Filter columns used in training
                # Best way: model.feature_name()
                model_features = model.feature_name()
                
                # Check missing
                missing_feats = [f for f in model_features if f not in df_features.columns]
                if missing_feats:
                    st.warning(f"Missing features filled with defaults: {missing_feats}")
                    for mf in missing_feats:
                        df_features[mf] = 0.0 # or appropriate default
                        
                # Predict using ONLY model features
                X_pred = df_features[model_features]
                
                # Ensure categories match if possible? 
                # LGBM handles int/category matching loosely if consistent.
                
                preds = model.predict(X_pred)
                df_features['score'] = preds
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                # Fallback score
                df_features['score'] = df_features['motor_rate'] 
                
        # 4. Display Results
        st.divider()
        st.subheader("🤖 AI Prediction Ranking")
        
        rank_df = df_features[['boat_number', 'score']].sort_values('score', ascending=False)
        rank_df['rank'] = range(1, len(rank_df) + 1)
        st.dataframe(rank_df.set_index('rank'))
        
        # 5. Betting Strategy (Top 5 Pattern)
        st.subheader("🎯 Recommended Strategy (Top 5 Pattern)")
        
        boats = rank_df['boat_number'].tolist()
        import itertools
        # Generate combos based on Score Product
        combo_scores = []
        boat_score_map = dict(zip(rank_df['boat_number'], rank_df['score']))
        
        # Only if we have at least 3 boats
        if len(boats) >= 3:
            combos = list(itertools.permutations(boats, 3))
            for c in combos:
                s = boat_score_map[c[0]] * boat_score_map[c[1]] * boat_score_map[c[2]]
                combo_scores.append({'combo': f"{c[0]}-{c[1]}-{c[2]}", 'score': s})
                
            combo_df = pd.DataFrame(combo_scores).sort_values('score', ascending=False).head(5)
            
            st.success("✅ BUY THESE 5 POINTS (Flat Bet)")
            for idx, row in combo_df.iterrows():
                st.metric(label=f"Rank {idx+1}", value=row['combo'])
        else:
            st.warning("Not enough boats to generate 3-rentan combos.")
            
    else:
        st.error("Failed to load race data. Race might be invalid or cancelled.")
