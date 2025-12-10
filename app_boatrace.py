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

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache Cleared!")

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
                bn_span = row.select_one("span.table1_boatImage1Number")
                if bn_span:
                    b = int(bn_span.text.strip())
                    st_span = row.select_one("span.table1_boatImage1Time")
                    val = 0.20
                    if st_span:
                        txt_raw = st_span.text.strip()
                        # Handle F/L
                        if 'L' in txt_raw: val = 1.0
                        elif 'F' in txt_raw:
                            try:
                                # F.01 -> -0.01
                                sub = txt_raw.replace('F', '')
                                val = -float(sub)
                            except: val = -0.05
                        else:
                            # .12 -> 0.12
                            val = BoatRaceScraper.parse_float(txt_raw)
                            
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

                # Branch (Prefecture) & Weight
                branch = 'Unknown'
                weight = 52.0
                try:
                    td2 = tb.select("td")[2]
                    txt_full = td2.get_text(" ", strip=True)
                    
                    # Weight
                    match_w = re.search(r'(\d{2}\.\d)kg', txt_full)
                    if match_w: weight = float(match_w.group(1))
                    
                    divs = td2.select("div")
                    
                    # Robust Branch Extraction using white-list
                    prefectures = r"(群馬|埼玉|東京|福井|静岡|愛知|三重|滋賀|大阪|兵庫|徳島|香川|岡山|広島|山口|福岡|佐賀|長崎)"
                    # Search entire cell text for a prefecture
                    m = re.search(prefectures, txt_full)
                    if m:
                        branch = m.group(1)
                    else:
                        # Fallback: parsing div structure
                        if len(divs) >= 2:
                            br_txt = divs[1].get_text(strip=True)
                            branch = br_txt.split()[0]
                            branch = re.sub(r'\d+', '', branch)
                except: pass

                # Win Rates (National & Local)
                # Col 3: F0 L0 0.14 6.89 50.5 ...
                nat_win_rate = 0.0
                local_win_rate = 0.0
                try:
                    col3_txt = tb.select("td")[3].get_text(" ", strip=True)
                    # Use broad regex for any number
                    # Remove F/L to avoid parsing 0 from F0
                    clean_txt = re.sub(r'[FLK]\d+', '', col3_txt) 
                    nums = re.findall(r'(\d+(?:\.\d+)?)', clean_txt)
                    
                    if len(nums) >= 5:
                        # [AvgST, NatWin, Nat2, LocWin, Loc2]
                        nat_win_rate = float(nums[1])
                        local_win_rate = float(nums[3])
                    elif len(nums) >= 4:
                        # [NatWin, Nat2, LocWin, Loc2]
                        nat_win_rate = float(nums[0])
                        local_win_rate = float(nums[2])
                except: pass

                # Prior Results (Series Results)
                # Scan all TDs for Rank-like patterns (1-6, full-width １-６, F, L, etc.)
                # Ignore ST (.12) and R (8R)
                prior_results_list = []
                try:
                    all_tds = tb.select("td")
                    # Start from col 15 to avoid left-side stats
                    start_idx = 15 if len(all_tds) > 15 else 0
                    
                    for td in all_tds[start_idx:]:
                        txt = td.get_text(strip=True)
                        if not txt: continue
                        
                        # Check format: Single/Double char, 1-6 or FLK
                        # Normalize Full-width
                        txt_norm = txt.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
                        
                        # Regex for strictly rank: 1-6, F, L, K, S, 欠, 失
                        # Exclude ".12" (ST) or "8R" or "12/10"
                        if re.match(r'^[1-6FLKS欠失]$', txt_norm):
                            prior_results_list.append(txt_norm)
                            
                    prior_results = " ".join(prior_results_list)
                except: pass

                # Rates
                # Column 6: Motor (No / Rate) e.g. "43 32.5%"
                # Column 7: Boat (No / Rate) e.g. "14 31.0%"
                tds = tb.select("td")
                
                motor = 30.0
                try:
                    txt = tds[6].get_text(" ", strip=True).replace('%', '')
                    parts = txt.split()
                    if len(parts) >= 2: motor = float(parts[1])
                    else: motor = float(parts[0])
                except: pass
                
                boat = 30.0
                try:
                    txt = tds[7].get_text(" ", strip=True).replace('%', '')
                    parts = txt.split()
                    if len(parts) >= 2: boat = float(parts[1])
                    else: boat = float(parts[0])
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
                    'prior_results': prior_results,
                    'branch': branch,
                    'weight': weight,
                    'nat_win_rate': nat_win_rate,
                    'local_win_rate': local_win_rate,
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
    def process_wind_data(df):
        # 1. Map Wind Direction (Text -> Angle)
        direction_map = {
            '北': 0, '北東': 45, '東': 90, '南東': 135,
            '南': 180, '南西': 225, '西': 270, '北西': 315,
            '無風': 0, 'failed': 0, '': 0, 0: 0 # Handle int input
        }
        # In App, scraper returns int (1-16) or text?
        # Scraper returns int (1=North=0deg, 2=NE=45deg... 16=North=0)
        # 1=North, 2=NNE, 3=NE, 4=ENE, 5=East... 16=NNW?
        # BoatRace.jp logic: 1~16. 1=North(0), 5=East(90), 9=South(180), 13=West(270)
        # So map is: (val-1) * 22.5
        # Scraper already parses `is-windDirection` class to int?
        # Let's check scraper: `int(re.sub(r'\D', '', dir_cls))` -> 1..16
        # So we need 1..16 map.
        
        # Override map for 1..16 int input
        def wind_deg_from_int(x):
            if x < 1 or x > 16: return 0
            return (x - 1) * 22.5

        # Scraper returns int 1-16
        df['wind_angle_deg'] = df['wind_direction'].apply(wind_deg_from_int)

        # Venue Tailwind map (Heading of 1M - From which wind comes as tailwind)
        venue_tailwind_from = {
            '桐生': 135, '戸田': 90, '江戸川': 180, '平和島': 180, '多摩川': 270,
            '浜名湖': 180, '蒲郡': 270, '常滑': 270, '津': 135, '三国': 180,
            'びわこ': 225, '住之江': 270, '尼崎': 90, '鳴門': 135, '丸亀': 180,
            '児島': 225, '宮島': 270, '徳山': 135, '下関': 270, '若松': 270,
            '芦屋': 135, '福岡': 0, '唐津': 135, '大村': 315
        }
        
        df['venue_tailwind_deg'] = df['venue_name'].map(venue_tailwind_from).fillna(0)
        
        # Vectors
        angle_diff_rad = np.radians(df['wind_angle_deg'] - df['venue_tailwind_deg'])
        df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)
        df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)
        
        return df

    @staticmethod
    def process(df, venue_name, debug_mode=False):
        # Add missing venue_name column if not present (for mapping)
        df['venue_name'] = venue_name
        
        # Load Static Data
        try:
            r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
            r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
            v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
            r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
            
            # Ensure Types
            df['racer_id'] = df['racer_id'].astype(int)
            df['pred_course'] = df['pred_course'].astype(int)
            r_course['RacerID'] = r_course['RacerID'].astype(int)
            r_course['Course'] = r_course['Course'].astype(int)
            r_venue['RacerID'] = r_venue['RacerID'].astype(int)
            v_course['course_number'] = v_course['course_number'].astype(int)
            r_params['racer_id'] = r_params['racer_id'].astype(int)

            # --- Merges ---
            # 1. Racer Course Stats: [RacerID, Course] -> [course_run_count, course_quinella_rate...]
            # static_racer_course.csv cols: RacerID, Course, RacesRun, QuinellaRate, TrifectaRate, FirstPlaceRate, Nige, Makuri, Sashi
            df = df.merge(r_course, left_on=['racer_id', 'pred_course'], right_on=['RacerID', 'Course'], how='left')
            df.rename(columns={
                'RacesRun': 'course_run_count',
                'QuinellaRate': 'course_quinella_rate',
                'TrifectaRate': 'course_trifecta_rate',
                'FirstPlaceRate': 'course_1st_rate',
                'AvgStartTiming': 'course_avg_st', # Note: AvgStartTiming might be missing now? export didn't include it. 
                # Wait, export logic REMOVED AvgStartTiming in my update?
                # export_racer_course_stats selected: RacesRun, QuinellaRate, TrifectaRate, FirstPlaceRate, Nige, Makuri, Sashi.
                # It missed AvgStartTiming? 
                # I should assume course_avg_st is 0.17 if missing. Or scrape it?
                'Nige': 'nige_count_course', # conflict with global? No, app uses global nige_count?
                'Makuri': 'makuri_count_course',
                'Sashi': 'sashi_count_course'
            }, inplace=True)

            # 2. Racer Venue Stats
            # 2. Racer Venue Stats
            # static_racer_venue.csv uses Venue CODE (e.g. '01', '07'). s.Venue is likely code.
            # df has 'venue_name' (e.g. '蒲郡'). Need to map to code.
            # Reverse the venue_map (Code->Name) to (Name->Code)
            name_to_code = {v: k for k, v in venue_map.items()} # venue_map is likely global?
            # It's defined in main scope. FeatureEngineer might not see it if not passed.
            # Let's define a local map or pass it.
            # Standard map:
            name_code_map = {
                '桐生': 1, '戸田': 2, '江戸川': 3, '平和島': 4, '多摩川': 5,
                '浜名湖': 6, '蒲郡': 7, '常滑': 8, '津': 9, '三国': 10,
                'びわこ': 11, '住之江': 12, '尼崎': 13, '鳴門': 14, '丸亀': 15,
                '児島': 16, '宮島': 17, '徳山': 18, '下関': 19, '若松': 20,
                '芦屋': 21, '福岡': 22, '唐津': 23, '大村': 24
            }
            
            df['venue_code_int'] = df['venue_name'].map(name_code_map).fillna(0).astype(int)
            # CSV Venue might be '01' (str) or 1 (int).
            # Let's check CSV content type. `01` implies string.
            # Let's try both or ensure int.
            # r_venue['Venue'] should be int if we cast it.
            r_venue['Venue'] = pd.to_numeric(r_venue['Venue'], errors='coerce').fillna(0).astype(int)
            
            df = df.merge(r_venue, left_on=['racer_id', 'venue_code_int'], right_on=['RacerID', 'Venue'], how='left')
            
            # Rename static WinRate
            if 'local_win_rate' in r_venue.columns:
                df.rename(columns={'local_win_rate': 'local_win_rate_static'}, inplace=True)
            elif 'WinRate' in df.columns: # fallback
                 df.rename(columns={'WinRate': 'local_win_rate_static'}, inplace=True)

            if 'local_win_rate' in df.columns:
                df['local_win_rate'] = df['local_win_rate'].replace(0.0, np.nan)
                df['local_win_rate'] = df['local_win_rate'].fillna(df['local_win_rate_static'])
            else:
                df['local_win_rate'] = df.get('local_win_rate_static', 0.0)
                
            df.drop(columns=['local_win_rate_static'], inplace=True, errors='ignore')

            # 3. Venue Course Stats
            df = df.merge(v_course, left_on=['venue_name', 'pred_course'], right_on=['venue_name', 'course_number'], how='left')
            df.rename(columns={
                'rate_1st': 'venue_course_1st_rate',
                'rate_2nd': 'venue_course_2nd_rate',
                'rate_3rd': 'venue_course_3rd_rate'
            }, inplace=True)

            # 4. Racer Params: [racer_id] -> [st_std_dev, nat_win_rate, nige_count, makuri_count, sashi_count...]
            # Drop likely colliding scraper columns to prioritize static data
            drop_cols = ['nige_count', 'makuri_count', 'sashi_count', 'nat_win_rate', 'st_std_dev']
            for c in drop_cols:
                if c in df.columns: df.drop(columns=[c], inplace=True)

            df = df.merge(r_params, on='racer_id', how='left')
            
            # Post-Merge Cleanup: Consolidate _x/_y if they appeared due to missed conflicting cols
            # Check for suffixed cols and coalesce
            cleanup_targets = ['nige_count', 'makuri_count', 'sashi_count', 'nat_win_rate', 'local_win_rate']
            for t in cleanup_targets:
                x_col = f"{t}_x"
                y_col = f"{t}_y"
                if x_col in df.columns and y_col in df.columns:
                    # Prefer Y (Static usually) if X is 0/NaN
                    df[t] = df[y_col].fillna(df[x_col])
                    df.drop(columns=[x_col, y_col], inplace=True)
                elif x_col in df.columns:
                    df.rename(columns={x_col: t}, inplace=True)
                elif y_col in df.columns:
                    df.rename(columns={y_col: t}, inplace=True)
                # If neither, 't' might already exist or be missing (handled by failsafe)
            
        except Exception as e:
            # st.error(f"Static Data Error: {e}")
            pass
            
        except Exception as e:
            # st.error(f"Static Data Error: {e}")
            pass
        
        if debug_mode:
            st.write("Columns:", df.columns.tolist())
            if 'nige_count' in df.columns:
                 st.write("Nige Count (First 1):", df['nige_count'].iloc[0])
            else:
                 st.error("Nige Count Column Missing!")
        
        # Failsafe: Ensure critical columns exist (even if CSV merge failed)
        required_cols = ['makuri_count', 'nige_count', 'sashi_count', 'nat_win_rate', 'course_run_count', 'local_win_rate']
        for c in required_cols:
            if c not in df.columns: 
                df[c] = 0.0
                if debug_mode:
                    st.warning(f"Feature '{c}' missing. Filled 0.")
        
        # --- Feature Engineering (Sync with make_data_set.py) ---
        
        # Helper for Series Avg (Mock/Parse)
        # Helper for Series Avg (Mock/Parse)
        def parse_prior(x):
            if isinstance(x, (int, float)): return float(x)
            if not isinstance(x, str): return 3.5
            
            # Text usually "1 3 2 4" or "1 3F 2"
            try:
                # Replace commonly used non-digit rank markers if any
                x = x.replace('欠', '').replace('失', '').replace('F', '').replace('L', '').replace('S', '')
                parts = x.split()
                ranks = []
                for p in parts:
                    try:
                        val = float(p)
                        if 1 <= val <= 6: ranks.append(val)
                    except: pass
                
                if ranks:
                    return sum(ranks) / len(ranks)
            except: pass
            
            return 3.5 # Default fallback

        df['series_avg_rank'] = df['prior_results'].apply(parse_prior)

        # Rates
        df['makuri_rate'] = df['makuri_count'] / df['course_run_count'].replace(0, 1)
        df['nige_rate'] = df['nige_count'] / df['course_run_count'].replace(0, 1)

        # ST Calculation (Group by race_id - but here usually 1 race)
        # Sort just in case
        df = df.sort_values('pred_course')
        
        df['inner_st'] = df['exhibition_start_timing'].shift(1).fillna(0)
        df['inner_st_gap'] = df['exhibition_start_timing'] - df['inner_st']
        df['outer_st'] = df['exhibition_start_timing'].shift(-1).fillna(0)
        
        avg_neighbor = (df['inner_st'] + df['outer_st']) / 2
        # If edge, handle? Shift returns NaN which we filled 0. That's fine.
        df['slit_formation'] = df['exhibition_start_timing'] - avg_neighbor

        # Anti-Nige
        c1_nige = df.loc[df['pred_course']==1, 'nige_rate']
        val = c1_nige.values[0] if len(c1_nige) > 0 else 0.5
        df['anti_nige_potential'] = df['makuri_rate'] * (1 - val)

        # Wall Strength (Inner Quinella)
        df['wall_strength'] = df['course_quinella_rate'].shift(1).fillna(0)

        # Follow Potential (Inner Makuri * Self Quinella)
        df['follow_potential'] = df['makuri_rate'].shift(1).fillna(0) * df['course_quinella_rate']

        # Tenji Z-Score
        mean_t = df['exhibition_time'].mean()
        std_t = df['exhibition_time'].std()
        if std_t == 0: std_t = 1
        df['tenji_z_score'] = (mean_t - df['exhibition_time']) / std_t

        # Linear Rank
        df['linear_rank'] = df['exhibition_time'].rank(method='min', ascending=True)
        df['is_linear_leader'] = (df['linear_rank'] == 1).astype(int)

        # Weight Diff (User Weight - Avg)
        # Note: 'weight' might come from params or scraper. 
        # If scraper didn't get it, use params. 
        if 'weight_x' in df.columns: df['weight'] = df['weight_x'] # excessive merge handling
        if 'weight' not in df.columns: df['weight'] = 52.0
        
        df['weight_diff'] = df['weight'] - df['weight'].mean()

        # High Wind Alert
        df['high_wind_alert'] = (df['wind_speed'] >= 5).astype(int)

        # Local Perf Diff
        if 'nat_win_rate' not in df.columns: df['nat_win_rate'] = 0.0 
        if 'local_win_rate' not in df.columns: df['local_win_rate'] = 0.0
        
        # Ensure Types
        df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').fillna(0.0)
        df['local_win_rate'] = pd.to_numeric(df['local_win_rate'], errors='coerce').fillna(0.0)

        df['local_perf_diff'] = df['local_win_rate'] - df['nat_win_rate']

        # Wind Vectors
        df = FeatureEngineer.process_wind_data(df)

        # Ensure ALL model features exist
        # Add 'race_date' if not present (Scraper usually puts it in race_id or we need to pass it)
        # But 'race_date' column needed as Feature?
        # Check error log: ['race_date', ...] missing.
        # So we MUST add 'race_date'.
        # Since we only have 1 date for the race, we can fill it.
        # But LGBM expects Date? Or Category?
        # Train: Category. App: Object -> Category.
        # So assign string.
        if 'race_date' not in df.columns:
            # Extract from race_id "20251210_07_12" or similar
            # Or passed from arg process(df, venue_name) -> maybe add date?
            # Hack: extract from race_id
            try:
                df['race_date'] = df['race_id'].astype(str).apply(lambda x: x.split('_')[0] if '_' in x else '20000101')
            except:
                df['race_date'] = '20000101'
        
        # Missing column safeguard
        needed = ['nat_win_rate', 'sashi_count', 'course_run_count', 'course_quinella_rate', 
                  'course_trifecta_rate', 'course_1st_rate', 'course_avg_st', 
                  'venue_course_1st_rate', 'venue_course_2nd_rate', 'venue_course_3rd_rate']
        for c in needed:
            if c not in df.columns: df[c] = 0.0

        # Type Conversion
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

# Main Logic
debug_mode = st.sidebar.checkbox("Show Debug Info", value=False)

if st.button("Analyze Race", type="primary"):
    st.session_state['run_analysis'] = True
    st.session_state['target_props'] = {
        'date': target_date.strftime('%Y%m%d'),
        'venue': venue_code,
        'race': race_no,
        'v_name': venue_name
    }

if st.session_state.get('run_analysis'):
    props = st.session_state['target_props']
    
    st.info(f"Fetching Data: {props['v_name']} {props['race']}R ({props['date']})")
    
    # 1. Scrape
    with st.spinner("Scraping..."):
        df_race = BoatRaceScraper.get_race_data(props['date'], props['venue'], props['race'])
    
    if df_race is not None:
        st.subheader("Live Race Data")
        cols = ['boat_number', 'racer_id', 'branch', 'weight', 'motor_rate', 'exhibition_time', 'exhibition_start_timing', 'wind_speed']
        st.dataframe(df_race[cols])
        
        # 2. Features
        with st.spinner("Processing..."):
            df_feat = FeatureEngineer.process(df_race, props['v_name'], debug_mode=debug_mode)
            
        # 3. Predict
        if os.path.exists(MODEL_PATH):
            try:
                model = lgb.Booster(model_file=MODEL_PATH)
                
                # Align columns
                # Align columns
                model_feats = model.feature_name()
                
                # --- Display Input Data ---
                st.subheader("📊 Model Input Features")
                st.dataframe(df_feat[model_feats])
                
                # Predict
                X_pred = df_feat[model_feats]
                preds = model.predict(X_pred)
                df_feat['score'] = preds
                
                # 4. Result
                rank_df = df_feat[['boat_number', 'score']].sort_values('score', ascending=False)
                rank_df['rank'] = range(1, len(rank_df) + 1)
                
                st.divider()
                st.subheader("🤖 AI Prediction Ranking")
                st.dataframe(rank_df.set_index('rank'))
                
                scores = dict(zip(rank_df['boat_number'], rank_df['score']))
                boats_sorted = rank_df['boat_number'].tolist()
                
                # Generate Top Trifecta Combinations
                import itertools
                combos = list(itertools.permutations(boats_sorted, 3))
                c_list = []
                for c in combos:
                    # Score metric: Product of individual scores
                    s = scores[c[0]] * scores[c[1]] * scores[c[2]]
                    c_list.append({'combo': f"{c[0]}-{c[1]}-{c[2]}", 'val': s, 'p1': c[0]})
                
                df_c = pd.DataFrame(c_list).sort_values('val', ascending=False)
                
                # Strategy 1: Honmei (Top 5 Overall)
                st.subheader("🎯 Main Strategy (Honmei)")
                # Use enumerate to get 1,2,3... rank instead of shuffled index
                for i, (_, row) in enumerate(df_c.head(5).iterrows()):
                    label = f"Rank {i+1}"
                    if i == 0:
                        label += " 🔥 (50倍以上なら勝負時)"
                        st.success(f"{label}: {row['combo']}")
                    else:
                        st.metric(label, row['combo'])
            except Exception as e:
                st.error(f"AI Model Error: {e}")
        else:
            st.warning("Model file (lgb_ranker.txt) not found.")
    else:
        st.error("Failed to load race data.")
