# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 10.2 : Roulement par ligne strict + export HTML interactif avec gestion date/heure
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from collections import defaultdict
import re
import json
from datetime import datetime


# ========================
# CONFIG STREAMLIT
# ========================
st.set_page_config(page_title="Planning Juges - Crossfit Amiens", layout="wide")
st.title("🏋️‍♂️ Planning Juges - Crossfit Amiens 🦄")


# ========================
# FONCTION NETTOYAGE TEXTE ROBUSTE
# ========================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
        'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n', 'ß': 'ss',
        'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A',
        'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
        'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O',
        'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
        'Ç': 'C', 'Ñ': 'N',
        'œ': 'oe', 'æ': 'ae', '€': 'E', '£': 'GBP',
        '§': 'S', 'µ': 'u', '°': 'deg', '²': '2', '³': '3'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


# ========================
# LECTURE DU FICHIER EXCEL (feuille "Heats")
# ========================
def load_schedule_from_excel(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = None
        for name in xls.sheet_names:
            if name.lower() == "heats":
                sheet_name = name
                break
        
        if sheet_name is None:
            st.error("❌ Feuille 'Heats' introuvable dans le fichier Excel.")
            return None
        
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        required_cols = ['Lane', 'Competitor', 'Division', 'Workout', 'Workout Location', 'Heat #', 'Heat Start Time', 'Heat End Time']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Colonnes manquantes: {missing_cols}")
            return None
        
        for col in ['Workout', 'Competitor', 'Division', 'Workout Location', 'Heat #']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: clean_text(str(x)) if pd.notna(x) else "")
        
        df = df[df['Competitor'].notna() & (df['Competitor'] != "")]
        df = df[~df['Competitor'].str.contains('EMPTY LANE', na=False)]
        return df
    except Exception as e:
        st.error(f"Erreur Excel : {e}")
        return None


# ========================
# PDF AVEC LOGO EN BAS DE PAGE
# ========================
class FooterLogoPDF(FPDF):
    def __init__(self, logo_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo_path = logo_path
        self.set_auto_page_break(auto=True, margin=15)

    def footer(self):
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.set_y(-70)
                page_width = 210
                logo_width = 50
                x = (page_width - logo_width) / 2
                self.image(self.logo_path, x=x, y=self.get_y(), w=logo_width)
            except:
                pass


# ========================
# GENERATION PDFS
# ========================
def generate_pdf_tableau(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    pdf = FooterLogoPDF(logo_path=logo_path, orientation='P')
    for juge, creneaux in planning.items():
        if not creneaux:
            continue
        creneaux = sorted(creneaux, key=lambda c: str(c.get('start', '')))
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, clean_text(competition_name), 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning : {clean_text(juge)}", 0, 1, 'C')
        pdf.ln(8)

        headers = ["Heure", "Lane", "WOD", "Heat", "Athlete", "Division"]
        col_widths = [22, 14, 22, 18, 72, 32]
        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, 1, 0, 'C', fill=True)
        pdf.ln()
        
        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]
        for i, c in enumerate(creneaux):
            if pdf.get_y() > 250:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 10)
                pdf.set_fill_color(211, 211, 211)
                for h, w in zip(headers, col_widths):
                    pdf.cell(w, 8, h, 1, 0, 'C', fill=True)
                pdf.ln()
                pdf.set_font("Arial", '', 9)
            
            pdf.set_fill_color(*row_colors[i % 2])
            vals = [
                f"{clean_text(str(c['start']))[:5]}-{clean_text(str(c['end']))[:5]}",
                clean_text(str(c['lane'])),
                clean_text(str(c['wod']))[:12],
                clean_text(str(c['heat']))[:8],
                clean_text(str(c['athlete']))[:32],
                clean_text(str(c['division']))[:17]
            ]
            for v, w in zip(vals, col_widths):
                pdf.cell(w, 7, v, 1, 0, 'C', fill=True)
            pdf.ln()
        pdf.ln(5)
        pdf.cell(0, 8, f"Total : {len(creneaux)} creneaux ", 0, 1)
    return pdf


def generate_heat_pdf(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (c['wod'], c['heat'], c['start'], c['end'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FooterLogoPDF(logo_path=logo_path, orientation='P')
    heats = sorted(heat_map.items(), key=lambda x: (x[0][2], x[0][0], x[0][1]))

    col_width = 85
    row_height = 6
    spacing_x = 10
    header_height = 12
    
    for i in range(0, len(heats), 4):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, clean_text(competition_name), 0, 1, 'C')
        pdf.ln(4)

        current_y = 25
        page_heats = heats[i:i+4]
        
        for idx, ((wod, heat_num, start, end), lanes) in enumerate(page_heats):
            is_left = (idx % 2 == 0)
            x = 10 if is_left else 10 + col_width + spacing_x
            if idx == 2:
                max_lanes = max(len(heats[i][1]), len(heats[i+1][1])) if i+1 < len(heats) else len(heats[i][1])
                current_y += header_height + (max_lanes * row_height) + 4
            
            y_start = current_y
            pdf.set_font("Arial", 'B', 8)
            pdf.set_xy(x, y_start)
            pdf.cell(col_width, row_height, clean_text(f"{wod} | {heat_num} | {start}-{end}"), border=1, align='C', fill=True)
            
            pdf.set_font("Arial", 'B', 7)
            pdf.set_xy(x, y_start + row_height)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(col_width * 0.35, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width * 0.65, row_height, "Juge", border=1, align='C', fill=True)
            
            pdf.set_font("Arial", '', 7)
            pdf.set_fill_color(255, 255, 255)
            row_num = 0
            for lane_num, juge_name in sorted(lanes.items()):
                pdf.set_xy(x, y_start + row_height * 2 + row_num * row_height)
                pdf.cell(col_width * 0.35, row_height, clean_text(str(lane_num)), border=1, align='C')
                pdf.cell(col_width * 0.65, row_height, clean_text(juge_name)[:16], border=1, align='C')
                row_num += 1
    return pdf


def extract_heat_number(heat_str):
    if pd.isna(heat_str): return 0
    if isinstance(heat_str, (int, float)): return int(heat_str)
    numbers = re.findall(r'\d+', str(heat_str))
    return int(numbers[0]) if numbers else 0


# ========================================================
# EXTRACTION DATE + HEURE ROBUSTE (US / ISO / heure seule)
# ========================================================
def extract_datetime_parts(value, default_date_str):
    """
    Retourne (time_str 'HH:MM', date_str 'YYYY-MM-DD') depuis différentes formes :
    - datetime pandas -> direct
    - '10/17/2026 08:30:00' (format US) -> ('08:30', '2026-10-17')
    - '2026-10-17 08:30:00' (ISO) -> ('08:30', '2026-10-17')
    - '08:30:00' (heure seule) -> ('08:30', default_date_str)
    - '08:30' -> ('08:30', default_date_str)
    """
    if pd.isna(value):
        return "", default_date_str

    # Cas datetime / Timestamp pandas
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%H:%M"), value.strftime("%Y-%m-%d")

    s = str(value).strip()

    # --- Cas format US : MM/DD/YYYY HH:MM[:SS] ---
    m_us = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+(\d{1,2}):(\d{2}))?', s)
    if m_us:
        month, day, year, hh, mm = m_us.groups()
        date_str = f"{year}-{int(month):02d}-{int(day):02d}"
        if hh is not None and mm is not None:
            time_str = f"{int(hh):02d}:{mm}"
        else:
            time_str = ""
        return time_str, date_str

    # --- Cas format ISO : YYYY-MM-DD HH:MM[:SS] ---
    m_iso = re.search(r'(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?', s)
    if m_iso:
        year, month, day, hh, mm = m_iso.groups()
        date_str = f"{year}-{month}-{day}"
        if hh is not None and mm is not None:
            time_str = f"{int(hh):02d}:{mm}"
        else:
            time_str = ""
        return time_str, date_str

    # --- Cas heure seule : HH:MM[:SS] ou HHhMM ---
    m_time = re.search(r'(\d{1,2})[:hH](\d{2})', s)
    if m_time:
        return f"{int(m_time.group(1)):02d}:{m_time.group(2)}", default_date_str

    return "", default_date_str


# ========================================================
# NOUVEL ALGORITHME : SUIVI ET ROULEMENT STRICT PAR LIGNE
# ========================================================
def assign_judges_equitable(
    schedule,
    judges,
    disponibilites,
    rotation_system
):

    planning = {j: [] for j in judges}

    ON = rotation_system["on"]
    OFF = rotation_system["off"]

    judge_order = {j: i for i, j in enumerate(judges)}

    df = schedule.copy()
    df["heat_num"] = df["Heat #"].apply(extract_heat_number)
    df = df.sort_values(["Heat Start Time", "heat_num", "Lane"]).reset_index(drop=True)

    heats = []
    for (wod, heat_num, start, end), g in df.groupby(["Workout", "heat_num", "Heat Start Time", "Heat End Time"]):
        heats.append({
            "wod": wod,
            "heat_num": heat_num,
            "start": start,
            "end": end,
            "rows": sorted(g.to_dict("records"), key=lambda r: int(float(r["Lane"])))
        })

    # État des juges
    state = {}
    for j in judges:
        state[j] = {
            "phase": "AVAILABLE",
            "remaining": 0,
            "count": 0,
            "lane": None,
            "consecutive_heats": 0
        }

    lane_assignment = {}

    # ==================================================
    # Boucle heats
    # ==================================================
    for heat_idx, heat in enumerate(heats):
        wod = heat["wod"]
        dispo = disponibilites.get(wod, judges)
        
        lanes_presentes = sorted({str(int(float(r["Lane"]))) for r in heat["rows"]}, key=int)

        # Ensemble des juges déjà utilisés dans CE heat
        judges_used_this_heat = set()

        # Libération des lanes des juges OFF
        lanes_to_remove = []
        for lane, judge in lane_assignment.items():
            if state[judge]["phase"] != "ON":
                lanes_to_remove.append(lane)
        for lane in lanes_to_remove:
            lane_assignment.pop(lane, None)

        # ==========================================
        # Remplissage des lanes vides
        # ==========================================
        for lane in lanes_presentes:
            if lane in lane_assignment:
                judge = lane_assignment[lane]
                if judge in judges_used_this_heat:
                    lane_assignment.pop(lane, None)
                else:
                    judges_used_this_heat.add(judge)
                    continue

            # Niveau 1 : Juges AVAILABLE et pas déjà utilisés dans ce heat
            candidates = [
                j for j in judges 
                if j in dispo 
                and state[j]["phase"] == "AVAILABLE"
                and j not in judges_used_this_heat
            ]
            
            # Niveau 2 : Juges qui finissent leur OFF dans 1 heat
            if not candidates:
                candidates = [
                    j for j in judges 
                    if j in dispo 
                    and state[j]["phase"] == "OFF" 
                    and state[j]["remaining"] <= 1
                    and j not in judges_used_this_heat
                ]
            
            # Niveau 3 : Juges en OFF mais en dessous de la moyenne
            if not candidates:
                avg_heats = sum(state[j]["count"] for j in judges) / len(judges) if judges else 0
                candidates = [
                    j for j in judges 
                    if j in dispo 
                    and state[j]["phase"] == "OFF" 
                    and state[j]["count"] <= avg_heats + 1
                    and j not in judges_used_this_heat
                ]
            
            # Niveau 4 : DERNIER RECOURS - n'importe quel juge dispo
            if not candidates:
                candidates = [
                    j for j in judges 
                    if j in dispo 
                    and j not in judges_used_this_heat
                ]
                if not candidates:
                    candidates = [j for j in judges if j not in judges_used_this_heat]
            
            candidates = sorted(candidates, key=lambda x: (state[x]["count"], judge_order[x]))
            selected = candidates[0]

            lane_assignment[lane] = selected
            judges_used_this_heat.add(selected)
            state[selected]["phase"] = "ON"
            state[selected]["remaining"] = ON
            state[selected]["lane"] = lane
            state[selected]["consecutive_heats"] = 0

        # ==========================================
        # Attribution du heat
        # ==========================================
        worked_this_heat = set()
        
        for row in heat["rows"]:
            lane = str(int(float(row["Lane"])))
            judge = lane_assignment.get(lane)
            
            if judge is None:
                available_judges = [j for j in judges if j in dispo and j not in judges_used_this_heat]
                if not available_judges:
                    available_judges = [j for j in judges if j not in judges_used_this_heat]
                judge = min(available_judges, key=lambda x: state[x]["count"])
                lane_assignment[lane] = judge
                judges_used_this_heat.add(judge)
                state[judge]["phase"] = "ON"
                state[judge]["remaining"] = ON
                state[judge]["lane"] = lane

            planning[judge].append({
                "wod": clean_text(str(row["Workout"])),
                "lane": lane,
                "athlete": clean_text(str(row["Competitor"])),
                "division": clean_text(str(row["Division"])),
                "start": clean_text(str(row["Heat Start Time"])),
                "end": clean_text(str(row["Heat End Time"])),
                "heat": clean_text(str(row["Heat #"])),
                "heat_num": heat["heat_num"]
            })

            worked_this_heat.add(judge)
            state[judge]["count"] += 1
            state[judge]["consecutive_heats"] += 1

        # ==========================================
        # Mise à jour ON / OFF
        # ==========================================
        for judge in judges:
            if judge in worked_this_heat:
                if state[judge]["phase"] == "ON":
                    state[judge]["remaining"] -= 1
                    
                    if state[judge]["remaining"] <= 0 or state[judge]["consecutive_heats"] >= ON:
                        state[judge]["phase"] = "OFF"
                        state[judge]["remaining"] = OFF
                        state[judge]["lane"] = None
                        state[judge]["consecutive_heats"] = 0
            else:
                if state[judge]["phase"] == "OFF":
                    state[judge]["remaining"] -= 1
                    if state[judge]["remaining"] <= 0:
                        state[judge]["phase"] = "AVAILABLE"
                        state[judge]["remaining"] = 0

    return planning

# ========================
# GENERATION HTML INTERACTIF
# ========================
def generate_html_planning(planning: dict, competition_name: str, judges: list, competition_date=None) -> str:
    """
    Génère un HTML autonome avec sélection du juge et mise en évidence
    du prochain heat à juger (fond bleu) et des suivants (encadré bleu),
    les heats passés étant plus transparents.

    La comparaison "passé / prochain / à venir" se fait sur le DATETIME complet
    (date + heure du heat), donc fonctionne même si la compétition est sur
    plusieurs jours.

    Compatible Safari iOS en file://, mail, Fichiers, etc.
    Les données sont injectées en JS via json.dumps (aucune balise script JSON).
    """
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if competition_date is None:
        default_date_str = datetime.now().strftime("%Y-%m-%d")
    elif isinstance(competition_date, str):
        default_date_str = competition_date
    else:
        default_date_str = competition_date.strftime("%Y-%m-%d")

    data = {}
    for juge, creneaux in planning.items():
        if not creneaux:
            data[juge] = []
            continue

        enriched = []
        for c in creneaux:
            start_raw = c.get('start', '')
            end_raw = c.get('end', '')

            start_time, start_date = extract_datetime_parts(start_raw, default_date_str)
            end_time, end_date = extract_datetime_parts(end_raw, default_date_str)

            dt_start = f"{start_date}T{start_time}" if start_time else ""
            dt_end = f"{end_date}T{end_time}" if end_time else ""

            enriched.append({
                "start": start_time or clean_text(str(start_raw)),
                "end": end_time or clean_text(str(end_raw)),
                "lane": clean_text(str(c.get('lane', ''))),
                "wod": clean_text(str(c.get('wod', ''))),
                "heat": clean_text(str(c.get('heat', ''))),
                "athlete": clean_text(str(c.get('athlete', ''))),
                "division": clean_text(str(c.get('division', ''))),
                "dt_start": dt_start,
                "dt_end": dt_end,
                "sort_key": dt_start or f"{default_date_str}T{start_time}",
            })

        enriched.sort(key=lambda x: x["sort_key"])
        data[juge] = enriched

    judges_sorted = sorted(judges)

    # --- Sérialisation JS robuste ---
    # json.dumps produit une string JSON valide.
    # On l'échappe pour qu'elle soit utilisable comme littéral JS string
    # (double-encode), puis on la parse côté JS avec JSON.parse.
    # Cela évite TOUT problème avec les <script type="application/json">.
    def to_js_string(obj):
        # 1) JSON standard
        json_str = json.dumps(obj, ensure_ascii=False)
        # 2) Encodage en littéral JS via json.dumps (ajoute les guillemets et échappe)
        return json.dumps(json_str, ensure_ascii=False)

    data_js = to_js_string(data)          # "{\"juge1\": [...]}"  (string JS)
    judges_js = to_js_string(judges_sorted)
    now_js = to_js_string(now_iso)

    comp_name = clean_text(competition_name)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Planning Juges - {comp_name}</title>
<style>
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    background: #f5f7fa;
    margin: 0;
    padding: 20px 14px;
    color: #222;
  }}
  h1 {{
    text-align: center;
    margin: 0 0 6px 0;
    font-size: 24px;
  }}
  .subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 20px;
    font-size: 14px;
  }}
  .controls {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }}
  .controls label {{ font-weight: 600; }}
  select {{
    padding: 12px 14px;
    font-size: 16px;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    background: #fff;
    min-width: 240px;
    max-width: 100%;
    cursor: pointer;
    -webkit-appearance: menulist;
    appearance: menulist;
  }}
  .summary {{
    text-align: center;
    margin-bottom: 16px;
    font-size: 14px;
    color: #475569;
  }}
  .summary strong {{ color: #1e3a8a; }}
  .heat-list {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 14px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .heat-card {{
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    transition: all .2s ease;
    opacity: 0.45;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .heat-card.next {{
    background: #2563eb;
    border-color: #1d4ed8;
    color: #ffffff;
    opacity: 1;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35);
  }}
  .heat-card.next .heat-time {{ color: #dbeafe; }}
  .heat-card.next .heat-meta {{ color: #bfdbfe; }}
  .heat-card.next .heat-athlete {{ color: #fff; }}
  .heat-card.next .heat-date {{ color: #dbeafe; }}
  .heat-card.next .heat-division {{ color: #dbeafe; }}
  .heat-card.upcoming {{
    background: #eff6ff;
    border-color: #3b82f6;
    border-width: 2px;
    opacity: 1;
  }}
  .heat-card.upcoming .heat-time {{ color: #1e40af; }}
  .heat-card.past {{
    opacity: 0.45;
  }}
  .heat-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .heat-time {{
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    white-space: nowrap;
  }}
  .heat-date {{
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
    margin-bottom: 6px;
  }}
  .heat-badge {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .5px;
    padding: 3px 8px;
    border-radius: 999px;
    background: #e0e7ff;
    color: #3730a3;
  }}
  .heat-card.next .heat-badge {{
    background: #ffffff;
    color: #1d4ed8;
  }}
  .heat-card.upcoming .heat-badge {{
    background: #dbeafe;
    color: #1e40af;
  }}
  .heat-meta {{
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
  }}
  .heat-athlete {{
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 2px;
    color: #0f172a;
  }}
  .heat-division {{
    font-size: 12px;
    color: #64748b;
  }}
  .empty {{
    text-align: center;
    color: #94a3b8;
    padding: 40px;
    font-size: 16px;
  }}
  .now-info {{
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-bottom: 14px;
  }}
  .debug {{
    text-align: center;
    color: #c00;
    font-size: 12px;
    margin: 10px 0;
    padding: 8px;
    background: #fee;
    border: 1px solid #fbb;
    border-radius: 6px;
  }}
  @media (max-width: 600px) {{
    select {{ min-width: 0; width: 100%; }}
    .controls {{ flex-direction: column; align-items: stretch; }}
    .controls label {{ text-align: center; }}
  }}
</style>
</head>
<body>
  <h1>🏋️‍♂️ {comp_name}</h1>
  <div class="subtitle">Planning des juges — sélectionnez un juge pour voir ses heats</div>

  <div class="controls">
    <label for="judge-select">👨‍⚖️ Juge :</label>
    <select id="judge-select"></select>
  </div>

  <div id="debug" class="debug" style="display:none;"></div>

  <div class="now-info" id="now-info"></div>
  <div class="summary" id="summary"></div>

  <div class="heat-list" id="heat-list"></div>

<script>
(function () {{
  "use strict";

  // ---- Données injectées en littéraux JS string (universel, y compris file://) ----
  var DATA_RAW = {data_js};
  var JUDGES_RAW = {judges_js};
  var NOW_RAW = {now_js};

  var DATA = {{}};
  var JUDGES = [];
  var NOW_ISO = "";
  var errors = [];

  try {{ DATA = JSON.parse(DATA_RAW); }}
  catch (e) {{ errors.push('DATA: ' + e.message); }}

  try {{ JUDGES = JSON.parse(JUDGES_RAW); }}
  catch (e) {{ errors.push('JUDGES: ' + e.message); }}

  try {{ NOW_ISO = JSON.parse(NOW_RAW); }}
  catch (e) {{ errors.push('NOW: ' + e.message); }}

  function showError(msg) {{
    var d = document.getElementById('debug');
    if (d) {{
      d.style.display = 'block';
      d.textContent = msg;
    }}
  }}

  if (errors.length > 0) {{
    showError('⚠️ Erreur de chargement : ' + errors.join(' | '));
  }}

  function parseLocalDateTime(iso) {{
    if (!iso) return null;
    var m = String(iso).match(/^(\\d{{4}})-(\\d{{2}})-(\\d{{2}})T(\\d{{2}}):(\\d{{2}})/);
    if (!m) return null;
    return new Date(
      parseInt(m[1], 10),
      parseInt(m[2], 10) - 1,
      parseInt(m[3], 10),
      parseInt(m[4], 10),
      parseInt(m[5], 10),
      0
    );
  }}

  function nowDate() {{
    return new Date();
  }}

  function fmtNow() {{
    var d = new Date();
    var hh = ('0' + d.getHours()).slice(-2);
    var mm = ('0' + d.getMinutes()).slice(-2);
    var dd = ('0' + d.getDate()).slice(-2);
    var mo = ('0' + (d.getMonth() + 1)).slice(-2);
    var yy = d.getFullYear();
    return dd + '/' + mo + '/' + yy + ' ' + hh + ':' + mm;
  }}

  function formatDateFr(isoDate) {{
    if (!isoDate) return "";
    var m = String(isoDate).match(/^(\\d{{4}})-(\\d{{2}})-(\\d{{2}})/);
    if (!m) return isoDate;
    return m[3] + '/' + m[2] + '/' + m[1];
  }}

  function escapeHtml(s) {{
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }}

  function render(judgeName) {{
    var list = document.getElementById('heat-list');
    var summary = document.getElementById('summary');
    var nowInfo = document.getElementById('now-info');
    list.innerHTML = '';

    var heats = DATA[judgeName] || [];
    if (heats.length === 0) {{
      summary.innerHTML = '';
      nowInfo.textContent = '';
      list.innerHTML = '<div class="empty">Aucun heat pour ce juge.</div>';
      return;
    }}

    var nowMs = nowDate().getTime();

    var nextIndex = -1;
    for (var i = 0; i < heats.length; i++) {{
      var s = parseLocalDateTime(heats[i].dt_start);
      var e = parseLocalDateTime(heats[i].dt_end);
      if (s && e && nowMs >= s.getTime() && nowMs <= e.getTime()) {{
        nextIndex = i;
        break;
      }}
    }}
    if (nextIndex === -1) {{
      for (var j = 0; j < heats.length; j++) {{
        var s2 = parseLocalDateTime(heats[j].dt_start);
        if (s2 && s2.getTime() > nowMs) {{
          nextIndex = j;
          break;
        }}
      }}
    }}

    var pastCount = (nextIndex === -1) ? heats.length : nextIndex;

    nowInfo.textContent = "🕐 Heure actuelle : " + fmtNow();

    if (nextIndex === -1) {{
      summary.innerHTML = "Tous les heats sont terminés — <strong>" + heats.length + "</strong> heats au total";
    }} else {{
      var remaining = heats.length - nextIndex;
      summary.innerHTML = "✅ <strong>" + pastCount + "</strong> terminé(s) · 🎯 <strong>" + remaining + "</strong> restant(s) · " + heats.length + " au total";
    }}

    heats.forEach(function (h, idx) {{
      var card = document.createElement('div');
      card.className = 'heat-card';

      var badgeHtml = '';
      if (idx === nextIndex) {{
        card.classList.add('next');
        badgeHtml = '<span class="heat-badge">🎯 Prochain</span>';
      }} else if (nextIndex !== -1 && idx > nextIndex) {{
        card.classList.add('upcoming');
        badgeHtml = '<span class="heat-badge">À venir</span>';
      }} else {{
        card.classList.add('past');
        badgeHtml = '<span class="heat-badge" style="background:#f1f5f9;color:#94a3b8;">Passé</span>';
      }}

      var datePart = '';
      var mDate = String(h.dt_start || '').match(/^(\\d{{4}})-(\\d{{2}})-(\\d{{2}})/);
      if (mDate) {{
        datePart = '<div class="heat-date">📅 ' + formatDateFr(mDate[0]) + '</div>';
      }}

      card.innerHTML =
        '<div class="heat-header">' +
          '<span class="heat-time">' + escapeHtml(h.start) + ' – ' + escapeHtml(h.end) + '</span>' +
          badgeHtml +
        '</div>' +
        datePart +
        '<div class="heat-meta">' + escapeHtml(h.wod) + ' · ' + escapeHtml(h.heat) + ' · Lane ' + escapeHtml(h.lane) + '</div>' +
        '<div class="heat-athlete">' + escapeHtml(h.athlete) + '</div>' +
        '<div class="heat-division">' + escapeHtml(h.division) + '</div>';

      list.appendChild(card);
    }});
  }}

  function populateSelect() {{
    var select = document.getElementById('judge-select');
    if (!select) {{
      showError('⚠️ Élément select introuvable');
      return;
    }}
    // Vider (méthode Safari-safe)
    while (select.firstChild) {{
      select.removeChild(select.firstChild);
    }}

    if (!JUDGES || JUDGES.length === 0) {{
      var optEmpty = document.createElement('option');
      optEmpty.value = '';
      optEmpty.textContent = '(aucun juge)';
      select.appendChild(optEmpty);
      return;
    }}

    JUDGES.forEach(function (j) {{
      var opt = document.createElement('option');
      opt.value = j;
      opt.textContent = j;
      select.appendChild(opt);
    }});
  }}

  function init() {{
    populateSelect();
    var select = document.getElementById('judge-select');

    if (!select) {{
      showError('⚠️ Impossible d’initialiser la liste des juges');
      return;
    }}

    select.addEventListener('change', function () {{
      render(select.value);
    }});

    if (JUDGES.length > 0) {{
      select.value = JUDGES[0];
      render(JUDGES[0]);
    }}

    setInterval(function () {{
      if (select.value) render(select.value);
    }}, 30000);
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', init);
  }} else {{
    init();
  }}
}})();
</script>
</body>
</html>"""
    return html


# ========================
# MAIN STREAMLIT
# ========================
def main():

    with st.sidebar:

        st.header("📂 Fichier d'entrée")

        schedule_file = st.file_uploader(
            "Planning (Excel)",
            type=["xlsx"]
        )

        competition_name = st.text_input(
            "Nom de la compétition",
            "Unicorn Throwdown"
        )

        competition_date = st.date_input(
            "Date de la compétition (fallback si absente du fichier)",
            value=datetime.now().date()
        )

        st.header("👨‍⚖️ Juges")

        judges_file = st.file_uploader(
            "Liste des juges (CSV)",
            type=["csv"]
        )

        if judges_file:

            try:

                judges_df = pd.read_csv(
                    judges_file,
                    header=None,
                    encoding="latin1"
                )

                judges = [
                    clean_text(str(j))
                    for j in judges_df[0].dropna().tolist()
                ]

            except Exception as e:

                st.error(f"Erreur lecture CSV : {e}")
                judges = []

        else:

            judges_text = st.text_area(
                "Saisir les juges (un par ligne)",
                "Juge 1\nJuge 2\nJuge 3"
            )

            judges = [
                clean_text(j.strip())
                for j in judges_text.split("\n")
                if j.strip()
            ]

        st.header("🖼️ Logo")

        logo_file = st.file_uploader(
            "Uploader un logo",
            type=["png", "jpg", "jpeg"]
        )

        logo_path = None

        if logo_file:

            temp_logo = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png"
            )

            temp_logo.write(logo_file.read())
            temp_logo.close()

            logo_path = temp_logo.name

    # ==================================================
    # Lecture planning
    # ==================================================

    if not schedule_file or not judges:

        st.info(
            "👉 Veuillez importer le fichier Excel et renseigner les juges."
        )
        return

    schedule = load_schedule_from_excel(schedule_file)

    if schedule is None:
        return

    if schedule.empty:
        st.error("❌ Aucun heat trouvé")
        return

    # ==================================================
    # Liste WODs
    # ==================================================

    wods = sorted(
        schedule["Workout"]
        .dropna()
        .astype(str)
        .unique()
    )

    # ==================================================
    # Roulement par WOD
    # ==================================================

    st.header("⚙️ Roulement par WOD")

    rotation_system = st.selectbox(
        "Roulement",
        [
            {"name": "1-on / 1-off", "on": 1, "off": 1},
            {"name": "2-on / 1-off", "on": 2, "off": 1},
            {"name": "2-on / 2-off", "on": 2, "off": 2},
            {"name": "3-on / 2-off", "on": 3, "off": 2},
            {"name": "3-on / 3-off", "on": 3, "off": 3},
            {"name": "4-on / 2-off", "on": 4, "off": 2},
        ],
        index=4,
        format_func=lambda x: x["name"]
    )

    # ==================================================
    # Disponibilités
    # ==================================================

    st.header("📅 Disponibilités des juges")

    disponibilites = {}

    cols = st.columns(3)

    for i, wod in enumerate(wods):

        with cols[i % 3]:

            with st.expander(wod):

                all_selected = st.checkbox(
                    f"Tous disponibles ({wod})",
                    value=True,
                    key=f"all_{wod}"
                )

                if all_selected:

                    disponibilites[wod] = judges

                else:

                    disponibilites[wod] = st.multiselect(
                        "Juges disponibles",
                        judges,
                        key=f"multi_{wod}"
                    )

    # ==================================================
    # Génération
    # ==================================================

    if st.button("🦄 Générer le planning"):

        try:

            planning = assign_judges_equitable(
                schedule,
                judges,
                disponibilites,
                rotation_system
            )

            st.success("✅ Planning généré")

            # ======================================
            # Stats
            # ======================================

            st.header("📊 Répartition")

            counts = {
                j: len(planning[j])
                for j in judges
            }

            total_assignments = sum(counts.values())

            moyenne = (
                round(
                    total_assignments / len(judges),
                    1
                )
                if judges else 0
            )

            st.info(
                f"Total heats arbitrés : {total_assignments} | "
                f"Moyenne par juge : {moyenne}"
            )

            stats_df = pd.DataFrame({
                "Juge": list(counts.keys()),
                "Heats": list(counts.values())
            })

            stats_df = stats_df.sort_values(
                "Heats",
                ascending=False
            )

            st.dataframe(
                stats_df,
                use_container_width=True
            )

            # ======================================
            # PDF JUGES
            # ======================================

            pdf_juges = generate_pdf_tableau(
                planning,
                competition_name,
                logo_path
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                pdf_juges.output(tmp.name)

                with open(tmp.name, "rb") as f:

                    st.download_button(
                        "📘 Télécharger planning par juge",
                        f.read(),
                        "planning_juges.pdf",
                        mime="application/pdf"
                    )

            # ======================================
            # PDF HEATS
            # ======================================

            pdf_heats = generate_heat_pdf(
                planning,
                competition_name,
                logo_path
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                pdf_heats.output(tmp.name)

                with open(tmp.name, "rb") as f:

                    st.download_button(
                        "📗 Télécharger planning par heat",
                        f.read(),
                        "planning_heats.pdf",
                        mime="application/pdf"
                    )

            # ======================================
            # HTML INTERACTIF
            # ======================================

            html_content = generate_html_planning(
                planning,
                competition_name,
                judges,
                competition_date
            )

            st.download_button(
                "🌐 Télécharger planning interactif (HTML)",
                html_content.encode("utf-8"),
                "planning_interactif.html",
                mime="text/html"
            )

        except Exception as e:

            st.error(
                f"❌ Erreur génération planning : {str(e)}"
            )


if __name__ == "__main__":
    main()
