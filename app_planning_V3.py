# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 10.0 : Stabilité des lignes (Même Lane pendant le bloc ON)
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from collections import defaultdict
import re

# ========================
# CONFIG STREAMLIT
# ========================
st.set_page_config(page_title="Planning Juges - Crossfit Amiens", layout="wide")
st.title("🏋️‍♂️ Planning Juges - Crossfit Amiens 🦄")

# ========================
# FONCTION NETTOYAGE TEXTE
# ========================
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    replacements = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
        'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n', 'ß': 'ss', 'À': 'A', 'É': 'E', 'È': 'E'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode('ascii', 'ignore').decode('ascii')

# ========================
# LECTURE EXCEL
# ========================
def load_schedule_from_excel(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = next((n for n in xls.sheet_names if n.lower() == "heats"), None)
    if not sheet_name:
        st.error("❌ Feuille 'Heats' introuvable.")
        return None
    
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    df = df.dropna(subset=['Competitor'])
    df = df[df['Competitor'] != ""]
    df = df[~df['Competitor'].str.contains('EMPTY LANE', na=False)]
    return df

# ========================
# PDF CLASSES & GENERATORS
# ========================
class FooterLogoPDF(FPDF):
    def __init__(self, logo_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo_path = logo_path
    def footer(self):
        if self.logo_path and os.path.exists(self.logo_path):
            self.set_y(-25)
            self.image(self.logo_path, x=80, y=self.get_y(), w=50)

def generate_pdf_tableau(planning, competition_name, logo_path=None):
    pdf = FooterLogoPDF(logo_path=logo_path)
    for juge, creneaux in planning.items():
        if not creneaux: continue
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"{competition_name} - Juge: {juge}", 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 10)
        cols = [25, 15, 25, 20, 60, 40]
        headers = ["Heure", "Lane", "WOD", "Heat", "Athlete", "Division"]
        for h, w in zip(headers, cols): pdf.cell(w, 8, h, 1, 0, 'C')
        pdf.ln()
        pdf.set_font("Arial", '', 9)
        for c in sorted(creneaux, key=lambda x: x['start']):
            pdf.cell(cols[0], 7, f"{c['start']}-{c['end']}", 1, 0, 'C')
            pdf.cell(cols[1], 7, str(c['lane']), 1, 0, 'C')
            pdf.cell(cols[2], 7, str(c['wod'])[:12], 1, 0, 'C')
            pdf.cell(cols[3], 7, str(c['heat']), 1, 0, 'C')
            pdf.cell(cols[4], 7, str(c['athlete'])[:30], 1, 0, 'C')
            pdf.cell(cols[5], 7, str(c['division'])[:15], 1, 0, 'C')
            pdf.ln()
    return pdf

def generate_heat_pdf(planning, competition_name, logo_path=None):
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (c['wod'], c['heat'], c['start'], c['end'])
            heat_map[key][int(float(c['lane']))] = juge
    pdf = FooterLogoPDF(logo_path=logo_path)
    for (wod, heat, start, end), lanes in sorted(heat_map.items(), key=lambda x: (x[0][2], x[0][1])):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"{wod} - Heat {heat} ({start})", 0, 1, 'C')
        pdf.ln(5)
        for l_num in sorted(lanes.keys()):
            pdf.cell(40, 10, f"Lane {l_num}:", 1)
            pdf.cell(100, 10, f" {lanes[l_num]}", 1, 1)
    return pdf

def extract_heat_number(heat_str):
    numbers = re.findall(r'\d+', str(heat_str))
    return int(numbers[0]) if numbers else 0

# ====================================================
# ATTRIBUTION AVEC STABILITÉ DE LA LIGNE (LANE)
# ====================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation_config):
    planning = {j: [] for j in judges}
    df = schedule.copy()
    df["heat_num"] = df["Heat #"].apply(extract_heat_number)
    df = df.sort_values(["Heat Start Time", "heat_num", "Lane"])

    heats = []
    for (wod, h_n, start, end), g in df.groupby(["Workout", "heat_num", "Heat Start Time", "Heat End Time"], sort=False):
        heats.append({
            "wod": wod, "heat_num": h_n, "start": start, "end": end,
            "rows": g.to_dict("records")
        })

    state = {
        j: {
            "phase": "AVAILABLE", 
            "remaining": 0, 
            "consecutive": 0, 
            "count": 0,
            "last_lane": None  # <--- MÉMOIRE DE LA LIGNE
        } for j in judges
    }

    for heat in heats:
        wod = heat["wod"]
        rot = rotation_config.get(wod, {"on": 3, "off": 3})
        dispo = disponibilites.get(wod, judges)
        
        required_lanes = sorted([str(int(float(r["Lane"]))) for r in heat["rows"]], key=int)
        assigned_this_heat = {} # Format: { "Lane": "Juge" }
        
        # 1. PRIORITÉ : Juges déjà ON qui gardent leur ligne
        already_on = [j for j in judges if state[j]["phase"] == "ON" and state[j]["remaining"] > 0 and j in dispo]
        
        # On essaie de remettre chaque juge ON sur sa dernière lane connue
        for j in already_on:
            target_lane = state[j]["last_lane"]
            if target_lane in required_lanes and target_lane not in assigned_this_heat:
                assigned_this_heat[target_lane] = j

        # Si un juge ON a vu sa lane disparaître (ex: moins de compétiteurs), on lui donne une lane libre
        remaining_lanes = [l for l in required_lanes if l not in assigned_this_heat]
        for j in already_on:
            if j not in assigned_this_heat.values() and remaining_lanes:
                lane = remaining_lanes.pop(0)
                assigned_this_heat[lane] = j

        # 2. COMPLÉTER : Avec les AVAILABLE
        if remaining_lanes:
            avail = [j for j in judges if state[j]["phase"] == "AVAILABLE" and j in dispo]
            avail = sorted(avail, key=lambda x: state[x]["count"])
            for j in avail:
                if remaining_lanes:
                    lane = remaining_lanes.pop(0)
                    assigned_this_heat[lane] = j
                    state[j]["phase"] = "ON"
                    state[j]["remaining"] = rot["on"]

        # 3. FALLBACK : Si toujours vide
        if remaining_lanes:
            backup = [j for j in judges if j in dispo and j not in assigned_this_heat.values() and state[j]["consecutive"] < (rot["on"] + 1)]
            backup = sorted(backup, key=lambda x: state[x]["count"])
            for j in backup:
                if remaining_lanes:
                    lane = remaining_lanes.pop(0)
                    assigned_this_heat[lane] = j
                    if state[j]["phase"] == "OFF": state[j]["remaining"] = 1

        # Enregistrement et mise à jour des états
        working_now = assigned_this_heat.values()
        for lane_id, j_name in assigned_this_heat.items():
            row_data = next(r for r in heat["rows"] if str(int(float(r["Lane"]))) == lane_id)
            planning[j_name].append({
                "wod": wod, "lane": lane_id, "athlete": row_data["Competitor"],
                "division": row_data["Division"], "start": heat["start"], "end": heat["end"],
                "heat": row_data["Heat #"]
            })
            state[j_name]["count"] += 1
            state[j_name]["consecutive"] += 1
            state[j_name]["last_lane"] = lane_id # On mémorise la lane
            if state[j_name]["phase"] == "ON":
                state[j_name]["remaining"] -= 1
                if state[j_name]["remaining"] <= 0:
                    state[j_name]["phase"] = "OFF"
                    state[j_name]["remaining"] = rot["off"]

        for j in judges:
            if j not in working_now:
                state[j]["consecutive"] = 0
                state[j]["last_lane"] = None # Libère la lane quand il part en pause
                if state[j]["phase"] == "OFF":
                    state[j]["remaining"] -= 1
                    if state[j]["remaining"] <= 0: state[j]["phase"] = "AVAILABLE"
                elif state[j]["phase"] == "ON":
                    state[j]["phase"] = "AVAILABLE"

    return planning

# ========================
# MAIN
# ========================
def main():
    with st.sidebar:
        st.header("📂 Fichiers")
        schedule_file = st.file_uploader("Planning Excel", type=["xlsx"])
        judges_text = st.text_area("Juges (un par ligne)", "Juge 1\nJuge 2")
        judges = [clean_text(j.strip()) for j in judges_text.split('\n') if j.strip()]
        competition_name = st.text_input("Nom Compétition", "Unicorn")
        logo_file = st.file_uploader("Logo", type=["png", "jpg"])
        logo_path = None
        if logo_file:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            t.write(logo_file.read()); t.close()
            logo_path = t.name

    if schedule_file and judges:
        schedule = load_schedule_from_excel(schedule_file)
        if schedule is not None:
            wods = sorted(schedule['Workout'].unique())
            rotations = {}
            for w in wods:
                rotations[w] = st.selectbox(f"Rotation {w}", [{"on":2,"off":2},{"on":3,"off":3},{"on":1,"off":1}], format_func=lambda x: f"{x['on']}on/{x['off']}off")
            
            dispos = {w: st.multiselect(f"Dispos {w}", judges, default=judges) for w in wods}

            if st.button("🚀 Générer"):
                results = assign_judges_equitable(schedule, judges, dispos, rotations)
                
                pdf_j = generate_pdf_tableau(results, competition_name, logo_path)
                pdf_h = generate_heat_pdf(results, competition_name, logo_path)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f1:
                    pdf_j.output(f1.name)
                    st.download_button("📥 Planning Juges", open(f1.name, "rb"), "juges.pdf")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f2:
                    pdf_h.output(f2.name)
                    st.download_button("📥 Planning Heats", open(f2.name, "rb"), "heats.pdf")

if __name__ == "__main__":
    main()
