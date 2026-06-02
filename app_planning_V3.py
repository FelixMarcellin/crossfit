# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 10.0 : Roulement par ligne strict (3-on/3-off respecté à +/- 1)
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
# NOUVEL ALGORITHME : SUIVI ET ROULEMENT STRICT PAR LIGNE
# ========================================================
def assignjudgesequitableschedule(schedule, judges, disponibilites, rotationconfig):
    planning = {j: [] for j in judges}
    warningslist = []

    ONTARGET = rotationconfig["on"]
    OFFTARGET = rotationconfig["off"]
    MAXALLOWED = ONTARGET + 1

    df = schedule.copy()
    df["heatnum"] = df["Heat"].apply(extract_heat_number)
    df = df.sort_values(["Heat Start Time", "heatnum", "Lane"]).reset_index(drop=True)

    heats = []
    for (wod, heatnum, start, end), g in df.groupby(["Workout", "heatnum", "Heat Start Time", "Heat End Time"]):
        heats.append({
            "wod": wod,
            "heatnum": heatnum,
            "start": start,
            "end": end,
            "rows": sorted(g.to_dict("records"), key=lambda r: int(float(r["Lane"])))
        })

    state = {
        j: {
            "status": "AVAILABLE",          # AVAILABLE, ON, OFF
            "heatssanspause": 0,            # heats travaillés consécutivement
            "consecutiveoff": 0,            # heats de repos consécutifs
            "totalcount": 0,                # équilibrage global
            "currentlane": None,            # ligne gardée pendant le bloc
        }
        for j in judges
    }

    def is_available_for_work(j):
        s = state[j]
        return s["status"] in ("AVAILABLE", "ON") and s["heatssanspause"] < ONTARGET

    def can_keep_lane(j, lane):
        s = state[j]
        return s["currentlane"] == lane and is_available_for_work(j)

    for heat in heats:
        wod = heat["wod"]
        heatlabel = f"Heat {heat['heatnum']} ({heat['start']}-{heat['end']})"
        dispo = disponibilites.get(wod, judges)
        if not dispo:
            dispo = judges

        judgesworkingthisheat = set()
        laneassignments = {}

        # 1) Mise à jour des statuts avant affectation
        for j in judges:
            s = state[j]

            if s["status"] == "OFF":
                s["consecutiveoff"] += 1
                if s["consecutiveoff"] >= OFFTARGET:
                    s["status"] = "AVAILABLE"
                    s["consecutiveoff"] = 0
                    s["heatssanspause"] = 0
                    s["currentlane"] = None

        # 2) Affectation ligne par ligne
        for row in heat["rows"]:
            lane = str(int(float(row["Lane"])))
            chosenjudge = None

            # Priorité 1 : garder le juge déjà sur cette ligne si possible
            prevjudge = None
            for j in judges:
                if state[j]["currentlane"] == lane and j in dispo:
                    prevjudge = j
                    break

            if prevjudge and prevjudge not in judgesworkingthisheat and is_available_for_work(prevjudge):
                chosenjudge = prevjudge

            # Priorité 2 : choisir un juge qui est déjà dans ce bloc de travail
            if chosenjudge is None:
                candidates = []
                for j in judges:
                    s = state[j]
                    if j in dispo and j not in judgesworkingthisheat and is_available_for_work(j):
                        candidates.append(j)

                candidates = sorted(
                    candidates,
                    key=lambda j: (
                        0 if state[j]["status"] == "AVAILABLE" else 1,
                        state[j]["heatssanspause"],
                        state[j]["totalcount"]
                    )
                )

                if candidates:
                    chosenjudge = candidates[0]

            # Priorité 3 : dernier recours, on autorise un juge encore bloqué seulement si impossible
            if chosenjudge is None:
                fallback = []
                for j in judges:
                    s = state[j]
                    if j in dispo and j not in judgesworkingthisheat:
                        fallback.append(j)

                fallback = sorted(
                    fallback,
                    key=lambda j: (
                        0 if state[j]["status"] == "AVAILABLE" else 1,
                        state[j]["heatssanspause"],
                        state[j]["totalcount"]
                    )
                )

                if fallback:
                    chosenjudge = fallback[0]

            if chosenjudge is None:
                warningslist.append(f"{wod} - {heatlabel} - Couloir {lane}: Aucun juge disponible !")
                chosenjudge = "SANS JUGE"

            laneassignments[lane] = chosenjudge

            if chosenjudge != "SANS JUGE":
                planning[chosenjudge].append({
                    "wod": clean_text(str(row["Workout"])),
                    "lane": lane,
                    "athlete": clean_text(str(row["Competitor"])),
                    "division": clean_text(str(row["Division"])),
                    "start": clean_text(str(row["Heat Start Time"])),
                    "end": clean_text(str(row["Heat End Time"])),
                    "heat": clean_text(str(row["Heat"]),
                    "heatnum": row["heatnum"],
                })
                judgesworkingthisheat.add(chosenjudge)

        # 3) Mise à jour des compteurs après le heat
        for j in judges:
            s = state[j]
            if j in judgesworkingthisheat:
                s["heatssanspause"] += 1
                s["consecutiveoff"] = 0
                s["totalcount"] += 1

                if s["heatssanspause"] >= ONTARGET:
                    s["status"] = "OFF"
                    s["consecutiveoff"] = 0
                    s["currentlane"] = None
            else:
                if s["status"] == "AVAILABLE":
                    s["heatssanspause"] = 0
                elif s["status"] == "ON":
                    s["heatssanspause"] += 1
                    if s["heatssanspause"] >= ONTARGET:
                        s["status"] = "OFF"
                        s["consecutiveoff"] = 0
                        s["currentlane"] = None

        # 4) Mémorisation de la ligne dans le bloc courant
        for lane, j in laneassignments.items():
            if j != "SANS JUGE":
                s = state[j]
                if s["currentlane"] is None:
                    s["currentlane"] = lane
                elif s["currentlane"] != lane and s["heatssanspause"] < ONTARGET:
                    s["currentlane"] = lane

    return planning, warningslist


# ========================
# MAIN STREAMLIT
# ========================
def main():
    with st.sidebar:
        st.header("📂 Fichier d'entrée")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
        competition_name = st.text_input("Nom de la compétition", "Unicorn Throwdown")

        st.header("🙅‍♂️ Juges")
        judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
        if judges_file:
            try:
                judges_df = pd.read_csv(judges_file, header=None, encoding='latin1')
                judges = [clean_text(str(j)) for j in judges_df[0].dropna().tolist()]
            except:
                judges = []
        else:
            judges_text = st.text_area("Saisir les juges (un par ligne)", "Juge 1\nJuge 2\nJuge 3")
            judges = [clean_text(j.strip()) for j in judges_text.split('\n') if j.strip()]

        st.header("⚙️ Configuration du roulement")
        rotation_system = st.selectbox(
            "Système de roulement",
            options=[
                {"name": "3-on / 3-off (Recommandé)", "on": 3, "off": 3},
                {"name": "2-on / 2-off", "on": 2, "off": 2},
                {"name": "1-on / 1-off", "on": 1, "off": 1},
                {"name": "2-on / 1-off", "on": 2, "off": 1}
            ],
            format_func=lambda x: x["name"]
        )
        logo_file = st.file_uploader("Uploader un logo", type=["png", "jpg", "jpeg"])
        logo_path = None
        if logo_file:
            temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_logo.write(logo_file.read())
            temp_logo.close()
            logo_path = temp_logo.name

    if schedule_file and judges:
        schedule = load_schedule_from_excel(schedule_file)
        if schedule is None or schedule.empty: st.stop()
        
        wods = sorted(schedule['Workout'].dropna().unique())
        st.header("📅 Disponibilités des juges")
        disponibilites = {}
        cols = st.columns(3)
        for i, wod in enumerate(wods):
            with cols[i % 3]:
                with st.expander(f"{wod}"):
                    select_all = st.checkbox(f"Tout sélectionner ({wod})", key=f"sel_{wod}", value=True)
                    if select_all:
                        disponibilites[wod] = judges
                    else:
                        disponibilites[wod] = st.multiselect("Juges disponibles", judges, key=f"multi_{wod}")

        if st.button("🦄 Générer le planning"):
            planning, warnings = assign_judges_equitable(schedule, judges, disponibilites, rotation_system)

            if warnings:
                st.warning("🚨 **Alertes effectifs :**")
                for w in warnings: st.write(w)

            st.subheader("📊 Équilibre des assignations (Nombre total de Heats)")
            counts = {j: len(planning[j]) for j in judges}
            for j in sorted(counts, key=counts.get, reverse=True):
                st.write(f"✅ {j}: {counts[j]} Heats arbitrés au total")

            try:
                pdf_juges = generate_pdf_tableau(planning, competition_name, logo_path)
                pdf_heats = generate_heat_pdf(planning, competition_name, logo_path)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t1:
                    pdf_juges.output(t1.name)
                    with open(t1.name, "rb") as f:
                        st.download_button("📘 Télécharger planning par juge", f, "planning_juges.pdf")
                    os.unlink(t1.name)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t2:
                    pdf_heats.output(t2.name)
                    with open(t2.name, "rb") as f:
                        st.download_button("📗 Télécharger planning par heat", f, "planning_heats.pdf")
                    os.unlink(t2.name)
                st.success("✅ Planning généré en respectant le roulement et la conservation des lignes !")
            except Exception as e:
                st.error(f"❌ Erreur PDF: {str(e)}")
    else:
        st.info("👉 Veuillez importer le fichier Excel et spécifier les juges dans la barre latérale.")


if __name__ == "__main__":
    main()
