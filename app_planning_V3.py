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
def assign_judges_equitable(schedule, judges, disponibilites, rotation_config):
    planning = {j: [] for j in judges}
    warnings_list = []

    # Tri chronologique exact des Heats
    df = schedule.copy()
    df["heat_num"] = df["Heat #"].apply(extract_heat_number)
    df = df.sort_values(["Heat Start Time", "heat_num", "Lane"]).reset_index(drop=True)

    heats = []
    for (wod, heat_num, start, end), g in df.groupby(["Workout", "heat_num", "Heat Start Time", "Heat End Time"]):
        heats.append({
            "wod": wod, "heat_num": heat_num, "start": start, "end": end,
            "rows": sorted(g.to_dict("records"), key=lambda r: int(float(r["Lane"])))
        })

    ON_TARGET = rotation_config["on"]
    OFF_TARGET = rotation_config["off"]

    # État individuel de chaque juge
    state = {
        j: {
            "status": "AVAILABLE",  # AVAILABLE, ON, OFF
            "history": [],          # Liste de 'ON' ou 'OFF' pour les derniers heats globaux
            "consecutive_on": 0,    # Nombre de Heats consécutifs arbitrés
            "consecutive_off": 0,   # Nombre de Heats consécutifs de repos
            "total_count": 0        # Compteur global pour équilibrer
        } for j in judges
    }

    # Dictionnaire pour forcer la conservation de la ligne : { lane_number: nom_du_juge }
    lane_assignments = {}

    for heat in heats:
        wod = heat["wod"]
        heat_label = f"Heat {heat['heat_num']} ({heat['start']})"
        dispo = disponibilites.get(wod, judges)
        if not dispo: dispo = judges

        # Étape 1 : Mettre à jour l'état des juges qui ont atteint leur limite au Heat précédent
        for j in judges:
            if state[j]["status"] == "ON" and state[j]["consecutive_on"] >= ON_TARGET:
                state[j]["status"] = "OFF"
                state[j]["consecutive_on"] = 0
                state[j]["consecutive_off"] = 0
                # Si ce juge avait une ligne attitrée, on la libère pour qu'il aille se reposer
                for l, assigned_juge in list(lane_assignments.items()):
                    if assigned_juge == j:
                        del lane_assignments[l]

            elif state[j]["status"] == "OFF" and state[j]["consecutive_off"] >= OFF_TARGET:
                state[j]["status"] = "AVAILABLE"
                state[j]["consecutive_off"] = 0

        # Liste des juges qui vont travailler sur ce Heat (pour éviter qu'un juge fasse 2 lignes)
        judges_working_this_heat = set()

        # Étape 2 : Assigner les lignes du Heat courant
        for row in heat["rows"]:
            lane = str(int(float(row["Lane"])))
            chosen_judge = None

            # 1. Tenter de conserver le juge qui était déjà sur cette ligne
            if lane in lane_assignments:
                previous_judge = lane_assignments[lane]
                # Le juge peut rester si : il est dispo, pas encore en OFF, et n'est pas déjà pris dans ce heat
                if previous_judge in dispo and state[previous_judge]["status"] != "OFF" and previous_judge not in judges_working_this_heat:
                    # On tolère +1 heat si nécessaire pour conserver la ligne avant le switch
                    if state[previous_judge]["consecutive_on"] < (ON_TARGET + 1):
                        chosen_judge = previous_judge

            # 2. Si pas de juge ou juge en repos, on cherche un NOUVEAU juge pour la ligne
            if chosen_judge is None:
                # Candidats : Disponibles pour le WOD, libres dans ce heat, et pas en repos
                candidates = [j for j in judges if j in dispo and j not in judges_working_this_heat and state[j]["status"] in ["AVAILABLE", "ON"]]
                
                # Priorité absolue à ceux qui reviennent de repos (AVAILABLE)
                candidates = sorted(candidates, key=lambda j: (0 if state[j]["status"] == "AVAILABLE" else 1, state[j]["total_count"]))
                
                # Si aucun dispo standard, on pioche de force chez ceux en repos (OFF) ou indispos
                if not candidates:
                    candidates = [j for j in judges if j not in judges_working_this_heat]
                    candidates = sorted(candidates, key=lambda j: (state[j]["consecutive_on"], state[j]["total_count"]))

                if candidates:
                    chosen_judge = candidates[0]
                    lane_assignments[lane] = chosen_judge
                    state[chosen_judge]["status"] = "ON"
                else:
                    chosen_judge = "SANS JUGE"
                    warnings_list.append(f"⚠️ {wod} | {heat_label} | Couloir {lane} : Aucun juge disponible !")

            # Enregistrement du créneau si un juge est trouvé
            if chosen_judge != "SANS JUGE":
                planning[chosen_judge].append({
                    "wod": clean_text(str(row["Workout"])),
                    "lane": lane,
                    "athlete": clean_text(str(row["Competitor"])),
                    "division": clean_text(str(row["Division"])),
                    "start": clean_text(str(row["Heat Start Time"])),
                    "end": clean_text(str(row["Heat End Time"])),
                    "heat": clean_text(str(row["Heat #"])),
                    "heat_num": heat["heat_num"]
                })
                judges_working_this_heat.add(chosen_judge)

        # Étape 3 : Incrémentation des compteurs de fin de Heat pour tout le monde
        for j in judges:
            if j in judges_working_this_heat:
                state[j]["status"] = "ON"
                state[j]["consecutive_on"] += 1
                state[j]["consecutive_off"] = 0
                state[j]["total_count"] += 1
            else:
                if state[j]["status"] == "ON":
                    # Le juge n'a pas travaillé sur ce heat mais était actif (trou dans le planning)
                    # On ne compte pas de heat consécutif mais on ne le met pas en repos forcé tout de suite
                    state[j]["consecutive_on"] = max(0, state[j]["consecutive_on"] - 1)
                elif state[j]["status"] == "OFF":
                    state[j]["consecutive_off"] += 1
                else:
                    # Un juge AVAILABLE qui ne travaille pas commence à accumuler du repos virtuel
                    state[j]["consecutive_off"] += 1

    return planning, warnings_list


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
