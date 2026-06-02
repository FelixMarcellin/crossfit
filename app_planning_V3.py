# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 9.6 : Lecture directe de la feuille "Heats" + logo en bas de page
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
    """Lit le fichier Excel et extrait la feuille 'Heats' avec les bonnes colonnes"""
    # Lire toutes les feuilles
    xls = pd.ExcelFile(uploaded_file)
    
    # Chercher la feuille "Heats" (insensible à la casse)
    sheet_name = None
    for name in xls.sheet_names:
        if name.lower() == "heats":
            sheet_name = name
            break
    
    if sheet_name is None:
        st.error("❌ Feuille 'Heats' introuvable dans le fichier Excel.")
        return None
    
    # Lire la feuille
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    
    # Vérifier les colonnes nécessaires (adapté à la structure du fichier brut)
    # Colonnes attendues: Lane, Competitor, Division, Workout, Workout Location, Heat #, Heat Start Time, Heat End Time
    required_cols = ['Lane', 'Competitor', 'Division', 'Workout', 'Workout Location', 'Heat #', 'Heat Start Time', 'Heat End Time']
    
    # Vérifier si toutes les colonnes sont présentes
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes dans la feuille 'Heats': {missing_cols}")
        st.info("📋 Les colonnes attendues sont: Lane, Competitor, Division, Workout, Workout Location, Heat #, Heat Start Time, Heat End Time")
        return None
    
    # Renommer les colonnes pour correspondre au reste du code
    df = df.rename(columns={
        'Workout Location': 'Workout Location',
        'Heat #': 'Heat #',
        'Heat Start Time': 'Heat Start Time',
        'Heat End Time': 'Heat End Time'
    })
    
    # Nettoyer les données
    for col in ['Workout', 'Competitor', 'Division', 'Workout Location', 'Heat #']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_text(str(x)) if pd.notna(x) else "")
    
    # Filtrer les lignes vides
    df = df[df['Competitor'].notna()]
    df = df[df['Competitor'] != ""]
    df = df[~df['Competitor'].str.contains('EMPTY LANE', na=False)]
    
    return df


# ========================
# PDF AVEC LOGO EN BAS DE PAGE
# ========================
class FooterLogoPDF(FPDF):
    def __init__(self, logo_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo_path = logo_path
        self.set_auto_page_break(auto=True, margin=15)

    def footer(self):
        """Pied de page avec logo centré"""
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.set_y(-70)
                page_width = 210
                logo_width = 50
                x = (page_width - logo_width) / 2
                self.image(self.logo_path, x=x, y=self.get_y(), w=logo_width)
            except Exception as e:
                print(f"Erreur logo pied de page: {e}")


# ========================
# PDF PAR JUGE
# ========================
def generate_pdf_tableau(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    pdf = FooterLogoPDF(logo_path=logo_path, orientation='P')
    total_width = 180
    
    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        def parse_time(x):
            try:
                return pd.to_datetime(x, format='%H:%M')
            except:
                try:
                    return pd.to_datetime(str(x))
                except:
                    return pd.NaT

        creneaux = sorted(creneaux, key=lambda c: parse_time(c.get('start', '')))

        pdf.add_page()
        
        # En-tête
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
            
            start = clean_text(str(c['start']))[:5]
            end = clean_text(str(c['end']))[:5]
            athlete_name = clean_text(str(c['athlete']))
            if len(athlete_name) > 35:
                athlete_name = athlete_name[:32] + "..."
            division_name = clean_text(str(c['division']))
            if len(division_name) > 20:
                division_name = division_name[:17] + "..."
            wod_name = clean_text(str(c['wod']))
            if len(wod_name) > 15:
                wod_name = wod_name[:12] + "..."
            heat_name = clean_text(str(c['heat']))
            if len(heat_name) > 10:
                heat_name = heat_name[:8] + "..."
            
            vals = [
                f"{start}-{end}",
                clean_text(str(c['lane'])),
                wod_name,
                heat_name,
                athlete_name,
                division_name
            ]
            
            for v, w in zip(vals, col_widths):
                pdf.cell(w, 7, v, 1, 0, 'C', fill=True)
            pdf.ln()

        pdf.ln(5)
        pdf.set_font("Arial", 'I', 9)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total : {len(creneaux)} créneaux ", 0, 1)

    return pdf


# ========================
# PDF PAR HEAT
# ========================
def generate_heat_pdf(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (c['wod'], c['heat'], c['start'], c['end'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FooterLogoPDF(logo_path=logo_path, orientation='P')

    heats = sorted(
        heat_map.items(),
        key=lambda x: (
            x[0][2],      # 1. Heure de début (Chronologique avant tout)
            x[0][0],      # 2. Nom du WOD (Si deux WODs commencent en même temps)
            x[0][1]       # 3. Numéro du Heat
        )
    )

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
                prev_lanes = len(heats[i+1][1]) if i+1 < len(heats) else 0
                max_lanes = max(len(heats[i][1]), len(heats[i+1][1]))
                block_height = header_height + (max_lanes * row_height) + 4
                current_y += block_height
            
            y_start = current_y
            
            # En-tête du heat
            pdf.set_font("Arial", 'B', 8)
            pdf.set_xy(x, y_start)
            header_text = clean_text(f"{wod} | {heat_num} | {start}-{end}")
            pdf.cell(col_width, row_height, header_text, border=1, align='C', fill=True)
            
            # Sous-en-tête
            pdf.set_font("Arial", 'B', 7)
            pdf.set_xy(x, y_start + row_height)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(col_width * 0.35, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width * 0.65, row_height, "Juge", border=1, align='C', fill=True)
            
            # Contenu
            pdf.set_font("Arial", '', 7)
            pdf.set_fill_color(255, 255, 255)
            
            row_num = 0
            for lane_num, juge_name in sorted(lanes.items()):
                y_pos = y_start + row_height * 2 + row_num * row_height
                pdf.set_xy(x, y_pos)
                pdf.cell(col_width * 0.35, row_height, clean_text(str(lane_num)), border=1, align='C')
                juge_display = clean_text(juge_name)
                if len(juge_display) > 18:
                    juge_display = juge_display[:16] + ".."
                pdf.cell(col_width * 0.65, row_height, juge_display, border=1, align='C')
                row_num += 1

    return pdf


# ========================
# HELPERS
# ========================
def extract_heat_number(heat_str):
    if pd.isna(heat_str):
        return 0
    if isinstance(heat_str, (int, float)):
        return int(heat_str)
    numbers = re.findall(r'\d+', str(heat_str))
    return int(numbers[0]) if numbers else 0


# ========================
# ATTRIBUTION ÉQUILIBRÉE
# ========================
def assign_judges_equitable(schedule, judges, disponibilites, rotation_config):

    planning = {j: [] for j in judges}

    judge_order = {
        judge: idx
        for idx, judge in enumerate(judges)
    }

    df = schedule.copy()
    df["heat_num"] = df["Heat #"].apply(extract_heat_number)

    df = df.sort_values(
        ["Heat Start Time", "heat_num", "Lane"]
    ).reset_index(drop=True)

    heats = []

    for (wod, heat_num, start, end), g in df.groupby(
        ["Workout", "heat_num", "Heat Start Time", "Heat End Time"]
    ):
        heats.append({
            "wod": wod,
            "heat_num": heat_num,
            "start": start,
            "end": end,
            "rows": sorted(
                g.to_dict("records"),
                key=lambda r: int(float(r["Lane"]))
            )
        })



    # ====================================================
    # Etat des juges
    # ====================================================

    state = {
        j: {
            "phase": "AVAILABLE",
            "remaining": 0,
            "off_value": 0,
            "lane": None,
            "count": 0
        }
        for j in judges
    }

    current_block = {}  # lane -> judge

    # ====================================================
    # Parcours des heats
    # ====================================================

    for heat_idx, heat in enumerate(heats):

        wod = heat["wod"]

        rotation = rotation_config.get(
            wod,
            {"on": 3, "off": 3}
        )
        
        ON = rotation["on"]
        OFF = rotation["off"]

        dispo = disponibilites.get(wod)

        if not dispo:
            dispo = judges

        lanes = sorted(
            {str(int(float(r["Lane"]))) for r in heat["rows"]},
            key=int
        )

        # ====================================================
        # Début d'un nouveau bloc
        # ====================================================

        new_block = False

        if not current_block:
            new_block = True

        else:
            active_judges = {
                j
                for j, s in state.items()
                if s["phase"] == "ON"
            }

            if len(active_judges) == 0:
                new_block = True

        # ====================================================
        # Construction du bloc
        # ====================================================

        if new_block:

            current_block = {}

            # candidats réellement libres
            candidates = []

            for j in judges:

                s = state[j]

                if j not in dispo:
                    continue

                if s["phase"] == "OFF":
                    continue

                candidates.append(j)

            # équilibre global
            candidates = sorted(
                candidates,
                key=lambda j: (
                    state[j]["count"],
                    judge_order[j]
                )
            )

            # ====================================================
            # Si pas assez de juges :
            # autorisation OFF restant = 1
            # ====================================================

            if len(candidates) < len(lanes):

                extra = []

                for j in judges:

                    s = state[j]

                    if j not in dispo:
                        continue

                    if s["phase"] == "OFF" and s["remaining"] <= 1:
                        extra.append(j)

                extra = sorted(
                    extra,
                    key=lambda j: (
                        state[j]["count"],
                        judge_order[j]
                    )
                )

                for j in extra:
                    if j not in candidates:
                        candidates.append(j)

            # ====================================================
            # Affectation lane fixe
            # ====================================================

            for lane, judge in zip(lanes, candidates):

                current_block[lane] = judge

                s = state[judge]

                s["phase"] = "ON"
                s["remaining"] = ON
                s["off_value"] = OFF
                s["lane"] = lane

        # ====================================================
        # Attribution du heat
        # ====================================================

        worked_this_heat = set()

        for row in heat["rows"]:

            lane = str(int(float(row["Lane"])))

            judge = current_block.get(lane)

            if judge is None:
                continue

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

        # ====================================================
        # Mise à jour des compteurs
        # ====================================================

        for judge in judges:

            s = state[judge]

            if judge in worked_this_heat:

                if s["phase"] == "ON":

                    s["remaining"] -= 1

                    if s["remaining"] <= 0:

                        s["phase"] = "OFF"
                        s["remaining"] = s["off_value"]
                        s["lane"] = None

            else:

                if s["phase"] == "OFF":

                    s["remaining"] -= 1

                    if s["remaining"] <= 0:

                        s["phase"] = "AVAILABLE"
                        s["remaining"] = 0

        # ====================================================
        # Nettoyage du bloc :
        # retirer les juges passés OFF
        # ====================================================

        lanes_to_remove = []

        for lane, judge in current_block.items():

            if state[judge]["phase"] != "ON":
                lanes_to_remove.append(lane)

        for lane in lanes_to_remove:
            del current_block[lane]

    return planning

# ========================
# MAIN STREAMLIT
# ========================
def main():
    with st.sidebar:
        st.header("📂 Fichier d'entrée")
        st.info("📌 Le fichier Excel doit contenir une feuille nommée **'Heats'** avec les colonnes: Lane, Competitor, Division, Workout, Workout Location, Heat #, Heat Start Time, Heat End Time")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])

        st.header("🏋️‍♀️ Nom de la compétition")
        competition_name = st.text_input("Nom à afficher sur les PDF", "Unicorn")

        st.header("🙅‍♂️ Juges")
        judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
        if judges_file:
            judges_df = pd.read_csv(judges_file, header=None, encoding='latin1')
            judges = [clean_text(str(j)) for j in judges_df[0].dropna().tolist()]
        else:
            judges_text = st.text_area("Saisir les juges (un par ligne)", "Juge 1\nJuge 2\nJuge 3")
            judges = [clean_text(j.strip()) for j in judges_text.split('\n') if j.strip()]

    
        
        
        st.header("🖼️ Logo (pied de page)")
        st.info("Le logo apparaîtra en bas de chaque page")
        logo_file = st.file_uploader("Uploader un logo", type=["png", "jpg", "jpeg"])

        logo_path = None
        if logo_file:
            temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_logo.write(logo_file.read())
            temp_logo.close()
            logo_path = temp_logo.name

    if schedule_file and judges:
        try:
            # Lire directement la feuille "Heats"
            schedule = load_schedule_from_excel(schedule_file)
            
            if schedule is None:
                st.stop()
            
            if schedule.empty:
                st.error("❌ Aucune donnée trouvée dans la feuille 'Heats'")
                st.stop()
            
            wods = sorted(schedule['Workout'].dropna().unique())

            rotation_by_wod = {}

            st.header("⚙️ Roulement par WOD")
            st.info(
                    "Choisissez un roulement spécifique pour chaque WOD "
                    "(ex : 3-on/3-off, 2-on/2-off, 4-on/2-off)."
                )
            
            rotation_options = [
                {"name": "1-on/1-off", "on": 1, "off": 1},
                {"name": "2-on/1-off", "on": 2, "off": 1},
                {"name": "2-on/2-off", "on": 2, "off": 2},
                {"name": "3-on/2-off", "on": 3, "off": 2},
                {"name": "3-on/3-off", "on": 3, "off": 3},
                {"name": "4-on/2-off", "on": 4, "off": 2}
            ]
            
            for wod in wods:
            
                rotation_by_wod[wod] = st.selectbox(
                    f"Roulement {wod}",
                    rotation_options,
                    index=4,
                    format_func=lambda x: x["name"],
                    key=f"rotation_{wod}"
                )
                        
            st.header("📅 Disponibilités des juges")
            disponibilites = {}
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"{wod}"):
                        select_all = st.checkbox(f"Tout sélectionner ({wod})", key=f"sel_{wod}")
                        if select_all:
                            disponibilites[wod] = judges
                        else:
                            disponibilites[wod] = st.multiselect("Juges disponibles", judges, key=f"multi_{wod}")

            if st.button("🦄 Générer le planning"):
                planning = assign_judges_equitable(schedule,judges,disponibilites,rotation_by_wod)

                st.subheader("📊 Équilibre des assignations")
                counts = {j: len(planning[j]) for j in judges}
                total = sum(counts.values())
                target = total // len(judges) if len(judges) > 0 else 0
                
                st.write("### Roulements utilisés")

                for wod, rotation in rotation_by_wod.items():
                    st.write(
                        f"**{wod}** : {rotation['on']}-on / {rotation['off']}-off"
                    )
                
                for j in sorted(counts, key=counts.get, reverse=True):
                    ecart = counts[j] - target
                    emoji = "✅" if abs(ecart) <= 1 else "⚠️"
                    st.write(f"{emoji} {j}: {counts[j]} créneaux ({ecart:+d})")

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
                    
                    st.success("✅ Planning généré avec succès !")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération des PDF: {str(e)}")
                    st.info("💡 Essayez de simplifier les noms des juges ou des athlètes")

        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")
            st.info("💡 Vérifiez que le fichier Excel contient bien une feuille nommée 'Heats'")

    else:
        st.info("👉 Veuillez importer un fichier Excel contenant une feuille 'Heats' et saisir la liste des juges.")


if __name__ == "__main__":
    main()
