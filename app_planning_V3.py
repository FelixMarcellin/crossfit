# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 11.0 : Algorithme par Faisceaux de Lignes (Tracks) - Équité & Continuité Absolue
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


def load_schedule_from_excel(uploaded_file):
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
    
    df = df[df['Competitor'].notna() & (df['Competitor'] != "") & (~df['Competitor'].str.contains('EMPTY LANE', na=False))]
    return df


class FooterLogoPDF(FPDF):
    def __init__(self, logo_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo_path = logo_path
        self.set_auto_page_break(auto=True, margin=15)

    def footer(self):
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.set_y(-25)
                page_width = 210
                logo_width = 35
                x = (page_width - logo_width) / 2
                self.image(self.logo_path, x=x, y=self.get_y(), w=logo_width)
            except Exception as e:
                print(f"Erreur logo : {e}")


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
            if pdf.get_y() > 240:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 10)
                pdf.set_fill_color(211, 211, 211)
                for h, w in zip(headers, col_widths):
                    pdf.cell(w, 8, h, 1, 0, 'C', fill=True)
                pdf.ln()
                pdf.set_font("Arial", '', 9)
            
            pdf.set_fill_color(*row_colors[i % 2])
            vals = [
                f"{str(c['start'])[:5]}-{str(c['end'])[:5]}",
                str(c['lane']),
                str(c['wod'])[:12],
                str(c['heat'])[:8],
                str(c['athlete'])[:32],
                str(c['division'])[:17]
            ]
            for v, w in zip(vals, col_widths):
                pdf.cell(w, 7, clean_text(v), 1, 0, 'C', fill=True)
            pdf.ln()

        pdf.ln(5)
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 8, f"Total : {len(creneaux)} creneaux ", 0, 1)
    return pdf


def generate_heat_pdf(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (c['wod'], c['heat'], c['start'], c['end'])
            try:
                lane_key = int(float(c['lane']))
            except:
                lane_key = str(c['lane'])
            heat_map[key][lane_key] = juge

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
                block_height = header_height + (max_lanes * row_height) + 4
                current_y += block_height
            
            y_start = current_y
            pdf.set_font("Arial", 'B', 8)
            pdf.set_xy(x, y_start)
            header_text = f"{wod} | {heat_num} | {start[:5]}-{end[:5]}"
            pdf.cell(col_width, row_height, clean_text(header_text), border=1, align='C', fill=True)
            
            pdf.set_font("Arial", 'B', 7)
            pdf.set_xy(x, y_start + row_height)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(col_width * 0.35, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width * 0.65, row_height, "Juge", border=1, align='C', fill=True)
            
            pdf.set_font("Arial", '', 7)
            pdf.set_fill_color(255, 255, 255)
            
            row_num = 0
            for lane_num, juge_name in sorted(lanes.items(), key=lambda item: item[0]):
                y_pos = y_start + row_height * 2 + row_num * row_height
                pdf.set_xy(x, y_pos)
                pdf.cell(col_width * 0.35, row_height, clean_text(str(lane_num)), border=1, align='C')
                pdf.cell(col_width * 0.65, row_height, clean_text(juge_name)[:18], border=1, align='C')
                row_num += 1
    return pdf


def extract_heat_number(heat_str):
    if pd.isna(heat_str):
        return 0
    if isinstance(heat_str, (int, float)):
        return int(heat_str)
    numbers = re.findall(r'\d+', str(heat_str))
    return int(numbers[0]) if numbers else 0


# ====================================================
# ALGORITHME DE PLANIFICATION PAR FAISCEAUX DE LIGNES
# ====================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation_config):
    planning = {j: [] for j in judges}
    judge_counts = {j: 0 for j in judges}
    
    df = schedule.copy()
    df["heat_num"] = df["Heat #"].apply(extract_heat_number)
    df = df.sort_values(["Heat Start Time", "heat_num", "Lane"]).reset_index(drop=True)

    # Récupération ordonnée de la liste globale des Heats
    grouped_heats = df.groupby(["Workout", "heat_num", "Heat Start Time", "Heat End Time"], sort=False)
    heats_list = []
    for (wod, heat_num, start, end), g in grouped_heats:
        heats_list.append({
            "wod": wod,
            "heat_num": heat_num,
            "start": start,
            "end": end,
            "rows": sorted(g.to_dict("records"), key=lambda r: int(float(r["Lane"])))
        })

    total_heats = len(heats_list)
    assignments = {h_idx: {} for h_idx in range(total_heats)}

    # On traite la compétition WOD par WOD pour garantir la cohérence des blocs demandés
    wods_in_order = []
    for h in heats_list:
        if h["wod"] not in wods_in_order:
            wods_in_order.append(h["wod"])

    for current_wod in wods_in_order:
        # Extraire les indices de heats appartenant à ce WOD précis
        wod_heat_indices = [idx for idx, h in enumerate(heats_list) if h["wod"] == current_wod]
        if not wod_heat_indices:
            continue
            
        config = rotation_config.get(current_wod, {"on": 3, "off": 3})
        on_len = config["on"]
        off_len = config["off"]

        # Lister toutes les Lanes uniques requises pour ce WOD
        wod_lanes = set()
        for idx in wod_heat_indices:
            for r in heats_list[idx]["rows"]:
                wod_lanes.add(str(int(float(r["Lane"]))))
        wod_lanes = sorted(list(wod_lanes), key=int)

        # Pour chaque Lane, on va construire des blocs continus de Heats
        for lane in wod_lanes:
            # Récupérer uniquement les indices de heats où cette Lane est active
            active_lane_indices = []
            for idx in wod_heat_indices:
                if any(str(int(float(r["Lane"]))) == lane for r in heats_list[idx]["rows"]):
                    active_lane_indices.append(idx)

            # Découper ces indices actifs en blocs stricts de longueur maximale 'on_len'
            i = 0
            while i < len(active_lane_indices):
                # Constitution du sous-bloc de Heats consécutifs pour cette ligne
                block_indices = []
                while len(block_indices) < on_len and i < len(active_lane_indices):
                    # Vérifier s'il n'y a pas un "trou" temporel géant (ex: pause midi) pour scinder le bloc
                    if block_indices and (active_lane_indices[i] != active_lane_indices[i-1] + 1):
                        break
                    block_indices.append(active_lane_indices[i])
                    i += 1

                if not block_indices:
                    i += 1
                    continue

                # Recherche du meilleur juge disponible pour prendre CE BLOC COMPLET
                first_heat_idx = block_indices[0]
                eligible_judges = []

                for j in judges:
                    # Contrainte 1 : Disponibilité sur ce WOD spécifique
                    if j not in disponibilites.get(current_wod, judges):
                        continue
                    
                    # Contrainte 2 : Le juge ne doit pas être occupé ailleurs durant TOUT ce bloc
                    is_free = True
                    for b_idx in block_indices:
                        if j in assignments[b_idx].values():
                            is_free = False
                            break
                    if not is_free:
                        continue

                    # Contrainte 3 : Respect du temps de repos minimum requis avant le début du bloc
                    has_rested = True
                    for check_idx in range(max(0, first_heat_idx - off_len), first_heat_idx):
                        if j in assignments[check_idx].values():
                            has_rested = False
                            break
                    
                    if has_rested:
                        eligible_judges.append(j)

                # Sécurité fall-back : Si aucun juge ne respecte le repos, on prend n'importe quel juge libre
                if not eligible_judges:
                    eligible_judges = [
                        j for j in judges 
                        if j in disponibilites.get(current_wod, judges)
                        and all(j not in assignments[b_idx].values() for b_idx in block_indices)
                    ]

                if not eligible_judges:
                    continue # Protection ultime de planning s'il manque cruellement de personnel

                # ÉQUITÉ ABSOLUE : Choix du juge qui a le plus petit compteur d'heats cumulés au moment T
                eligible_judges = sorted(eligible_judges, key=lambda j: judge_counts[j])
                chosen_judge = eligible_judges[0]

                # Validation définitive du bloc pour ce juge
                for b_idx in block_indices:
                    assignments[b_idx][lane] = chosen_judge
                    judge_counts[chosen_judge] += 1

    # Reconstruction de la structure de données pour l'impression des PDF
    for h_idx, lanes_dict in assignments.items():
        h_data = heats_list[h_idx]
        for lane_id, j_name in lanes_dict.items():
            row_data = next((r for r in h_data["rows"] if str(int(float(r["Lane"]))) == lane_id), None)
            if row_data:
                planning[j_name].append({
                    "wod": clean_text(str(row_data["Workout"])),
                    "lane": lane_id,
                    "athlete": clean_text(str(row_data["Competitor"])),
                    "division": clean_text(str(row_data["Division"])),
                    "start": clean_text(str(row_data["Heat Start Time"])),
                    "end": clean_text(str(row_data["Heat End Time"])),
                    "heat": clean_text(str(row_data["Heat #"])),
                    "heat_num": h_data["heat_num"]
                })

    return planning


# ========================
# MAIN STREAMLIT
# ========================
def main():
    with st.sidebar:
        st.header("📂 Fichier d'entrée")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])

        st.header("🏋️‍♀️ Nom de la compétition")
        competition_name = st.text_input("Nom à afficher sur les PDF", "Unicorn Throwdown 2026")

        st.header("🙅‍♂️ Juges")
        judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
        if judges_file:
            try:
                judges_df = pd.read_csv(judges_file, header=None, encoding='latin1')
                judges = [clean_text(str(j)) for j in judges_df[0].dropna().tolist()]
            except Exception as e:
                st.error(f"Erreur CSV : {e}")
                judges = []
        else:
            judges_text = st.text_area("Saisir les juges (un par ligne)", "Orleane\nDamien\nElodie\nMorgane\nMelanie\nPierre\nClement\nJonathan R\nClea\nBenjamin\nMarie\nDavid\nGregory\nJonathan DO")
            judges = [clean_text(j.strip()) for j in judges_text.split('\n') if j.strip()]

        st.header("🖼️ Logo (pied de page)")
        logo_file = st.file_uploader("Uploader un logo", type=["png", "jpg", "jpeg"])
        logo_path = None
        if logo_file:
            temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_logo.write(logo_file.read())
            temp_logo.close()
            logo_path = temp_logo.name

    if schedule_file and judges:
        try:
            schedule = load_schedule_from_excel(schedule_file)
            if schedule is None or schedule.empty:
                st.stop()
            
            wods = sorted(schedule['Workout'].dropna().unique())
            rotation_by_wod = {}

            st.header("⚙️ Roulement par WOD")
            st.info("Choisissez la configuration de roulement.")
            
            rotation_options = [
                {"name": "1-on / 1-off", "on": 1, "off": 1},
                {"name": "2-on / 1-off", "on": 2, "off": 1},
                {"name": "2-on / 2-off", "on": 2, "off": 2},
                {"name": "3-on / 2-off", "on": 3, "off": 2},
                {"name": "3-on / 3-off", "on": 3, "off": 3},
                {"name": "4-on / 2-off", "on": 4, "off": 2}
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
                        select_all = st.checkbox(f"Tout sélectionner ({wod})", key=f"sel_{wod}", value=True)
                        if select_all:
                            disponibilites[wod] = judges
                        else:
                            disponibilites[wod] = st.multiselect("Juges disponibles", judges, default=judges, key=f"multi_{wod}")

            if st.button("🦄 Générer le planning parfait"):
                planning = assign_judges_equitable(schedule, judges, disponibilites, rotation_by_wod)

                st.subheader("📊 Équilibre final des assignations")
                counts = {j: len(planning[j]) for j in judges}
                
                # Tri décroissant pour voir les écarts en direct
                for j in sorted(counts, key=counts.get, reverse=True):
                    st.write(f"🔹 **{j}** : {counts[j]} heats arbitrés")

                try:
                    pdf_juges = generate_pdf_tableau(planning, competition_name, logo_path)
                    pdf_heats = generate_heat_pdf(planning, competition_name, logo_path)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t1:
                        pdf_juges.output(t1.name)
                        with open(t1.name, "rb") as f:
                            st.download_button("📘 Télécharger planning par juge", f, "planning_juges_v11.pdf")
                        os.unlink(t1.name)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t2:
                        pdf_heats.output(t2.name)
                        with open(t2.name, "rb") as f:
                            st.download_button("📗 Télécharger planning par heat", f, "planning_heats_v11.pdf")
                        os.unlink(t2.name)
                    
                    st.success("✅ Planning mathématiquement parfait généré avec succès !")
                except Exception as e:
                    st.error(f"❌ Erreur PDF : {str(e)}")

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
    else:
        st.info("👉 Importez le fichier Excel 'Heats' et configurez vos juges sur le panneau latéral.")


if __name__ == "__main__":
    main()
