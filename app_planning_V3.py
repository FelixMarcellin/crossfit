# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:48:50 2025
@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from typing import Dict, List
from collections import defaultdict
import traceback

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin")

# ---------------------------- PDF FUNCTIONS ---------------------------- #

def generate_pdf_tableau(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)

    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Nom de la compétition", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(10)

        col_widths = [30, 10, 15, 50, 25, 40]
        headers = ["Heure", "Lane", "WOD", "Athlete", "Division", "Emplacement"]

        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])
            start_time = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end_time = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']

            data = [
                f"{start_time} - {end_time}",
                str(c['lane']),
                c['wod'],
                c['athlete'],
                c['division'],
                c['location']
            ]

            for val, width in zip(data, col_widths):
                pdf.cell(width, 10, str(val), border=1, align='C', fill=True)
            pdf.ln()

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf

def generate_heat_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))

    for juge, creneaux in planning.items():
        for c in creneaux:
            start = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']
            key = (c['wod'], c['heat'], start, end, c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = sorted(heat_map.items(), key=lambda x: (x[0][0], x[0][1]))

    for i in range(0, len(heats), 2):
        pdf.add_page()
        col_width = 90
        row_height = 8
        spacing = 15

        for j in range(2):
            if i + j >= len(heats):
                break

            (wod, heat, start, end, location), lanes = heats[i + j]
            x_position = 10 + j * (col_width + spacing)

            pdf.set_font("Arial", 'B', 10)
            pdf.set_xy(x_position, 15)
            pdf.cell(col_width, row_height, f"WOD: {wod} - Heat {heat}", border=1, align='C', fill=True)

            pdf.set_font("Arial", '', 9)
            pdf.set_xy(x_position, 15 + row_height)
            pdf.cell(col_width, row_height, f"Heure: {start} - {end}", border=1, align='C')

            pdf.set_xy(x_position, 15 + 2 * row_height)
            pdf.cell(col_width, row_height, f"Emplacement: {location}", border=1, align='C')

            pdf.set_font("Arial", 'B', 9)
            pdf.set_xy(x_position, 15 + 3 * row_height)
            pdf.cell(col_width / 2, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width / 2, row_height, "Juge", border=1, align='C', fill=True)

            pdf.set_font("Arial", '', 9)
            for k, lane in enumerate(sorted(lanes)):
                y_position = 15 + (4 + k) * row_height
                pdf.set_xy(x_position, y_position)
                pdf.cell(col_width / 2, row_height, str(lane), border=1, align='C')
                pdf.cell(col_width / 2, row_height, lanes[lane], border=1, align='C')

    return pdf

# ---------------------------- MAIN APP ---------------------------- #

def main():
    with st.sidebar:
        st.header("Import des fichiers")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
        
        st.header("Saisie des juges")
        input_method = st.radio("Méthode de saisie des juges", options=["Fichier CSV", "Saisie manuelle"], index=0)
        
        judges = []
        if input_method == "Fichier CSV":
            judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
            if judges_file:
                judges = pd.read_csv(judges_file, header=None, encoding='latin1')[0].dropna().tolist()
        else:
            judges_text = st.text_area("Saisir les noms des juges (un par ligne)", value="Juge 1\nJuge 2\nJuge 3", height=150)
            judges = [j.strip() for j in judges_text.split('\n') if j.strip()]
        
        st.header("Rotation des juges")
        rotation_heats = st.selectbox("Nombre de heats consécutifs par juge", options=[1,2], index=0)

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            st.subheader("Aperçu du planning importé")
            st.dataframe(schedule.head())

            required_columns = ['Workout', 'Heat #', 'Lane', 'Competitor', 'Division', 'Workout Location', 'Heat Start Time', 'Heat End Time']
            if not all(col in schedule.columns for col in required_columns):
                st.error("Colonnes manquantes.")
                return

            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            wods = sorted(schedule['Workout'].unique())

            st.header("Disponibilité des Juges par WOD")
            disponibilites = {}
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"WOD: {wod}"):
                        select_all = st.checkbox(f"Tout sélectionner pour {wod}", key=f"select_all_{wod}")
                        if select_all:
                            selected_judges = judges
                        else:
                            selected_judges = st.multiselect(f"Sélection pour {wod}", judges, key=f"dispo_{wod}")
                        disponibilites[wod] = list(selected_judges)

            if st.button("Générer les plannings"):
                planning = {j: [] for j in judges}

                for wod in wods:
                    data_wod = schedule[schedule['Workout'] == wod].sort_values(['Heat #','Lane'])
                    heats = sorted(data_wod['Heat #'].unique())
                    juge_order = disponibilites[wod]
                    if not juge_order:
                        st.error(f"Aucun juge sélectionné pour {wod}")
                        continue
                    heat_idx = 0
                    while heat_idx < len(heats):
                        for juge in juge_order:
                            for r in range(rotation_heats):
                                if heat_idx+r >= len(heats):
                                    break
                                heat_num = heats[heat_idx + r]
                                lines = data_wod[data_wod['Heat #'] == heat_num]
                                for _, row in lines.iterrows():
                                    planning[juge].append({
                                        'wod': wod,
                                        'heat': heat_num,
                                        'lane': row['Lane'],
                                        'athlete': row['Competitor'],
                                        'division': row['Division'],
                                        'location': row['Workout Location'],
                                        'start': row['Heat Start Time'],
                                        'end': row['Heat End Time']
                                    })
                        heat_idx += rotation_heats * len(juge_order)

                pdf_juges = generate_pdf_tableau({k:v for k,v in planning.items() if v})
                pdf_heats = generate_heat_pdf({k:v for k,v in planning.items() if v})

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_juges:
                    pdf_juges.output(tmp_juges.name)
                    with open(tmp_juges.name,"rb") as f:
                        st.download_button("Télécharger planning par juge", f, "planning_juges.pdf", "application/pdf")
                    os.unlink(tmp_juges.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_heats:
                    pdf_heats.output(tmp_heats.name)
                    with open(tmp_heats.name,"rb") as f:
                        st.download_button("Télécharger planning par heat", f, "planning_heats.pdf", "application/pdf")
                    os.unlink(tmp_heats.name)

                st.success("PDF générés avec succès!")
                st.header("Récapitulatif des affectations")
                for juge, creneaux in planning.items():
                    if creneaux:
                        with st.expander(f"{juge} ({len(creneaux)} créneaux)"):
                            st.table(pd.DataFrame(creneaux))

        except Exception:
            st.error("Erreur lors du traitement:")
            st.code(traceback.format_exc())
    else:
        st.info("Uploader le fichier et saisir les juges pour commencer.")

if __name__ == "__main__":
    main()
