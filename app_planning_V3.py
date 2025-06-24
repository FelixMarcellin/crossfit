# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:46:15 2025

@author: felima
"""
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

st.set_page_config(page_title="Planning Juges CrossFit", layout="wide")
st.title("🧑‍⚖️ Gestion des Juges - Unicorn Throwdown 2025")

def generate_pdf_tableau(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    print("✅ Appel de generate_pdf_tableau() ✔️")

    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)

    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Unicorn Throwdown 2025", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(10)

        col_widths = [30, 10, 15, 50, 25, 40]
        headers = ["Heure", "Lane", "WOD", "Athlète", "Division", "Emplacement"]

        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])
            start = c['start']
            end = c['end']
            start_time = start if isinstance(start, str) else start.strftime('%H:%M')
            end_time = end if isinstance(end, str) else end.strftime('%H:%M')

            data = [
                f"{start_time} - {end_time}",
                str(c['lane']),
                c['wod'],
                c['athlete'],
                c['division'],
                c['location']
            ]

            heights = []
            for value, width in zip(data, col_widths):
                lines = FPDF().multi_cell(width, 5, str(value), border=0, align='C', split_only=True)
                heights.append(len(lines) * 5)
            max_height = max(heights)

            x_init = pdf.get_x()
            y_init = pdf.get_y()
            for val, width in zip(data, col_widths):
                x = pdf.get_x()
                y = pdf.get_y()
                pdf.multi_cell(width, 5, str(val), border=1, align='C', fill=True)
                pdf.set_xy(x + width, y)
            pdf.ln(max_height)

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf

def generate_heat_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    print("✅ Appel de generate_heat_pdf() ✔️")

    heat_map = defaultdict(lambda: defaultdict(str))

    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (str(c['start']), str(c['end']), c['wod'], c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = list(heat_map.items())
    heats.sort(key=lambda x: x[0][0])

    for i in range(0, len(heats), 2):
        pdf.add_page()
        for j in range(2):
            if i + j >= len(heats):
                break

            (start, end, wod, location), lanes = heats[i + j]
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"📆 HEAT – {start} à {end}", ln=1)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 6, f"🏋️ WOD : {wod} | 📍 Emplacement : {location}", ln=1)
            pdf.ln(2)

            for lane in sorted(lanes):
                juge = lanes[lane]
                pdf.cell(0, 6, f"Lane {lane} : {juge}", ln=1)

            pdf.ln(6)

    return pdf

def main():
    with st.sidebar:
        st.header("📤 Import des fichiers")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
        judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])

    if schedule_file and judges_file:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            # ➕ Correction encodage ici :
            judges = pd.read_csv(judges_file, header=None, encoding='latin1')[0].dropna().tolist()

            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=True)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            wods = sorted(schedule['Workout'].unique())

            st.header("📝 Disponibilité des Juges par WOD")
            disponibilites = {wod: set() for wod in wods}

            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"WOD: {wod}"):
                        disponibilites[wod] = set(st.multiselect(
                            f"Sélectionnez les juges disponibles",
                            judges,
                            key=f"dispo_{wod}"
                        ))

            if st.button("✨ Générer les plannings"):
                planning = {juge: [] for juge in judges}

                for _, row in schedule.iterrows():
                    wod = row['Workout']
                    juges_dispo = disponibilites[wod]

                    if not juges_dispo:
                        st.error(f"Aucun juge disponible pour le {wod}!")
                        continue

                    juge_attribue = min(juges_dispo, key=lambda j: len(planning[j]))
                    planning[juge_attribue].append({
                        'wod': wod,
                        'lane': row['Lane'],
                        'athlete': row['Competitor'],
                        'division': row['Division'],
                        'location': row['Workout Location'],
                        'start': row['Heat Start Time'],
                        'end': row['Heat End Time']
                    })

                pdf_juges = generate_pdf_tableau({k: v for k, v in planning.items() if v})
                pdf_heats = generate_heat_pdf({k: v for k, v in planning.items() if v})

                # Téléchargement PDF juges
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_juges:
                    pdf_juges.output(tmp_juges.name)
                    with open(tmp_juges.name, "rb") as f:
                        st.download_button(
                            "📥 Télécharger planning par juge",
                            data=f,
                            file_name="planning_juges.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(tmp_juges.name)

                # Téléchargement PDF heats
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_heats:
                    pdf_heats.output(tmp_heats.name)
                    with open(tmp_heats.name, "rb") as f:
                        st.download_button(
                            "📥 Télécharger planning par heat",
                            data=f,
                            file_name="planning_heats.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(tmp_heats.name)

                st.success("✅ PDF générés avec succès !")
                st.header("📊 Récapitulatif des affectations")
                for juge, creneaux in planning.items():
                    if creneaux:
                        with st.expander(f"Juge: {juge} ({len(creneaux)} créneaux)"):
                            st.table(pd.DataFrame(creneaux))

        except Exception as e:
            st.error(f"Erreur lors du traitement: {str(e)}")
    else:
        st.info("Veuillez uploader les fichiers pour commencer")

if __name__ == "__main__":
    main()
