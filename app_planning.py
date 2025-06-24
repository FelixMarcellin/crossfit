# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:14:43 2025

@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
from typing import Dict, List

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

        # En-tête
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Unicorn Throwdown 2025", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(10)

        # Colonnes
        col_widths = [30, 10, 15, 50, 25, 40]
        headers = ["Heure", "Lane", "WOD", "Athlète", "Division", "Emplacement"]

        # En-tête tableau
        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, border=1, align='C', fill=True)
        pdf.ln()

        # Lignes
        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])

            start_time = c['start'] if isinstance(c['start'], str) else c['start'].strftime('%H:%M')
            end_time = c['end'] if isinstance(c['end'], str) else c['end'].strftime('%H:%M')

            data = [
                f"{start_time} - {end_time}",
                str(c['lane']),
                c['wod'],
                c['athlete'],
                c['division'],
                c['location']
            ]

            # Calcul hauteur max de ligne
            heights = []
            for value, width in zip(data, col_widths):
                lines = pdf.multi_cell(width, 5, str(value), border=0, align='C', split_only=True)
                heights.append(len(lines) * 5)
            max_height = max(heights)

            # Dessin des cellules
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
    from collections import defaultdict

    print("✅ Appel de generate_heat_pdf() ✔️")

    # Regrouper les créneaux par heat
    heat_map = defaultdict(lambda: defaultdict(str))  # (start, end, wod, location) -> {lane: juge}

    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (str(c['start']), str(c['end']), c['wod'], c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = list(heat_map.items())
    heats.sort(key=lambda x: x[0][0])  # tri par heure de début

    for i in range(0, len(heats), 2):  # 2 heats par page
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
            judges = pd.read_csv(judges_file, header=None)[0].dropna().tolist()

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

                pdf = generate_pdf_tableau({k: v for k, v in planning.items() if v})

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            "📥 Télécharger tous les plannings",
                            data=f,
                            file_name="plannings_juges.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(tmp.name)

                st.success("PDF généré avec succès!")
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
