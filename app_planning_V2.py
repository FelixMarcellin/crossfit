# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:47:51 2025

@author: felima
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:47:51 2025

@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
from typing import Dict, List

# Configuration de l'application
st.set_page_config(page_title="Planning Juges CrossFit", layout="wide")
st.title("🧑‍⚖️ Gestion des Juges - Unicorn Throwdown 2025")


def generate_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    """Génère un PDF avec mise en page tabulaire professionnelle"""
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

        # Définition du tableau
        col_widths = [30, 10, 15, 50, 25, 40]
        headers = ["Heure", "Lane", "WOD", "Athlète", "Division", "Emplacement"]

        # En-tête
        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 10, header, border=1, align='C', fill=True)
        pdf.ln()

        # Contenu du tableau
        pdf.set_font("Arial", size=9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, creneau in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])

            # Formatage des heures
            start = creneau['start']
            end = creneau['end']
            start_time = start if isinstance(start, str) else start.strftime('%H:%M')
            end_time = end if isinstance(end, str) else end.strftime('%H:%M')

            row_data = [
                f"{start_time} - {end_time}",
                str(creneau['lane']),
                creneau['wod'],
                creneau['athlete'],
                creneau['division'],
                creneau['location']
            ]

            # Sauvegarde la position pour multi_cell
            x_start = pdf.get_x()
            y_start = pdf.get_y()
            max_y = y_start

            # Première passe pour mesurer les hauteurs
            heights = []
            for value, width in zip(row_data, col_widths):
                nb_lines = len(pdf.multi_cell(width, 5, str(value), border=0, align='C', split_only=True))
                heights.append(nb_lines * 5)
            max_height = max(heights)

            # Deuxième passe pour dessiner chaque cellule
            for j, (text, width) in enumerate(zip(row_data, col_widths)):
                x = pdf.get_x()
                y = pdf.get_y()
                pdf.multi_cell(width, 5, str(text), border=1, align='C', fill=True)
                pdf.set_xy(x + width, y)
            pdf.ln(max_height)

        # Pied de page
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

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

                pdf = generate_pdf({k: v for k, v in planning.items() if v})

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
