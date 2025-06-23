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
    from fpdf import FPDF

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        pdf.add_page()

        # Titre
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Unicorn Throwdown 2025", ln=1, align='C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", ln=1, align='C')
        pdf.ln(5)

        # En-tête tableau
        headers = ["Heure", "Lane", "WOD", "Athlète", "Division", "Emplacement"]
        col_widths = [30, 10, 15, 50, 25, 40]
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(200, 200, 200)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 8, header, border=1, align='C', fill=True)
        pdf.ln()

        # Données
        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            fill = row_colors[i % 2]
            pdf.set_fill_color(*fill)

            start = c['start']
            end = c['end']
            heure = f"{start if isinstance(start, str) else start.strftime('%H:%M')} - {end if isinstance(end, str) else end.strftime('%H:%M')}"
            values = [
                heure,
                str(c['lane']),
                c['wod'],
                c['athlete'],
                c['division'],
                c['location']
            ]

            # Calcul hauteur max de la ligne
            heights = []
            for val, w in zip(values, col_widths):
                lines = pdf.multi_cell(w, 5, str(val), border=0, align='C', split_only=True)
                heights.append(5 * len(lines))
            max_h = max(heights)

            # Écriture cellule par cellule avec multi_cell
            x = pdf.get_x()
            y = pdf.get_y()
            for val, w in zip(values, col_widths):
                pdf.set_xy(x, y)
                pdf.multi_cell(w, 5, str(val), border=1, align='C', fill=True)
                x += w
            pdf.ln(max_h)

        # Résumé
        pdf.ln(4)
        pdf.set_font("Arial", 'I', 10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", ln=1)

    return pdf



def main():
    with st.sidebar:
        st.header("📤 Import des fichiers")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
        judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
    
    if schedule_file and judges_file:
        try:
            # Lecture des fichiers
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            judges = pd.read_csv(judges_file, header=None)[0].dropna().tolist()
            
            # Nettoyage des données
            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=True)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            
            # Détection des WODs
            wods = sorted(schedule['Workout'].unique())
            
            # Interface de sélection
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
            
            # Génération du planning
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
                
                # Génération du PDF
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
                
                # Affichage du récapitulatif
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