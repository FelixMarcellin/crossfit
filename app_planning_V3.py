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

        col_widths = [30, 10, 15, 15, 50, 25, 40]
        headers = ["Heure", "Heat", "Lane", "WOD", "Athlete", "Division", "Emplacement"]

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
                str(c['heat']),
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
        pdf.cell(0, 8, f"Total: {len(creneaux)} creneaux sur {total_wods} WODs", 0, 1)

    return pdf

def generate_heat_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))

    for juge, creneaux in planning.items():
        for c in creneaux:
            start = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']
            key = (start, end, c['wod'], c['location'], c['heat'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = sorted(heat_map.items(), key=lambda x: (x[0][0], x[0][4]))  # Tri par heure et numéro de heat

    for i in range(0, len(heats), 2):
        pdf.add_page()
        
        # Configuration du tableau
        col_width = 90
        row_height = 8
        spacing = 15
        
        for j in range(2):
            if i + j >= len(heats):
                break

            (start, end, wod, location, heat_num), lanes = heats[i + j]
            
            # Position X pour le tableau (gauche ou droite)
            x_position = 10 + j * (col_width + spacing)
            
            # En-tête du tableau
            pdf.set_font("Arial", 'B', 10)
            pdf.set_xy(x_position, 15)
            pdf.cell(col_width, row_height, f"HEAT {heat_num}: {start} - {end}", border=1, align='C', fill=True)
            
            pdf.set_font("Arial", '', 9)
            pdf.set_xy(x_position, 15 + row_height)
            pdf.cell(col_width, row_height, f"WOD: {wod}", border=1, align='C')
            
            pdf.set_xy(x_position, 15 + 2*row_height)
            pdf.cell(col_width, row_height, f"Location: {location}", border=1, align='C')
            
            # En-tête des colonnes
            pdf.set_font("Arial", 'B', 9)
            pdf.set_xy(x_position, 15 + 3*row_height)
            pdf.cell(col_width/2, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width/2, row_height, "Juge", border=1, align='C', fill=True)
            
            # Contenu du tableau
            pdf.set_font("Arial", '', 9)
            for k, lane in enumerate(sorted(lanes)):
                y_position = 15 + (4 + k) * row_height
                pdf.set_xy(x_position, y_position)
                pdf.cell(col_width/2, row_height, str(lane), border=1, align='C')
                pdf.cell(col_width/2, row_height, lanes[lane], border=1, align='C')

    return pdf

def main():
    with st.sidebar:
        st.header("Import des fichiers")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
        
        # Nouvelle section pour le choix de la méthode de saisie des juges
        st.header("Saisie des juges")
        input_method = st.radio(
            "Méthode de saisie des juges",
            options=["Fichier CSV", "Saisie manuelle"],
            index=0
        )
        
        judges = []
        if input_method == "Fichier CSV":
            judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
            if judges_file:
                judges = pd.read_csv(judges_file, header=None, encoding='latin1')[0].dropna().tolist()
        else:
            # Version corrigée pour la saisie manuelle
            judges_text = st.text_area(
                "Saisir les noms des juges (un par ligne)",
                value="Juge 1\nJuge 2\nJuge 3",  # Valeur par défaut pour l'exemple
                height=150,
                help="Entrez un nom de juge par ligne"
            )
            judges = [j.strip() for j in judges_text.split('\n') if j.strip()]
            
            # Afficher un aperçu des juges saisis
            if judges:
                st.write("Juges saisis:")
                st.write(judges)
        
        # Paramètre pour le nombre de heats consécutifs
        st.header("Paramètres d'affectation")
        heats_consecutifs = st.number_input(
            "Nombre de heats consécutifs par juge avant rotation",
            min_value=1,
            max_value=10,
            value=2,
            help="Chaque juge jugera ce nombre de heats consécutifs avant de passer au juge suivant"
        )

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')

            st.subheader("Aperçu du planning importé")
            st.dataframe(schedule.head())

            # Colonnes requises incluant maintenant "Heat #"
            required_columns = ['Workout', 'Lane', 'Competitor', 'Division', 'Workout Location', 'Heat Start Time', 'Heat End Time', 'Heat #']
            if not all(col in schedule.columns for col in required_columns):
                st.error("Erreur: Colonnes manquantes.")
                st.write("Colonnes requises:", required_columns)
                st.write("Colonnes trouvées:", list(schedule.columns))
                return

            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            wods = sorted(schedule['Workout'].unique())

            st.header("Disponibilité des Juges par WOD")
            disponibilites = {wod: set() for wod in wods}
            
            # Bouton pour sélectionner tous les juges pour tous les WODs
            if st.button("Sélectionner tous les juges pour tous les WODs"):
                for wod in wods:
                    disponibilites[wod] = set(judges)
                st.success("Tous les juges sélectionnés pour tous les WODs!")
            
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"WOD: {wod}"):
                        # Bouton pour sélectionner tous les juges pour ce WOD spécifique
                        if st.button(f"Tous pour {wod}", key=f"all_{wod}"):
                            disponibilites[wod] = set(judges)
                            st.rerun()
                        
                        disponibilites[wod] = set(st.multiselect(
                            f"Sélection pour {wod}",
                            judges,
                            default=list(disponibilites[wod]),
                            key=f"dispo_{wod}"
                        ))

            if st.button("Générer les plannings"):
                planning = {juge: [] for juge in judges}
                
                # Traitement par WOD
                for wod in wods:
                    juges_dispo = list(disponibilites[wod])
                    if not juges_dispo:
                        st.error(f"Aucun juge pour {wod}!")
                        continue
                    
                    # Filtrer le planning pour le WOD courant
                    wod_schedule = schedule[schedule['Workout'] == wod].copy()
                    
                    # Trier par heat et lane pour assurer l'ordre chronologique
                    wod_schedule = wod_schedule.sort_values(['Heat #', 'Lane'])
                    
                    # Grouper par heat
                    heats = wod_schedule.groupby('Heat #')
                    
                    juge_index = 0
                    heat_count = 0
                    
                    for heat_num, heat_data in heats:
                        # Attribuer le même juge pour X heats consécutifs
                        juge_attribue = juges_dispo[juge_index]
                        
                        for _, row in heat_data.iterrows():
                            planning[juge_attribue].append({
                                'wod': wod,
                                'heat': int(row['Heat #']),
                                'lane': row['Lane'],
                                'athlete': row['Competitor'],
                                'division': row['Division'],
                                'location': row['Workout Location'],
                                'start': row['Heat Start Time'],
                                'end': row['Heat End Time']
                            })
                        
                        heat_count += 1
                        
                        # Changer de juge après X heats consécutifs
                        if heat_count >= heats_consecutifs:
                            juge_index = (juge_index + 1) % len(juges_dispo)
                            heat_count = 0

                # Génération des PDF
                pdf_juges = generate_pdf_tableau({k: v for k, v in planning.items() if v})
                pdf_heats = generate_heat_pdf({k: v for k, v in planning.items() if v})

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_juges:
                    pdf_juges.output(tmp_juges.name)
                    with open(tmp_juges.name, "rb") as f:
                        st.download_button(
                            "Télécharger planning par juge",
                            data=f,
                            file_name="planning_juges.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(tmp_juges.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_heats:
                    pdf_heats.output(tmp_heats.name)
                    with open(tmp_heats.name, "rb") as f:
                        st.download_button(
                            "Télécharger planning par heat",
                            data=f,
                            file_name="planning_heats.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(tmp_heats.name)

                st.success("PDF générés avec succès!")
                st.header("Récapitulatif des affectations")
                for juge, creneaux in planning.items():
                    if creneaux:
                        with st.expander(f"Juge: {juge} ({len(creneaux)} créneaux)"):
                            df_affectations = pd.DataFrame(creneaux)
                            # Trier par WOD, heat et lane pour une meilleure lisibilité
                            df_affectations = df_affectations.sort_values(['wod', 'heat', 'lane'])
                            st.table(df_affectations)

        except Exception as e:
            st.error("Erreur lors du traitement:")
            st.code(traceback.format_exc())
    else:
        st.info("Veuillez uploader le fichier de planning et saisir les juges pour commencer")

if __name__ == "__main__":
    main()