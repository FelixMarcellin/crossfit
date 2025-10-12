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
import re
from datetime import datetime

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin")

def extract_heat_number(heat_value):
    """Extrait le numéro du heat depuis différentes formats possibles"""
    if pd.isna(heat_value):
        return 0
    if isinstance(heat_value, (int, float)):
        return int(heat_value)
    
    # Cherche des chiffres dans la chaîne
    numbers = re.findall(r'\d+', str(heat_value))
    if numbers:
        return int(numbers[0])
    return 0

def time_overlap(start1, end1, start2, end2):
    """Vérifie si deux créneaux horaires se chevauchent"""
    return not (end1 <= start2 or end2 <= start1)

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

def is_juge_available(juge, creneau, planning, schedule):
    """Vérifie si un juge est disponible pour un créneau donné"""
    if juge not in planning:
        return True
    
    new_start = creneau['start']
    new_end = creneau['end']
    
    for existing_creneau in planning[juge]:
        existing_start = existing_creneau['start']
        existing_end = existing_creneau['end']
        
        # Vérifier le chevauchement horaire
        if time_overlap(new_start, new_end, existing_start, existing_end):
            return False
    
    return True

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
            help="Chaque juge jugera 1 lane par heat pendant ce nombre de heats avant de passer au juge suivant"
        )

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')

            st.subheader("Aperçu du planning importé")
            st.dataframe(schedule.head())

            # Afficher un échantillon des valeurs de la colonne Heat #
            st.write("Valeurs d'exemple dans la colonne 'Heat #':")
            st.write(schedule['Heat #'].head(10).tolist())

            # Colonnes requises incluant maintenant "Heat #"
            required_columns = ['Workout', 'Lane', 'Competitor', 'Division', 'Workout Location', 'Heat Start Time', 'Heat End Time', 'Heat #']
            if not all(col in schedule.columns for col in required_columns):
                st.error("Erreur: Colonnes manquantes.")
                st.write("Colonnes requises:", required_columns)
                st.write("Colonnes trouvées:", list(schedule.columns))
                return

            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            
            # Extraire les numéros de heat
            schedule['Heat_Number'] = schedule['Heat #'].apply(extract_heat_number)
            
            wods = sorted(schedule['Workout'].unique())

            st.header("Disponibilité des Juges par WOD")
            
            # Initialisation de l'état de sélection
            if 'disponibilites' not in st.session_state:
                st.session_state.disponibilites = {wod: set() for wod in wods}
            
            # Bouton pour sélectionner tous les juges pour tous les WODs
            if st.button("Sélectionner tous les juges pour tous les WODs"):
                for wod in wods:
                    st.session_state.disponibilites[wod] = set(judges)
                st.success("Tous les juges sélectionnés pour tous les WODs!")
                st.rerun()
            
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"WOD: {wod}"):
                        # Bouton pour sélectionner tous les juges pour ce WOD spécifique
                        if st.button(f"Tous pour {wod}", key=f"all_{wod}"):
                            st.session_state.disponibilites[wod] = set(judges)
                            st.rerun()
                        
                        # Récupérer la sélection actuelle depuis session_state
                        current_selection = list(st.session_state.disponibilites[wod])
                        
                        # Widget multiselect
                        selected_judges = st.multiselect(
                            f"Sélection pour {wod}",
                            judges,
                            default=current_selection,
                            key=f"dispo_{wod}"
                        )
                        
                        # Mettre à jour session_state
                        st.session_state.disponibilites[wod] = set(selected_judges)

            if st.button("Générer les plannings"):
                planning = {juge: [] for juge in judges}
                
                # Traitement par WOD
                for wod in wods:
                    juges_dispo = list(st.session_state.disponibilites[wod])
                    if not juges_dispo:
                        st.error(f"Aucun juge pour {wod}!")
                        continue
                    
                    # Filtrer le planning pour le WOD courant
                    wod_schedule = schedule[schedule['Workout'] == wod].copy()
                    
                    # Trier par heat et lane pour assurer l'ordre chronologique
                    wod_schedule = wod_schedule.sort_values(['Heat_Number', 'Lane'])
                    
                    # Grouper par heat
                    heats_groups = list(wod_schedule.groupby('Heat_Number'))
                    
                    juge_index = 0
                    heat_count = 0
                    
                    # Parcourir les heats dans l'ordre
                    for heat_idx, (heat_num, heat_data) in enumerate(heats_groups):
                        # Trier les lanes dans l'ordre
                        lanes_sorted = heat_data.sort_values('Lane')
                        
                        # Pour chaque heat, attribuer 1 lane au juge courant
                        if len(lanes_sorted) > 0:
                            # Prendre la première lane disponible de ce heat
                            row = lanes_sorted.iloc[0]
                            creneau = {
                                'wod': wod,
                                'heat': int(heat_num),
                                'lane': row['Lane'],
                                'athlete': row['Competitor'],
                                'division': row['Division'],
                                'location': row['Workout Location'],
                                'start': row['Heat Start Time'],
                                'end': row['Heat End Time']
                            }
                            
                            juge_courant = juges_dispo[juge_index]
                            
                            # Vérifier si le juge est disponible pour ce créneau
                            if is_juge_available(juge_courant, creneau, planning, schedule):
                                planning[juge_courant].append(creneau)
                                heat_count += 1
                                
                                # Changer de juge après X heats consécutifs
                                if heat_count >= heats_consecutifs:
                                    juge_index = (juge_index + 1) % len(juges_dispo)
                                    heat_count = 0
                            else:
                                # Si le juge n'est pas disponible, essayer le suivant
                                juge_trouve = False
                                for i in range(len(juges_dispo)):
                                    juge_candidat = juges_dispo[(juge_index + i) % len(juges_dispo)]
                                    if is_juge_available(juge_candidat, creneau, planning, schedule):
                                        planning[juge_candidat].append(creneau)
                                        juge_index = (juge_index + i) % len(juges_dispo)
                                        heat_count = 1
                                        juge_trouve = True
                                        break
                                
                                if not juge_trouve:
                                    st.warning(f"Aucun juge disponible pour le créneau {creneau['start']}-{creneau['end']} (WOD {wod}, Lane {creneau['lane']})")

                # Vérifier les conflits horaires
                st.subheader("Vérification des conflits horaires")
                conflicts_found = False
                for juge, creneaux in planning.items():
                    creneaux_sorted = sorted(creneaux, key=lambda x: x['start'])
                    for i in range(len(creneaux_sorted) - 1):
                        current = creneaux_sorted[i]
                        next_creneau = creneaux_sorted[i + 1]
                        if time_overlap(current['start'], current['end'], next_creneau['start'], next_creneau['end']):
                            st.error(f"CONFLIT: {juge} a des créneaux qui se chevauchent: {current['start']}-{current['end']} et {next_creneau['start']}-{next_creneau['end']}")
                            conflicts_found = True
                
                if not conflicts_found:
                    st.success("✅ Aucun conflit horaire détecté")

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
                
                # Afficher le récapitulatif des affectations
                st.header("Récapitulatif des affectations")
                recap_data = []
                for juge, creneaux in planning.items():
                    if creneaux:
                        recap_data.append({
                            'Juge': juge,
                            'Total créneaux': len(creneaux),
                            'WODs différents': len({c['wod'] for c in creneaux})
                        })
                        with st.expander(f"Juge: {juge} ({len(creneaux)} créneaux)"):
                            df_affectations = pd.DataFrame(creneaux)
                            # Trier par WOD, heat et lane pour une meilleure lisibilité
                            df_affectations = df_affectations.sort_values(['wod', 'heat', 'lane'])
                            st.table(df_affectations)
                
                if recap_data:
                    df_recap = pd.DataFrame(recap_data)
                    st.subheader("Synthèse de la répartition")
                    st.dataframe(df_recap)
                    
                    # Vérifier l'équilibre
                    min_creneaux = df_recap['Total créneaux'].min()
                    max_creneaux = df_recap['Total créneaux'].max()
                    difference = max_creneaux - min_creneaux
                    
                    if difference <= 2:
                        st.success(f"✅ Répartition équilibrée! Écart maximum: {difference} créneau(x)")
                    else:
                        st.warning(f"⚠️ Répartition déséquilibrée. Écart maximum: {difference} créneaux")

        except Exception as e:
            st.error("Erreur lors du traitement:")
            st.code(traceback.format_exc())
    else:
        st.info("Veuillez uploader le fichier de planning et saisir les juges pour commencer")

if __name__ == "__main__":
    main()