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
from typing import Dict, List, Set, Any

# Configuration de l'application
st.set_page_config(page_title="Planning Juges CrossFit", layout="wide")
st.title("🧑‍⚖️ Gestion des Juges - Unicorn Throwdown 2025")

def generate_pdf(planning: Dict[str, List[Dict[str, Any]]]) -> FPDF:
    """Génère un PDF avec mise en page tabulaire professionnelle"""
    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for juge, creneaux in planning.items():
        pdf.add_page()
        
        # En-tête
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Unicorn Throwdown 2025", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(8)
        
        # Configuration du tableau
        col_widths = [30, 15, 15, 50, 30, 40]  # Largeurs des colonnes (mm)
        headers = ["Heure", "Lane", "WOD", "Athlète", "Division", "Emplacement"]
        
        # Couleurs
        header_color = (211, 211, 211)  # Gris clair pour l'en-tête
        row_colors = [(255, 255, 255), (240, 240, 240)]  # Alternance de couleurs
        
        # En-tête du tableau
        pdf.set_fill_color(*header_color)
        pdf.set_font("Arial", 'B', 10)
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 10, header, 1, 0, 'C', True)
        pdf.ln()
        
        # Contenu du tableau
        pdf.set_font("Arial", size=9)
        for i, creneau in enumerate(creneaux):
            # Alternance des couleurs de ligne
            pdf.set_fill_color(*row_colors[i % 2])
            
            # Formatage des données
            start_time = creneau['start'] if isinstance(creneau['start'], str) else creneau['start'].strftime('%H:%M')
            end_time = creneau['end'] if isinstance(creneau['end'], str) else creneau['end'].strftime('%H:%M')
            
            cells = [
                f"{start_time} - {end_time}",
                str(creneau['lane']),
                creneau['wod'],
                creneau['athlete'],
                creneau['division'],
                creneau['location']
            ]
            
            for width, cell in zip(col_widths, cells):
                pdf.cell(width, 8, str(cell), 1, 0, 'C', True)
            pdf.ln()
        
        # Pied de page
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)
        
        # Ligne décorative
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    
    return pdf

def process_data(schedule: pd.DataFrame, judges: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Traite les données et crée le planning"""
    # Nettoyage des données
    schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=True)]
    schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
    
    # Conversion des heures si nécessaire
    if not isinstance(schedule.iloc[0]['Heat Start Time'], str):
        schedule['Heat Start Time'] = schedule['Heat Start Time'].dt.strftime('%H:%M')
        schedule['Heat End Time'] = schedule['Heat End Time'].dt.strftime('%H:%M')
    
    # Détection des WODs
    wods = sorted(schedule['Workout'].unique())
    
    # Interface utilisateur
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
    planning = {juge: [] for juge in judges}
    if st.button("✨ Générer les plannings"):
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
        st.success("Planning généré avec succès!")
        st.header("📊 Récapitulatif des affectations")
        
        for juge, creneaux in planning.items():
            if creneaux:
                with st.expander(f"Juge: {juge} ({len(creneaux)} créneaux)"):
                    st.table(pd.DataFrame(creneaux))
    
    return planning

# Chargement des fichiers
with st.sidebar:
    st.header("📤 Import des fichiers")
    schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
    judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])

if schedule_file and judges_file:
    try:
        # Lecture des fichiers
        schedule = pd.read_excel(schedule_file, engine='openpyxl')
        judges = pd.read_csv(judges_file, header=None)[0].dropna().tolist()
        
        # Traitement des données
        process_data(schedule, judges)
    
    except Exception as e:
        st.error(f"Erreur lors du traitement des fichiers: {str(e)}")
else:
    st.info("Veuillez uploader les fichiers pour commencer")