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
import chardet
from typing import Dict, List, Set

# Configuration de l'application
st.set_page_config(page_title="Planning Juges CrossFit", layout="wide")
st.title("🧑‍⚖️ Gestion des Juges - Unicorn Throwdown 2025")

def detect_encoding(file):
    """Détecte l'encodage d'un fichier"""
    rawdata = file.read()
    file.seek(0)  # On remet le curseur au début du fichier
    result = chardet.detect(rawdata)
    return result['encoding']

# Fonction pour générer les PDF
def generate_pdf(planning: Dict[str, List[Dict]]):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for juge, creneaux in planning.items():
        pdf.add_page()
        
        # En-tête
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Planning de {juge}", 0, 1, 'C')
        pdf.ln(10)
        
        # Contenu
        pdf.set_font("Arial", size=12)
        for creneau in creneaux:
            pdf.cell(0, 10, f"WOD: {creneau['wod']}", 0, 1)
            pdf.cell(0, 10, f"Lane {creneau['lane']} - {creneau['athlete']}", 0, 1)
            pdf.cell(0, 10, f"Heure: {creneau['start']} à {creneau['end']}", 0, 1)
            pdf.ln(5)
    
    return pdf

# Chargement des données
with st.sidebar:
    st.header("📤 Import des fichiers")
    schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
    judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])

if schedule_file and judges_file:
    try:
        # Lecture du planning Excel
        schedule = pd.read_excel(schedule_file, engine='openpyxl')
        
        # Détection et lecture du fichier CSV avec le bon encodage
        judges_file.seek(0)
        encoding = detect_encoding(judges_file)
        st.info(f"Encodage détecté: {encoding}")
        
        juges_disponibles = pd.read_csv(judges_file, header=None, encoding=encoding)[0].tolist()
        juges_disponibles = [j.strip() for j in juges_disponibles if j and str(j).strip()]
        
        # Nettoyage des données
        schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=True)]
        schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
        
        # Détection des WODs
        wods = sorted(schedule['Workout'].unique())
        
        # Dictionnaire pour stocker les disponibilités
        disponibilites = {wod: set() for wod in wods}
        
        # Interface de sélection
        st.header("📝 Disponibilité des Juges par WOD")
        
        cols = st.columns(3)
        for i, wod in enumerate(wods):
            with cols[i % 3]:
                with st.expander(f"WOD: {wod}"):
                    disponibilites[wod] = set(st.multiselect(
                        f"Sélectionnez les juges disponibles",
                        juges_disponibles,
                        key=f"dispo_{wod}"
                    ))
        
        # Vérification des conflits et génération du planning
        if st.button("✨ Générer les plannings"):
            planning_final = {juge: [] for juge in juges_disponibles}
            probleme = False
            
            # Assignation des juges
            for _, creneau in schedule.iterrows():
                wod = creneau['Workout']
                juges_dispo = disponibilites[wod]
                
                if not juges_dispo:
                    st.error(f"Aucun juge disponible pour le {wod}!")
                    probleme = True
                    continue
                
                # Trouver le juge le moins occupé
                juge_attribue = min(juges_dispo, key=lambda j: len(planning_final[j]))
                
                planning_final[juge_attribue].append({
                    'wod': wod,
                    'lane': creneau['Lane'],
                    'athlete': creneau['Competitor'],
                    'start': creneau['Heat Start Time'],
                    'end': creneau['Heat End Time']
                })
            
            if not probleme:
                # Affichage du récapitulatif
                st.success("Planning généré avec succès!")
                st.header("📊 Récapitulatif des affectations")
                
                for juge, creneaux in planning_final.items():
                    with st.expander(f"Juge: {juge} ({len(creneaux)} créneaux)"):
                        st.table(pd.DataFrame(creneaux))
                
                # Génération du PDF
                pdf = generate_pdf(planning_final)
                
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
    
    except Exception as e:
        st.error(f"Erreur lors du traitement des fichiers: {str(e)}")
else:
    st.info("Veuillez uploader les fichiers pour commencer")