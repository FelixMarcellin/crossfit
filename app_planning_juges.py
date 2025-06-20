# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:29:06 2025

@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import math
from collections import defaultdict
import re
import base64

# Configuration de la page
st.set_page_config(page_title="Planificateur Juges CrossFit", layout="wide")
st.title("📋 Planificateur de Juges pour Compétition CrossFit")

# Fonctions de traitement (conservées depuis votre code)
def parser_fichier_competition(contenu):
    heats = []
    current_wod = None
    current_heat = None
    current_time = None
    
    for line in contenu.split('\n'):
        line = line.strip()
        if "WOD" in line and "FINALE" in line:
            wod_match = re.search(r'WOD (\d+)', line)
            if wod_match: current_wod = f"WOD {int(wod_match.group(1))}"
            continue
        # [...] (le reste de votre fonction parser_fichier_competition adapté)

def generer_pdf(judge_assignments):
    pdf = FPDF()
    # [...] (votre code de génération PDF)
    return pdf.output(dest='S').encode('latin1')

# Interface utilisateur
with st.sidebar:
    st.header("1. Importation des fichiers")
    fichier_compet = st.file_uploader("📄 Fichier de compétition (CSV)", type="csv")
    fichier_juges = st.file_uploader("👥 Liste des juges (CSV)", type="csv")
    
    st.header("2. Paramètres")
    tri_chrono = st.checkbox("Trier par ordre chronologique (WOD 1 → 2 → 3)", True)

# Traitement lorsque les fichiers sont uploadés
if fichier_compet and fichier_juges:
    try:
        # Chargement des données
        juges = pd.read_csv(fichier_juges, header=None)[0].tolist()
        contenu_compet = fichier_compet.getvalue().decode('utf-8')
        heats = parser_fichier_competition(contenu_compet)
        
        # Répartition des juges
        judge_assignments = defaultdict(list)
        judge_index = 0
        for heat in sorted(heats, key=lambda x: (int(x['WOD'].split()[1]), x['Time']) if tri_chrono else x['Time']):
            judge = juges[judge_index]
            judge_assignments[judge].append(heat)
            judge_index = (judge_index + 1) % len(juges)
        
        # Génération du PDF
        pdf_bytes = generer_pdf(judge_assignments)
        
        # Téléchargement
        st.success("✅ Planning généré avec succès !")
        st.download_button(
            label="⬇️ Télécharger le PDF",
            data=pdf_bytes,
            file_name="Planning_Juges.pdf",
            mime="application/pdf"
        )
        
        # Aperçu des données
        with st.expander("👀 Aperçu des assignations"):
            st.write(f"Total heats : {len(heats)} | Nombre de juges : {len(juges)}")
            for juge, assignments in judge_assignments.items():
                st.write(f"**{juge}** : {len(assignments)} assignations")
    
    except Exception as e:
        st.error(f"Erreur : {str(e)}")

else:
    st.info("ℹ️ Veuillez uploader les fichiers requis dans la barre latérale")