# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:29:06 2025

@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF  # Utilise fpdf2 en interne
import base64
import re
from collections import defaultdict

def main():
    st.title("📋 Planificateur de Juges CrossFit")
    
    # Upload des fichiers
    with st.sidebar:
        st.header("Importation des fichiers")
        file_compet = st.file_uploader("Fichier compétition (CSV)", type="csv")
        file_judges = st.file_uploader("Liste des juges (CSV)", type="csv")
    
    if file_compet and file_judges:
        try:
            # Lecture des fichiers
            juges = pd.read_csv(file_judges, header=None)[0].tolist()
            compet_data = file_compet.read().decode('utf-8')
            
            # Parser le fichier (exemple simplifié)
            heats = []
            for line in compet_data.split('\n'):
                if "WOD" in line:
                    current_wod = line.split("WOD")[1].split()[0]
                elif line.startswith(('1,', '2,', '3,')) and "EMPTY" not in line:
                    parts = line.split(',')
                    heats.append({
                        'WOD': f"WOD {current_wod}",
                        'Athlete': parts[1].strip(),
                        'Lane': parts[0].strip()
                    })
            
            # Répartition
            judge_assign = {juge: [] for juge in juges}
            for i, heat in enumerate(heats):
                judge_assign[juges[i % len(juges)]].append(heat)
            
            # Génération PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            for juge, assignments in judge_assign.items():
                pdf.cell(0, 10, f"Juge: {juge}", ln=1)
                for assign in assignments:
                    pdf.cell(0, 10, f"{assign['WOD']} - Lane {assign['Lane']}: {assign['Athlete']}", ln=1)
                pdf.ln(5)
            
            # Téléchargement
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button(
                "Télécharger le planning",
                pdf_output,
                "planning_juges.pdf",
                "application/pdf"
            )
            
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

if __name__ == '__main__':
    main()