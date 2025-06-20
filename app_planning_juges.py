# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:29:06 2025

@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
from collections import defaultdict
import re

def parser_fichier_competition(contenu):
    """Version robuste du parser qui gère les cas limites"""
    heats = []
    current_wod = "WOD 1"  # Valeur par défaut
    current_heat = "Heat 1"
    current_time = "08:00-08:15"
    
    for line in contenu.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Détection WOD
        if "WOD" in line and "FINALE" in line:
            match = re.search(r'WOD (\d+)', line)
            if match:
                current_wod = f"WOD {match.group(1)}"
            continue
            
        # Détection Heat
        if "Heat" in line and "from" in line:
            match = re.search(r'Heat (\d+)', line)
            if match:
                current_heat = f"Heat {match.group(1)}"
            
            match = re.search(r'from (\d+:\d+) to (\d+:\d+)', line)
            if match:
                current_time = f"{match.group(1)}-{match.group(2)}"
            continue
            
        # Lignes de données
        if line and line[0].isdigit():
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 3 and "EMPTY LANE" not in line:
                try:
                    lane = int(parts[0])
                    athlete = parts[1]
                    division = next((p for p in parts if p in ['TEAM INTER', 'TEAM RX', 'RX MEN', 'RX WOMEN']), "INDEPENDENT")
                    
                    heats.append({
                        'WOD': current_wod,
                        'Heat': current_heat,
                        'Time': current_time,
                        'Lane': lane,
                        'Athlete': athlete,
                        'Division': division
                    })
                except (ValueError, IndexError):
                    continue
    return heats

def main():
    st.title("📋 Planificateur de Juges CrossFit")
    
    # Upload des fichiers
    with st.sidebar:
        st.header("1. Importation des fichiers")
        fichier_compet = st.file_uploader("Fichier compétition (CSV)", type=["csv", "txt"])
        fichier_juges = st.file_uploader("Liste des juges (CSV)", type=["csv", "txt"])
        
        st.header("2. Options")
        tri_chrono = st.checkbox("Trier par ordre chronologique", True)

    if fichier_compet and fichier_juges:
        try:
            # Lecture des fichiers
            contenu_compet = fichier_compet.read().decode("utf-8")
            juges = pd.read_csv(fichier_juges, header=None)[0].tolist()
            
            # Validation des données
            if not juges:
                st.error("Le fichier des juges est vide ou mal formaté")
                return
                
            # Parsing
            heats = parser_fichier_competition(contenu_compet)
            if not heats:
                st.error("Aucune donnée valide trouvée dans le fichier de compétition")
                return
            
            # Répartition
            judge_assignments = defaultdict(list)
            judge_index = 0
            
            # Tri selon l'option choisie
            heats_sorted = sorted(heats, key=lambda x: (
                int(x['WOD'].split()[1]), 
                x['Time']
            )) if tri_chrono else heats
            
            for heat in heats_sorted:
                judge = juges[judge_index % len(juges)]
                judge_assignments[judge].append(heat)
                judge_index += 1
            
            # Génération PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            for juge, assignments in judge_assignments.items():
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"Planning du juge: {juge}", 0, 1)
                pdf.ln(10)
                
                # En-têtes
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(30, 10, "WOD", 1)
                pdf.cell(25, 10, "Heat", 1)
                pdf.cell(30, 10, "Horaire", 1)
                pdf.cell(20, 10, "Lane", 1)
                pdf.cell(70, 10, "Athlète", 1)
                pdf.cell(30, 10, "Division", 1)
                pdf.ln()
                
                # Données
                pdf.set_font("Arial", size=10)
                for assign in assignments:
                    pdf.cell(30, 8, assign['WOD'], 1)
                    pdf.cell(25, 8, assign['Heat'], 1)
                    pdf.cell(30, 8, assign['Time'], 1)
                    pdf.cell(20, 8, str(assign['Lane']), 1)
                    pdf.cell(70, 8, assign['Athlete'][:35], 1)  # Tronquer si trop long
                    pdf.cell(30, 8, assign['Division'], 1)
                    pdf.ln()
            
            # Téléchargement
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.success("✅ Planning généré avec succès!")
            st.download_button(
                "⬇️ Télécharger le PDF",
                data=pdf_output,
                file_name="Planning_Juges.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Erreur lors du traitement: {str(e)}")
    else:
        st.info("ℹ️ Veuillez uploader les deux fichiers pour continuer")

if __name__ == "__main__":
    main()