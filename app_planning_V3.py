# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:05:04 2025

@author: felima
"""

# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version finale : équité, rotation, et affichage "Heat X"
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from typing import Dict, List
from collections import defaultdict
import traceback
import random

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin")


# ============================================================
# PDF TABLEAU PAR JUGE
# ============================================================
def generate_pdf_tableau(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)

    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        # Trier les créneaux du juge par WOD puis heure de début
        def parse_time(x):
            if hasattr(x, 'strftime'):
                return x
            try:
                return pd.to_datetime(x, format='%H:%M')
            except Exception:
                try:
                    return pd.to_datetime(str(x))
                except Exception:
                    return pd.NaT

        creneaux = sorted(creneaux, key=lambda c: (c.get('wod', ''), parse_time(c.get('start', ''))))

        # ====== PAGE PAR JUGE ======
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Nom de la compétition", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(10)

        # Colonnes simplifiées
        col_widths = [35, 15, 25, 15, 60, 35]
        headers = ["Heure", "Lane", "WOD", "Heat #", "Athlete", "Division"]

        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])

            start_time = c.get('start')
            end_time = c.get('end')

            start_str = start_time.strftime('%H:%M') if hasattr(start_time, 'strftime') else str(start_time)
            end_str = end_time.strftime('%H:%M') if hasattr(end_time, 'strftime') else str(end_time)

            data = [
                f"{start_str} - {end_str}",
                str(c.get('lane', '')),
                str(c.get('wod', '')),
                str(c.get('heat', '')),
                str(c.get('athlete', '')),
                str(c.get('division', ''))
            ]

            for val, width in zip(data, col_widths):
                pdf.cell(width, 10, val, border=1, align='C', fill=True)
            pdf.ln()

        pdf.ln(6)
        pdf.set_font("Arial", 'I', 9)
        total_wods = len({c.get('wod', '') for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf


# ============================================================
# PDF PAR HEAT (carte globale)
# ============================================================
def generate_heat_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))

    for juge, creneaux in planning.items():
        for c in creneaux:
            start = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']
            key = (c['wod'], c.get('heat', ''), start, end, c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = sorted(heat_map.items(), key=lambda x: (x[0][0], x[0][1]))

    for i in range(0, len(heats), 2):
        pdf.add_page()
        col_width = 90
        row_height = 8
        spacing = 15

        for j in range(2):
            if i + j >= len(heats):
                break

            (wod, heat, start, end, location), lanes = heats[i + j]
            x_position = 10 + j * (col_width + spacing)

            pdf.set_font("Arial", 'B', 10)
            pdf.set_xy(x_position, 15)
            pdf.cell(col_width, row_height, f"{wod} - {heat}", border=1, align='C', fill=True)
            pdf.set_font("Arial", '', 9)
            pdf.set_xy(x_position, 15 + row_height)
            pdf.cell(col_width, row_height, f"{start} - {end} @ {location}", border=1, align='C')

            pdf.set_xy(x_position, 15 + 2*row_height)
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(col_width/2, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width/2, row_height, "Juge", border=1, align='C', fill=True)

            pdf.set_font("Arial", '', 9)
            for k, lane in enumerate(sorted(lanes)):
                y_position = 15 + (3 + k) * row_height
                pdf.set_xy(x_position, y_position)
                pdf.cell(col_width/2, row_height, str(lane), border=1, align='C')
                pdf.cell(col_width/2, row_height, lanes[lane], border=1, align='C')

    return pdf


# ============================================================
# EXTRACTION DU NUMÉRO DE HEAT
# ============================================================
def extract_heat_number(heat_str):
    """Extrait le numéro du heat d'une string comme 'Heat 4'"""
    if pd.isna(heat_str):
        return 0
    if isinstance(heat_str, (int, float)):
        return int(heat_str)
    if isinstance(heat_str, str):
        try:
            # Essaye d'extraire le nombre de "Heat X"
            import re
            numbers = re.findall(r'\d+', str(heat_str))
            if numbers:
                return int(numbers[0])
        except:
            pass
    return 0


# ============================================================
# ATTRIBUTION ÉQUITABLE DES JUGES - VERSION CORRIGÉE AVEC ROTATION
# ============================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation):
    planning = {j: [] for j in judges}

    # Préparer les données avec les numéros de heat
    schedule_with_heat_nums = schedule.copy()
    schedule_with_heat_nums['heat_num'] = schedule_with_heat_nums['Heat #'].apply(extract_heat_number)
    
    # Créer un mapping heat_text -> heat_num pour chaque WOD
    heat_mapping = {}
    for wod, group in schedule_with_heat_nums.groupby('Workout'):
        heat_mapping[wod] = {}
        for _, row in group.iterrows():
            heat_mapping[wod][row['Heat #']] = row['heat_num']

    for wod, group in schedule_with_heat_nums.groupby('Workout'):
        juges_dispo = list(disponibilites[wod])
        if not juges_dispo:
            st.error(f"Aucun juge sélectionné pour {wod}")
            continue

        # Trier par numéro de heat et lane
        group = group.sort_values(['heat_num', 'Lane'])
        
        # Initialiser les stats des juges
        juge_stats = {j: {
            'heats': 0, 
            'last_heat_num': -1, 
            'consecutive_count': 0,
            'assigned_heats': set()
        } for j in juges_dispo}

        # Traiter les heats dans l'ordre
        current_heat_num = -1
        current_heat_group = []
        
        for idx, row in group.iterrows():
            heat_text = row['Heat #']
            heat_num = row['heat_num']
            
            # Si on change de heat num, traiter le groupe précédent
            if heat_num != current_heat_num and current_heat_group:
                # Attribuer les juges pour le heat précédent
                _assign_judes_to_heat_group(current_heat_group, juge_stats, juges_dispo, rotation, planning)
                current_heat_group = []
            
            current_heat_num = heat_num
            current_heat_group.append((idx, row))
        
        # Traiter le dernier groupe
        if current_heat_group:
            _assign_judes_to_heat_group(current_heat_group, juge_stats, juges_dispo, rotation, planning)

    return planning


def _assign_judes_to_heat_group(heat_group, juge_stats, juges_dispo, rotation, planning):
    """Attribue les juges pour un groupe de lignes du même heat"""
    # Trier les juges par nombre d'assignations (pour l'équité)
    juges_tries = sorted(juges_dispo, key=lambda j: juge_stats[j]['heats'])
    
    assigned_judges = set()
    
    for idx, row in heat_group:
        heat_text = row['Heat #']
        heat_num = row['heat_num']
        
        # Trouver les juges disponibles pour ce heat
        available_judges = []
        
        for juge in juges_tries:
            if juge in assigned_judges:
                continue
                
            stats = juge_stats[juge]
            
            # Vérifier si le juge peut prendre ce heat selon la rotation
            can_assign = True
            
            if rotation == 1:
                # Rotation 1: pas de heats consécutifs
                if stats['last_heat_num'] != -1 and heat_num - stats['last_heat_num'] == 1:
                    can_assign = False
            elif rotation == 2:
                # Rotation 2: max 2 heats consécutifs, puis 2 heats de repos
                if stats['last_heat_num'] != -1:
                    gap = heat_num - stats['last_heat_num']
                    if gap == 1:  # Heat consécutif
                        if stats['consecutive_count'] >= 2:
                            can_assign = False
                    elif gap == 2:  # 1 heat de repos
                        if stats['consecutive_count'] >= 2:
                            can_assign = False
                    # Pour gap >= 3, toujours autorisé
            
            if can_assign:
                available_judges.append(juge)
        
        # Si pas de juges disponibles avec les contraintes, relâcher les contraintes
        if not available_judges:
            available_judges = [j for j in juges_tries if j not in assigned_judges]
        
        # Si toujours pas, prendre n'importe quel juge
        if not available_judges:
            available_judges = juges_dispo.copy()
        
        # Choisir le juge avec le moins d'assignations
        available_judges.sort(key=lambda j: juge_stats[j]['heats'])
        juge_choisi = available_judges[0]
        
        # Mettre à jour les stats
        stats = juge_stats[juge_choisi]
        if stats['last_heat_num'] != -1 and heat_num - stats['last_heat_num'] == 1:
            stats['consecutive_count'] += 1
        else:
            stats['consecutive_count'] = 1
            
        stats['last_heat_num'] = heat_num
        stats['heats'] += 1
        stats['assigned_heats'].add(heat_num)
        
        assigned_judges.add(juge_choisi)
        
        # Ajouter au planning
        planning[juge_choisi].append({
            'wod': row['Workout'],
            'lane': row.get('Lane', ''),
            'athlete': row.get('Competitor', ''),
            'division': row.get('Division', ''),
            'location': row.get('Workout Location', ''),
            'start': row.get('Heat Start Time', ''),
            'end': row.get('Heat End Time', ''),
            'heat': heat_text
        })


# ============================================================
# MAIN STREAMLIT
# ============================================================
def main():
    with st.sidebar:
        st.header("Import des fichiers")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])

        st.header("Saisie des juges")
        input_method = st.radio("Méthode de saisie des juges", ["Fichier CSV", "Saisie manuelle"], index=0)
        judges = []
        if input_method == "Fichier CSV":
            judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
            if judges_file:
                judges = pd.read_csv(judges_file, header=None, encoding='latin1')[0].dropna().tolist()
        else:
            judges_text = st.text_area("Saisir les noms des juges (un par ligne)", value="Juge 1\nJuge 2\nJuge 3", height=150)
            judges = [j.strip() for j in judges_text.split('\n') if j.strip()]
            if judges:
                st.write("Juges saisis:")
                st.write(judges)

        st.header("Paramètres de rotation")
        rotation = st.radio("Nombre de heats consécutifs par juge", 
                           options=[1, 2], 
                           index=1,
                           help="1 = pas de heats consécutifs, 2 = max 2 heats consécutifs puis repos")
        
        st.info("""
        **Explication de la rotation:**
        - **1 heat consécutif**: Les juges ne font jamais 2 heats de suite
        - **2 heats consécutifs**: Les juges peuvent faire max 2 heats de suite, puis au moins 2 heats de repos
        """)

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            
            # Vérification des colonnes requises
            required_columns = ['Workout', 'Lane', 'Competitor', 'Division', 'Workout Location',
                                'Heat Start Time', 'Heat End Time', 'Heat #']
            if not all(col in schedule.columns for col in required_columns):
                st.error("Erreur: Colonnes manquantes.")
                st.write("Colonnes trouvées:", list(schedule.columns))
                return

            # Nettoyage des données
            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            
            # Aperçu des données
            st.subheader("Structure des données chargées")
            st.write("Aperçu des données:")
            st.dataframe(schedule[['Workout', 'Heat #', 'Lane', 'Competitor']].head())
            
            # Extraire les numéros de heat pour debug
            schedule['heat_num_debug'] = schedule['Heat #'].apply(extract_heat_number)
            st.write("Numéros de heat extraits:", sorted(schedule['heat_num_debug'].unique()))
            
            wods = sorted(schedule['Workout'].unique())

            st.header("Disponibilité des Juges par WOD")
            disponibilites = {}
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"WOD: {wod}"):
                        select_all = st.checkbox(f"Tout sélectionner pour {wod}", key=f"select_all_{wod}")
                        if select_all:
                            selected = judges
                        else:
                            selected = st.multiselect(f"Juges pour {wod}", judges, key=f"dispo_{wod}")
                        disponibilites[wod] = selected

            if st.button("Générer le planning"):
                # Vérifier que tous les WODs ont au moins un juge
                wods_sans_juge = [wod for wod, j in disponibilites.items() if not j]
                if wods_sans_juge:
                    st.error(f"Les WODs suivants n'ont aucun juge attribué: {', '.join(wods_sans_juge)}")
                    return

                planning = assign_judges_equitable(schedule, judges, disponibilites, rotation)

                # Vérification finale avec analyse de la rotation
                st.subheader("Analyse de la rotation des juges")
                for juge, creneaux in planning.items():
                    if creneaux:
                        # Trier par WOD et numéro de heat
                        creneaux_sorted = sorted(creneaux, key=lambda x: (x['wod'], extract_heat_number(x['heat'])))
                        heats_by_wod = {}
                        for c in creneaux_sorted:
                            wod = c['wod']
                            heat_num = extract_heat_number(c['heat'])
                            if wod not in heats_by_wod:
                                heats_by_wod[wod] = []
                            heats_by_wod[wod].append(heat_num)
                        
                        # Analyser les séquences consécutives
                        consecutive_analysis = []
                        for wod, heat_nums in heats_by_wod.items():
                            if len(heat_nums) > 1:
                                consecutive_count = 1
                                for i in range(1, len(heat_nums)):
                                    if heat_nums[i] - heat_nums[i-1] == 1:
                                        consecutive_count += 1
                                    else:
                                        if consecutive_count > 1:
                                            consecutive_analysis.append(f"{wod}: {consecutive_count} heats consécutifs")
                                        consecutive_count = 1
                                if consecutive_count > 1:
                                    consecutive_analysis.append(f"{wod}: {consecutive_count} heats consécutifs")
                        
                        analysis_text = " | ".join(consecutive_analysis) if consecutive_analysis else "Aucune séquence consécutive"
                        st.write(f"**{juge}** ({len(creneaux)} créneaux): {analysis_text}")

                # Générer les PDFs
                pdf_juges = generate_pdf_tableau({k: v for k, v in planning.items() if v})
                pdf_heats = generate_heat_pdf({k: v for k, v in planning.items() if v})

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
                    pdf_juges.output(tmp1.name)
                    with open(tmp1.name, "rb") as f:
                        st.download_button("📘 Télécharger planning par juge", f, "planning_juges.pdf", "application/pdf")
                    os.unlink(tmp1.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
                    pdf_heats.output(tmp2.name)
                    with open(tmp2.name, "rb") as f:
                        st.download_button("📗 Télécharger planning par heat", f, "planning_heats.pdf", "application/pdf")
                    os.unlink(tmp2.name)

                st.success("✅ Plannings générés avec succès !")

        except Exception as e:
            st.error("Erreur lors du traitement :")
            st.code(traceback.format_exc())
    else:
        st.info("Veuillez uploader le fichier de planning et saisir les juges.")


if __name__ == "__main__":
    main()