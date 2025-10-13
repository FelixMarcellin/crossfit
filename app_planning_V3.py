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
# ATTRIBUTION ÉQUITABLE DES JUGES - VERSION CORRECTE
# ============================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation):
    planning = {j: [] for j in judges}
    
    # Préparer les données avec les numéros de heat
    schedule_with_heat_nums = schedule.copy()
    schedule_with_heat_nums['heat_num'] = schedule_with_heat_nums['Heat #'].apply(extract_heat_number)
    
    # Pour chaque WOD, organiser les heats
    for wod, group in schedule_with_heat_nums.groupby('Workout'):
        juges_dispo = list(disponibilites[wod])
        if not juges_dispo:
            st.error(f"Aucun juge sélectionné pour {wod}")
            continue

        # Grouper par heat (un heat = plusieurs lignes avec même Heat #)
        heats_by_number = {}
        for heat_num, heat_group in group.groupby('heat_num'):
            heats_by_number[heat_num] = {
                'heat_text': heat_group.iloc[0]['Heat #'],  # Texte original du heat
                'lanes': heat_group.to_dict('records')      # Toutes les lignes de ce heat
            }
        
        # Liste des heats triés
        heat_numbers = sorted(heats_by_number.keys())
        
        if rotation == 2:
            # MODE 2 HEATS CONSÉCUTIFS
            _assign_consecutive_mode(heat_numbers, heats_by_number, juges_dispo, planning, wod)
        else:
            # MODE 1 HEAT MAX
            _assign_single_mode(heat_numbers, heats_by_number, juges_dispo, planning, wod)
    
    # VÉRIFICATION : S'assurer que tous les juges ont au moins une affectation
    _ensure_all_judges_used(planning, judges, schedule_with_heat_nums, disponibilites)
    
    return planning


def _assign_consecutive_mode(heat_numbers, heats_by_number, juges_dispo, planning, wod):
    """Mode 2 heats consécutifs obligatoires"""
    juges_utilises = set()
    
    # Étape 1: Assigner des paires de heats consécutifs
    for i in range(len(heat_numbers) - 1):
        current_heat = heat_numbers[i]
        next_heat = heat_numbers[i + 1]
        
        # Vérifier si c'est une paire consécutive
        if next_heat - current_heat == 1:
            # Trouver un juge disponible pour les 2 heats
            juge_trouve = None
            for juge in juges_dispo:
                if juge not in juges_utilises:
                    # Vérifier que le juge n'est pas déjà assigné à un de ces heats
                    deja_assigné = any(
                        c['heat'] in [heats_by_number[current_heat]['heat_text'], heats_by_number[next_heat]['heat_text']] 
                        for c in planning[juge]
                    )
                    if not deja_assigné:
                        juge_trouve = juge
                        break
            
            if juge_trouve:
                # Assigner UNE SEULE ligne de chaque heat au juge
                # Pour le premier heat
                first_lane = heats_by_number[current_heat]['lanes'][0]
                planning[juge_trouve].append({
                    'wod': wod,
                    'lane': first_lane['Lane'],
                    'athlete': first_lane['Competitor'],
                    'division': first_lane['Division'],
                    'location': first_lane['Workout Location'],
                    'start': first_lane['Heat Start Time'],
                    'end': first_lane['Heat End Time'],
                    'heat': heats_by_number[current_heat]['heat_text']
                })
                
                # Pour le deuxième heat
                second_lane = heats_by_number[next_heat]['lanes'][0]
                planning[juge_trouve].append({
                    'wod': wod,
                    'lane': second_lane['Lane'],
                    'athlete': second_lane['Competitor'],
                    'division': second_lane['Division'],
                    'location': second_lane['Workout Location'],
                    'start': second_lane['Heat Start Time'],
                    'end': second_lane['Heat End Time'],
                    'heat': heats_by_number[next_heat]['heat_text']
                })
                
                juges_utilises.add(juge_trouve)
    
    # Étape 2: Assigner les heats restants
    _assign_remaining_heats(heat_numbers, heats_by_number, juges_dispo, planning, wod, juges_utilises)


def _assign_single_mode(heat_numbers, heats_by_number, juges_dispo, planning, wod):
    """Mode 1 heat max (pas de consécutifs)"""
    juges_utilises_ce_tour = set()
    
    for heat_num in heat_numbers:
        heat_data = heats_by_number[heat_num]
        
        # Trouver un juge disponible qui n'a pas fait le heat précédent
        juge_trouve = None
        for juge in juges_dispo:
            if juge not in juges_utilises_ce_tour:
                # Vérifier que le juge n'a pas fait le heat précédent
                peut_assigner = True
                if heat_num > min(heat_numbers):
                    prev_heat = heat_num - 1
                    if any(c['heat'] == heats_by_number[prev_heat]['heat_text'] for c in planning[juge]):
                        peut_assigner = False
                
                if peut_assigner:
                    juge_trouve = juge
                    break
        
        if not juge_trouve:
            # Si aucun juge trouvé avec la contrainte, prendre n'importe quel juge disponible
            for juge in juges_dispo:
                if juge not in juges_utilises_ce_tour:
                    juge_trouve = juge
                    break
        
        if juge_trouve:
            # Assigner UNE SEULE ligne du heat
            lane_data = heat_data['lanes'][0]
            planning[juge_trouve].append({
                'wod': wod,
                'lane': lane_data['Lane'],
                'athlete': lane_data['Competitor'],
                'division': lane_data['Division'],
                'location': lane_data['Workout Location'],
                'start': lane_data['Heat Start Time'],
                'end': lane_data['Heat End Time'],
                'heat': heat_data['heat_text']
            })
            juges_utilises_ce_tour.add(juge_trouve)


def _assign_remaining_heats(heat_numbers, heats_by_number, juges_dispo, planning, wod, juges_deja_utilises):
    """Assigner les heats non attribués"""
    for heat_num in heat_numbers:
        heat_data = heats_by_number[heat_num]
        heat_text = heat_data['heat_text']
        
        # Vérifier si ce heat est déjà attribué
        heat_deja_attribue = any(
            any(c['heat'] == heat_text for c in planning[j]) 
            for j in juges_dispo
        )
        
        if not heat_deja_attribue:
            # Trouver un juge disponible
            juge_trouve = None
            for juge in juges_dispo:
                if juge not in juges_deja_utilises:
                    juge_trouve = juge
                    break
            
            if not juge_trouve:
                # Si tous les juges sont utilisés, prendre le moins chargé
                juge_trouve = min(juges_dispo, key=lambda j: len(planning[j]))
            
            if juge_trouve:
                # Assigner UNE SEULE ligne du heat
                lane_data = heat_data['lanes'][0]
                planning[juge_trouve].append({
                    'wod': wod,
                    'lane': lane_data['Lane'],
                    'athlete': lane_data['Competitor'],
                    'division': lane_data['Division'],
                    'location': lane_data['Workout Location'],
                    'start': lane_data['Heat Start Time'],
                    'end': lane_data['Heat End Time'],
                    'heat': heat_data['heat_text']
                })
                juges_deja_utilises.add(juge_trouve)


def _ensure_all_judges_used(planning, judges, schedule, disponibilites):
    """S'assurer que tous les juges ont au moins une affectation"""
    juges_sans_affectation = [j for j in judges if not planning[j]]
    
    if juges_sans_affectation:
        st.warning(f"🔧 {len(juges_sans_affectation)} juges sans affectation. Réattribution en cours...")
        
        # Trouver des heats où on peut ajouter des juges
        for juge_sans in juges_sans_affectation:
            # Chercher un WOD où ce juge est disponible et où on peut l'ajouter
            for wod in disponibilites:
                if juge_sans in disponibilites[wod]:
                    # Trouver un heat de ce WOD qui n'a pas ce juge
                    wod_heats = schedule[schedule['Workout'] == wod]
                    for _, row in wod_heats.iterrows():
                        heat_text = row['Heat #']
                        
                        # Vérifier si ce juge est déjà sur ce heat
                        deja_sur_heat = any(
                            c['heat'] == heat_text and c['wod'] == wod 
                            for c in planning[juge_sans]
                        )
                        
                        if not deja_sur_heat:
                            # Ajouter ce juge à ce heat
                            planning[juge_sans].append({
                                'wod': wod,
                                'lane': row['Lane'],
                                'athlete': row['Competitor'],
                                'division': row['Division'],
                                'location': row['Workout Location'],
                                'start': row['Heat Start Time'],
                                'end': row['Heat End Time'],
                                'heat': heat_text
                            })
                            break
                    
                    if planning[juge_sans]:  # Si on a trouvé une affectation
                        break


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
        rotation = st.radio("Mode d'attribution", 
                           options=[1, 2], 
                           index=1,
                           format_func=lambda x: "1 heat consécutif max" if x == 1 else "2 heats consécutifs obligatoires",
                           help="2 heats consécutifs: Les juges font obligatoirement 2 heats de suite quand possible")

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
            
            # Extraire les numéros de heat
            schedule['heat_num'] = schedule['Heat #'].apply(extract_heat_number)
            st.write("Numéros de heat extraits:", sorted(schedule['heat_num'].unique()))
            
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

                # ANALYSE FINALE
                st.subheader("📊 Analyse finale des assignations")
                
                # Vérifications
                total_par_juge = {j: len(creneaux) for j, creneaux in planning.items()}
                juges_utilises = sum(1 for count in total_par_juge.values() if count > 0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Assignations par juge:**")
                    for juge, count in sorted(total_par_juge.items(), key=lambda x: x[1], reverse=True):
                        statut = "✅" if count > 0 else "❌"
                        st.write(f"{statut} {juge}: {count} créneaux")
                
                with col2:
                    st.write("**Statistiques:**")
                    st.write(f"Juges utilisés: {juges_utilises}/{len(judges)}")
                    st.write(f"Heats totaux: {sum(total_par_juge.values())}")
                    
                    # Vérifier les séquences consécutives
                    if rotation == 2:
                        st.write("**Séquences consécutives:**")
                        for juge, creneaux in planning.items():
                            if creneaux:
                                sequences = []
                                creneaux_tries = sorted(creneaux, key=lambda x: (x['wod'], extract_heat_number(x['heat'])))
                                for i in range(len(creneaux_tries) - 1):
                                    if (creneaux_tries[i]['wod'] == creneaux_tries[i+1]['wod'] and 
                                        extract_heat_number(creneaux_tries[i+1]['heat']) - extract_heat_number(creneaux_tries[i]['heat']) == 1):
                                        sequences.append(f"{creneaux_tries[i]['heat']}→{creneaux_tries[i+1]['heat']}")
                                if sequences:
                                    st.write(f"{juge}: {', '.join(sequences)}")

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