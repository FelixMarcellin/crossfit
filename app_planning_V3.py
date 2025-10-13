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
# ATTRIBUTION ÉQUITABLE DES JUGES - VERSION AMÉLIORÉE
# ============================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation):
    planning = {j: [] for j in judges}
    
    # Préparer les données avec les numéros de heat
    schedule_with_heat_nums = schedule.copy()
    schedule_with_heat_nums['heat_num'] = schedule_with_heat_nums['Heat #'].apply(extract_heat_number)
    
    # Filtrer les lignes vides
    schedule_with_heat_nums = schedule_with_heat_nums[
        ~schedule_with_heat_nums['Competitor'].str.contains('EMPTY LANE', na=False) &
        ~schedule_with_heat_nums['Competitor'].isna()
    ]
    
    # Calculer le nombre total de lignes à assigner
    total_lignes = len(schedule_with_heat_nums)
    cible_par_juge = total_lignes // len(judges)
    tolerance = 1  # ±1 ligne
    
    st.write(f"🎯 Cible: {cible_par_juge} lignes par juge (tolérance: ±{tolerance})")
    st.write(f"📊 Total lignes à assigner: {total_lignes}")
    
    if rotation == 2:
        # MODE AVEC BLOCS DE 2 HEATS CONSÉCUTIFS
        _assign_with_consecutive_priority(schedule_with_heat_nums, judges, disponibilites, planning, cible_par_juge)
    else:
        # MODE SANS CONSÉCUTIFS
        _assign_without_consecutive(schedule_with_heat_nums, judges, disponibilites, planning)
    
    # Équilibrer les assignations
    _balance_assignments(planning, judges, cible_par_juge, tolerance)
    
    return planning


def _assign_with_consecutive_priority(schedule, judges, disponibilites, planning, cible_par_juge):
    """Attribution avec priorité absolue aux blocs de 2 heats consécutifs"""
    
    # Étape 1: Identifier toutes les paires de heats consécutifs possibles
    consecutive_pairs = _find_all_consecutive_pairs(schedule)
    
    # Étape 2: Assigner d'abord les paires consécutives
    _assign_consecutive_pairs(consecutive_pairs, judges, disponibilites, planning, cible_par_juge)
    
    # Étape 3: Assigner les lignes restantes
    _assign_remaining_lines(schedule, judges, disponibilites, planning, cible_par_juge)


def _find_all_consecutive_pairs(schedule):
    """Trouve toutes les paires de heats consécutifs dans le planning"""
    consecutive_pairs = []
    
    for wod, group in schedule.groupby('Workout'):
        # Trier par heat num
        group = group.sort_values('heat_num')
        heat_numbers = sorted(group['heat_num'].unique())
        
        # Identifier les paires consécutives
        for i in range(len(heat_numbers) - 1):
            current_heat = heat_numbers[i]
            next_heat = heat_numbers[i + 1]
            
            if next_heat - current_heat == 1:
                # Récupérer les lignes de ces heats
                current_lines = group[group['heat_num'] == current_heat]
                next_lines = group[group['heat_num'] == next_heat]
                
                if len(current_lines) > 0 and len(next_lines) > 0:
                    consecutive_pairs.append({
                        'wod': wod,
                        'heat1': current_heat,
                        'heat2': next_heat,
                        'lines1': current_lines.to_dict('records'),
                        'lines2': next_lines.to_dict('records'),
                        'count': len(current_lines) + len(next_lines)
                    })
    
    # Trier par nombre de lignes (priorité aux paires avec plus de lignes)
    consecutive_pairs.sort(key=lambda x: x['count'], reverse=True)
    return consecutive_pairs


def _assign_consecutive_pairs(consecutive_pairs, judges, disponibilites, planning, cible_par_juge):
    """Assigner les paires de heats consécutifs"""
    
    for pair in consecutive_pairs:
        wod = pair['wod']
        juges_dispo = disponibilites[wod]
        
        if not juges_dispo:
            continue
            
        # Compter les assignations actuelles
        current_counts = {j: len(planning[j]) for j in judges}
        
        # Trouver le meilleur juge pour cette paire
        best_judge = None
        min_count = float('inf')
        
        for juge in juges_dispo:
            # Vérifier si le juge est sous-chargé et disponible
            if (current_counts[juge] < min_count and 
                current_counts[juge] <= cible_par_juge + 2):
                
                # Vérifier que le juge n'est pas déjà sur un de ces heats
                already_assigned = any(
                    c['heat_num'] in [pair['heat1'], pair['heat2']] and c['wod'] == wod
                    for c in planning[juge]
                )
                
                if not already_assigned:
                    best_judge = juge
                    min_count = current_counts[juge]
        
        if best_judge:
            # Assigner une ligne de chaque heat
            if pair['lines1']:
                ligne1 = pair['lines1'][0]
                planning[best_judge].append({
                    'wod': wod,
                    'lane': ligne1['Lane'],
                    'athlete': ligne1['Competitor'],
                    'division': ligne1['Division'],
                    'location': ligne1['Workout Location'],
                    'start': ligne1['Heat Start Time'],
                    'end': ligne1['Heat End Time'],
                    'heat': ligne1['Heat #'],
                    'heat_num': pair['heat1']
                })
                # Marquer cette ligne comme assignée
                pair['lines1'] = pair['lines1'][1:]
            
            if pair['lines2']:
                ligne2 = pair['lines2'][0]
                planning[best_judge].append({
                    'wod': wod,
                    'lane': ligne2['Lane'],
                    'athlete': ligne2['Competitor'],
                    'division': ligne2['Division'],
                    'location': ligne2['Workout Location'],
                    'start': ligne2['Heat Start Time'],
                    'end': ligne2['Heat End Time'],
                    'heat': ligne2['Heat #'],
                    'heat_num': pair['heat2']
                })
                # Marquer cette ligne comme assignée
                pair['lines2'] = pair['lines2'][1:]


def _assign_remaining_lines(schedule, judges, disponibilites, planning, cible_par_juge):
    """Assigner les lignes restantes après les paires consécutives"""
    
    # Compter les assignations actuelles
    current_counts = {j: len(planning[j]) for j in judges}
    
    # Parcourir toutes les lignes du planning
    for _, ligne in schedule.iterrows():
        wod = ligne['Workout']
        juges_dispo = disponibilites[wod]
        
        if not juges_dispo:
            continue
            
        # Vérifier si cette ligne est déjà assignée
        already_assigned = any(
            c['wod'] == wod and c['lane'] == ligne['Lane'] and c['heat_num'] == ligne['heat_num']
            for juge in judges
            for c in planning[juge]
        )
        
        if already_assigned:
            continue
            
        # Trouver le juge le plus sous-chargé disponible
        best_judge = None
        min_count = float('inf')
        
        for juge in juges_dispo:
            if current_counts[juge] < min_count:
                best_judge = juge
                min_count = current_counts[juge]
        
        if best_judge:
            planning[best_judge].append({
                'wod': wod,
                'lane': ligne['Lane'],
                'athlete': ligne['Competitor'],
                'division': ligne['Division'],
                'location': ligne['Workout Location'],
                'start': ligne['Heat Start Time'],
                'end': ligne['Heat End Time'],
                'heat': ligne['Heat #'],
                'heat_num': ligne['heat_num']
            })
            current_counts[best_judge] += 1


def _assign_without_consecutive(schedule, judges, disponibilites, planning):
    """Attribution sans heats consécutifs"""
    
    for wod, group in schedule.groupby('Workout'):
        juges_dispo = disponibilites[wod]
        if not juges_dispo:
            continue
            
        # Trier par heat num
        group = group.sort_values('heat_num')
        heat_numbers = sorted(group['heat_num'].unique())
        
        for heat_num in heat_numbers:
            heat_lines = group[group['heat_num'] == heat_num]
            
            # Pour chaque ligne du heat, trouver un juge
            for _, ligne in heat_lines.iterrows():
                # Trouver le juge le moins chargé pour ce WOD
                juge_candidat = None
                min_count = float('inf')
                
                for juge in juges_dispo:
                    # Compter les assignations de ce juge pour ce WOD
                    count_wod = sum(1 for c in planning[juge] if c['wod'] == wod)
                    if count_wod < min_count:
                        juge_candidat = juge
                        min_count = count_wod
                
                if juge_candidat:
                    planning[juge_candidat].append({
                        'wod': wod,
                        'lane': ligne['Lane'],
                        'athlete': ligne['Competitor'],
                        'division': ligne['Division'],
                        'location': ligne['Workout Location'],
                        'start': ligne['Heat Start Time'],
                        'end': ligne['Heat End Time'],
                        'heat': ligne['Heat #'],
                        'heat_num': heat_num
                    })


def _balance_assignments(planning, judges, cible_par_juge, tolerance):
    """Équilibrer le nombre total d'assignations"""
    current_counts = {j: len(planning[j]) for j in judges}
    
    # Identifier les déséquilibres
    surcharges = {j: count - cible_par_juge for j, count in current_counts.items() 
                  if count > cible_par_juge + tolerance}
    sous_charges = {j: cible_par_juge - count for j, count in current_counts.items() 
                    if count < cible_par_juge - tolerance}
    
    if surcharges or sous_charges:
        st.warning("🔧 Rééquilibrage des assignations en cours...")
        
        # Essayer de rééquilibrer en transférant des créneaux
        for juge_surcharge, excès in surcharges.items():
            for juge_sous_charge, manque in sous_charges.items():
                if excès > 0 and manque > 0:
                    # Transférer jusqu'à min(excès, manque) créneaux
                    transfer_count = min(excès, manque)
                    
                    # Trouver des créneaux à transférer (les derniers ajoutés)
                    if len(planning[juge_surcharge]) >= transfer_count:
                        creneaux_a_transferer = planning[juge_surcharge][-transfer_count:]
                        
                        # Effectuer le transfert
                        for creneau in creneaux_a_transferer:
                            planning[juge_surcharge].remove(creneau)
                            planning[juge_sous_charge].append(creneau)
                        
                        excès -= transfer_count
                        manque -= transfer_count
                    
                    if excès <= 0:
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
                           format_func=lambda x: "1 heat consécutif max" if x == 1 else "2 heats consécutifs + 2 repos",
                           help="2 heats consécutifs: Priorité absolue aux blocs de 2 heats de suite")

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

                # ANALYSE FINALE DÉTAILLÉE
                st.subheader("📊 Analyse finale des assignations")
                
                total_par_juge = {j: len(creneaux) for j, creneaux in planning.items()}
                total_lignes = sum(total_par_juge.values())
                cible = total_lignes // len(judges)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Assignations par juge:**")
                    sequences_par_juge = {}
                    
                    for juge, count in sorted(total_par_juge.items(), key=lambda x: x[1], reverse=True):
                        ecart = count - cible
                        statut = "✅" if abs(ecart) <= 1 else "⚠️"
                        
                        # Compter les séquences consécutives
                        sequences = []
                        if planning[juge]:
                            creneaux_tries = sorted(planning[juge], key=lambda x: (x['wod'], x['heat_num']))
                            for i in range(len(creneaux_tries) - 1):
                                if (creneaux_tries[i]['wod'] == creneaux_tries[i+1]['wod'] and 
                                    creneaux_tries[i+1]['heat_num'] - creneaux_tries[i]['heat_num'] == 1):
                                    sequences.append(f"{creneaux_tries[i]['heat']}→{creneaux_tries[i+1]['heat']}")
                        
                        sequences_par_juge[juge] = len(sequences)
                        st.write(f"{statut} {juge}: {count} lignes ({ecart:+d}) - {len(sequences)} séq.")
                
                with col2:
                    st.write("**Statistiques globales:**")
                    st.write(f"Total lignes assignées: {total_lignes}")
                    st.write(f"Cible par juge: {cible}")
                    st.write(f"Juges utilisés: {sum(1 for c in total_par_juge.values() if c > 0)}/{len(judges)}")
                    st.write(f"Écart max: {max(total_par_juge.values()) - min(total_par_juge.values())}")
                    
                    if rotation == 2:
                        total_sequences = sum(sequences_par_juge.values())
                        st.write(f"**Total séquences consécutives: {total_sequences}**")
                        
                        if total_sequences > 0:
                            st.write("**Détail des séquences:**")
                            for juge, seq_count in sorted(sequences_par_juge.items(), key=lambda x: x[1], reverse=True):
                                if seq_count > 0:
                                    st.write(f"  {juge}: {seq_count} séquences")

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