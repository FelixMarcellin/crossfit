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
# ATTRIBUTION ÉQUITABLE DES JUGES - VERSION STRICTE
# ============================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation):
    planning = {j: [] for j in judges}
    
    # Préparer les données avec les numéros de heat
    schedule_with_heat_nums = schedule.copy()
    schedule_with_heat_nums['heat_num'] = schedule_with_heat_nums['Heat #'].apply(extract_heat_number)
    
    # Calculer le nombre total de créneaux par WOD
    total_creneaux_par_wod = {}
    for wod, group in schedule_with_heat_nums.groupby('Workout'):
        total_creneaux_par_wod[wod] = len(group)
    
    # Calculer le nombre cible de créneaux par juge
    total_creneaux = sum(total_creneaux_par_wod.values())
    cible_par_juge = total_creneaux // len(judges)
    tolerance = 2  # ±2 créneaux
    
    st.write(f"🎯 Cible: {cible_par_juge} créneaux par juge (tolérance: ±{tolerance})")

    for wod, group in schedule_with_heat_nums.groupby('Workout'):
        juges_dispo = list(disponibilites[wod])
        if not juges_dispo:
            st.error(f"Aucun juge sélectionné pour {wod}")
            continue

        # Trier par numéro de heat
        group = group.sort_values('heat_num')
        
        # Grouper par heat num
        heats_par_num = {}
        for heat_num, heat_group in group.groupby('heat_num'):
            heats_par_num[heat_num] = heat_group
        
        # Pour rotation=2, essayer de créer des paires de heats consécutifs
        if rotation == 2:
            _assign_with_consecutive_heats(heats_par_num, juges_dispo, planning, wod, judges, cible_par_juge, tolerance)
        else:
            # Pour rotation=1, attribution normale
            _assign_without_consecutive(heats_par_num, juges_dispo, planning, wod)

    # Équilibrer le nombre total de créneaux par juge
    _balance_total_assignments(planning, judges, cible_par_juge, tolerance, schedule_with_heat_nums, disponibilites)
    
    return planning


def _assign_with_consecutive_heats(heats_par_num, juges_dispo, planning, wod, judges, cible_par_juge, tolerance):
    """Attribution avec priorité aux paires de heats consécutifs"""
    heat_nums = sorted(heats_par_num.keys())
    
    # Compter les créneaux actuels par juge
    current_counts = {j: len(planning[j]) for j in judges}
    
    # Essayer de former des paires de heats consécutifs
    used_heats = set()
    
    for i in range(len(heat_nums) - 1):
        current_heat = heat_nums[i]
        next_heat = heat_nums[i + 1]
        
        # Vérifier si c'est une paire consécutive
        if next_heat - current_heat == 1:
            # Vérifier si ces heats ne sont pas déjà attribués
            if current_heat not in used_heats and next_heat not in used_heats:
                # Trouver le juge le plus sous-chargé qui peut prendre les 2 heats
                juge_candidat = None
                min_count = float('inf')
                
                for juge in juges_dispo:
                    if current_counts[juge] < min_count and current_counts[juge] <= cible_par_juge + tolerance:
                        # Vérifier que le juge n'est pas déjà sur un de ces heats
                        juge_already_assigned = any(
                            c['heat'] in [str(heats_par_num[current_heat].iloc[0]['Heat #']), 
                            str(heats_par_num[next_heat].iloc[0]['Heat #'])] 
                            for c in planning[juge]
                        )
                        if not juge_already_assigned:
                            juge_candidat = juge
                            min_count = current_counts[juge]
                
                if juge_candidat:
                    # Attribuer les 2 heats consécutifs au même juge
                    for heat_num in [current_heat, next_heat]:
                        heat_group = heats_par_num[heat_num]
                        for _, row in heat_group.iterrows():
                            planning[juge_candidat].append({
                                'wod': wod,
                                'lane': row.get('Lane', ''),
                                'athlete': row.get('Competitor', ''),
                                'division': row.get('Division', ''),
                                'location': row.get('Workout Location', ''),
                                'start': row.get('Heat Start Time', ''),
                                'end': row.get('Heat End Time', ''),
                                'heat': row['Heat #']
                            })
                            current_counts[juge_candidat] += 1
                    
                    used_heats.add(current_heat)
                    used_heats.add(next_heat)
    
    # Attribuer les heats restants (ceux qui n'ont pas pu être mis en paires)
    remaining_heat_nums = [hn for hn in heat_nums if hn not in used_heats]
    
    for heat_num in remaining_heat_nums:
        heat_group = heats_par_num[heat_num]
        
        # Trouver le juge le plus sous-chargé
        juge_candidat = None
        min_count = float('inf')
        
        for juge in juges_dispo:
            if current_counts[juge] < min_count and current_counts[juge] <= cible_par_juge + tolerance:
                # Vérifier que le juge n'est pas déjà sur ce heat
                juge_already_assigned = any(
                    c['heat'] == str(heat_group.iloc[0]['Heat #']) 
                    for c in planning[juge]
                )
                if not juge_already_assigned:
                    juge_candidat = juge
                    min_count = current_counts[juge]
        
        if juge_candidat:
            for _, row in heat_group.iterrows():
                planning[juge_candidat].append({
                    'wod': wod,
                    'lane': row.get('Lane', ''),
                    'athlete': row.get('Competitor', ''),
                    'division': row.get('Division', ''),
                    'location': row.get('Workout Location', ''),
                    'start': row.get('Heat Start Time', ''),
                    'end': row.get('Heat End Time', ''),
                    'heat': row['Heat #']
                })
                current_counts[juge_candidat] += 1


def _assign_without_consecutive(heats_par_num, juges_dispo, planning, wod):
    """Attribution sans heats consécutifs"""
    heat_nums = sorted(heats_par_num.keys())
    
    for heat_num in heat_nums:
        heat_group = heats_par_num[heat_num]
        
        # Compter les créneaux actuels par juge pour ce WOD
        counts_this_wod = {j: sum(1 for c in planning[j] if c['wod'] == wod) for j in juges_dispo}
        
        # Trier les juges par nombre d'assignations (le moins chargé d'abord)
        juges_tries = sorted(juges_dispo, key=lambda j: (counts_this_wod[j], len(planning[j])))
        
        assigned_judges = set()
        
        for _, row in heat_group.iterrows():
            # Trouver le juge disponible le moins chargé
            for juge in juges_tries:
                if juge not in assigned_judges:
                    # Vérifier que le juge n'a pas fait le heat précédent
                    can_assign = True
                    if heat_num > min(heat_nums):
                        previous_heat = heat_num - 1
                        if any(c['heat'] == str(heats_par_num[previous_heat].iloc[0]['Heat #']) for c in planning[juge] if c['wod'] == wod):
                            can_assign = False
                    
                    if can_assign:
                        planning[juge].append({
                            'wod': wod,
                            'lane': row.get('Lane', ''),
                            'athlete': row.get('Competitor', ''),
                            'division': row.get('Division', ''),
                            'location': row.get('Workout Location', ''),
                            'start': row.get('Heat Start Time', ''),
                            'end': row.get('Heat End Time', ''),
                            'heat': row['Heat #']
                        })
                        assigned_judges.add(juge)
                        break


def _balance_total_assignments(planning, judges, cible_par_juge, tolerance, schedule, disponibilites):
    """Équilibrer le nombre total de créneaux par juge"""
    current_counts = {j: len(planning[j]) for j in judges}
    
    # Identifier les juges sur-chargés et sous-chargés
    surcharges = {j: count - cible_par_juge for j, count in current_counts.items() if count > cible_par_juge + tolerance}
    sous_charges = {j: cible_par_juge - count for j, count in current_counts.items() if count < cible_par_juge - tolerance}
    
    if surcharges or sous_charges:
        st.warning("🔧 Rééquilibrage des assignations en cours...")
        
        # Rééquilibrer en transférant des créneaux des surchargés vers les sous-chargés
        for juge_surcharge, excès in surcharges.items():
            for juge_sous_charge, manque in sous_charges.items():
                if excès > 0 and manque > 0:
                    # Trouver des créneaux à transférer
                    creneaux_a_transferer = []
                    for creneau in planning[juge_surcharge]:
                        # Vérifier si le juge sous-chargé est disponible pour ce WOD
                        wod = creneau['wod']
                        if juge_sous_charge in disponibilites[wod]:
                            creneaux_a_transferer.append(creneau)
                            if len(creneaux_a_transferer) >= min(excès, manque):
                                break
                    
                    # Effectuer le transfert
                    for creneau in creneaux_a_transferer:
                        planning[juge_surcharge].remove(creneau)
                        planning[juge_sous_charge].append(creneau)
                        excès -= 1
                        manque -= 1
                        
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
                           format_func=lambda x: "1 heat consécutif max" if x == 1 else "2 heats consécutifs obligatoires",
                           help="2 heats consécutifs: Les juges font obligatoirement 2 heats de suite quand possible")
        
        st.info("""
        **Mode 2 heats consécutifs:**
        - Les juges font OBLIGATOIREMENT 2 heats de suite quand c'est possible
        - Si impossible, ils peuvent faire 1 ou 3 heats
        - Répartition équitable du nombre total de heats par juge (±2)
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

                # Analyse détaillée des résultats
                st.subheader("📊 Analyse détaillée des assignations")
                
                # Nombre total de créneaux par juge
                total_par_juge = {j: len(creneaux) for j, creneaux in planning.items()}
                total_creneaux = sum(total_par_juge.values())
                cible_par_juge = total_creneaux // len(judges)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Répartition par juge:**")
                    for juge, count in sorted(total_par_juge.items(), key=lambda x: x[1], reverse=True):
                        ecart = count - cible_par_juge
                        statut = "✅" if abs(ecart) <= 2 else "⚠️"
                        st.write(f"{statut} {juge}: {count} créneaux ({ecart:+d})")
                
                with col2:
                    st.write("**Séquences consécutives:**")
                    for juge, creneaux in planning.items():
                        if creneaux:
                            # Analyser les séquences par WOD
                            sequences = []
                            for wod in set(c['wod'] for c in creneaux):
                                heats_wod = sorted([extract_heat_number(c['heat']) for c in creneaux if c['wod'] == wod])
                                if len(heats_wod) > 1:
                                    consecutive = 1
                                    for i in range(1, len(heats_wod)):
                                        if heats_wod[i] - heats_wod[i-1] == 1:
                                            consecutive += 1
                                    if consecutive > 1:
                                        sequences.append(f"{wod}:{consecutive}")
                            
                            if sequences:
                                st.write(f"**{juge}:** {', '.join(sequences)}")
                
                with col3:
                    st.write("**Statistiques globales:**")
                    st.write(f"Total créneaux: {total_creneaux}")
                    st.write(f"Cible par juge: {cible_par_juge}")
                    st.write(f"Juges utilisés: {sum(1 for j in judges if planning[j])}/{len(judges)}")
                    st.write(f"Écart max: {max(total_par_juge.values()) - min(total_par_juge.values())}")

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