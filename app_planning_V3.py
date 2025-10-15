# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 4 : Blocs de 2 heats + 2 repos, pas de double affectation sur un même heat
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from typing import Dict, List
from collections import defaultdict
import traceback

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄 - Version 4")


# ============================================================
# PDF TABLEAU PAR JUGE
# ============================================================
def generate_pdf_tableau(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)

    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        def parse_time(x):
            try:
                return pd.to_datetime(x, format='%H:%M')
            except Exception:
                try:
                    return pd.to_datetime(str(x))
                except Exception:
                    return pd.NaT

        creneaux = sorted(creneaux, key=lambda c: (c.get('wod', ''), parse_time(c.get('start', ''))))

        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Nom de la compétition", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(10)

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

            pdf.set_xy(x_position, 15 + 2 * row_height)
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(col_width / 2, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width / 2, row_height, "Juge", border=1, align='C', fill=True)

            pdf.set_font("Arial", '', 9)
            for k, lane in enumerate(sorted(lanes)):
                y_position = 15 + (3 + k) * row_height
                pdf.set_xy(x_position, y_position)
                pdf.cell(col_width / 2, row_height, str(lane), border=1, align='C')
                pdf.cell(col_width / 2, row_height, lanes[lane], border=1, align='C')

    return pdf


# ============================================================
# EXTRACTION DU NUMÉRO DE HEAT
# ============================================================
def extract_heat_number(heat_str):
    import re
    if pd.isna(heat_str):
        return 0
    if isinstance(heat_str, (int, float)):
        return int(heat_str)
    try:
        numbers = re.findall(r'\d+', str(heat_str))
        if numbers:
            return int(numbers[0])
    except Exception:
        pass
    return 0


# ============================================================
# ATTRIBUTION DES JUGES VERSION 4
# ============================================================
def assign_judges_equitable(schedule, judges, disponibilites, rotation):
    planning = {j: [] for j in judges}

    schedule = schedule.copy()
    schedule['heat_num'] = schedule['Heat #'].apply(extract_heat_number)
    schedule = schedule.sort_values(['Workout', 'heat_num', 'Lane'])

    heats = []
    for (wod, heat_num), group in schedule.groupby(['Workout', 'heat_num']):
        heats.append({'wod': wod, 'heat_num': heat_num, 'rows': group.to_dict('records')})

    last_heat_index = {j: -999 for j in judges}
    heats_par_juge = {j: 0 for j in judges}

    for idx, heat in enumerate(heats):
        wod = heat['wod']
        juges_dispo = disponibilites.get(wod, [])
        if not juges_dispo:
            continue

        for row in heat['rows']:
            candidats_valides = []
            for j in juges_dispo:
                if any(c['wod'] == wod and c['heat_num'] == heat['heat_num'] for c in planning[j]):
                    continue

                # Bloc de 2 heats + 2 repos
                diff = idx - last_heat_index[j]
                if heats_par_juge[j] % 2 == 0 and diff < 2:
                    continue

                candidats_valides.append(j)

            if not candidats_valides:
                candidats_valides = [j for j in juges_dispo
                                     if not any(c['wod'] == wod and c['heat_num'] == heat['heat_num']
                                                for c in planning[j])]

            if not candidats_valides:
                continue

            best = min(candidats_valides, key=lambda j: len(planning[j]))

            planning[best].append({
                'wod': wod,
                'lane': row['Lane'],
                'athlete': row['Competitor'],
                'division': row['Division'],
                'location': row['Workout Location'],
                'start': row['Heat Start Time'],
                'end': row['Heat End Time'],
                'heat': row['Heat #'],
                'heat_num': heat['heat_num']
            })
            last_heat_index[best] = idx
            heats_par_juge[best] += 1

    return planning


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
            judges_text = st.text_area("Saisir les noms des juges (un par ligne)",
                                       value="Juge 1\nJuge 2\nJuge 3", height=150)
            judges = [j.strip() for j in judges_text.split('\n') if j.strip()]
            if judges:
                st.write("Juges saisis:")
                st.write(judges)

        st.header("Paramètres de rotation")
        rotation = st.radio("Mode d'attribution",
                           options=[1, 2],
                           index=1,
                           format_func=lambda x: "1 heat consécutif max" if x == 1 else "2 heats consécutifs + 2 repos")

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')

            required_columns = ['Workout', 'Lane', 'Competitor', 'Division',
                                'Workout Location', 'Heat Start Time', 'Heat End Time', 'Heat #']
            if not all(col in schedule.columns for col in required_columns):
                st.error("Erreur: Colonnes manquantes.")
                st.write("Colonnes trouvées:", list(schedule.columns))
                return

            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")

            st.subheader("Aperçu des données")
            st.dataframe(schedule[['Workout', 'Heat #', 'Lane', 'Competitor']].head())

            schedule['heat_num'] = schedule['Heat #'].apply(extract_heat_number)
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
                wods_sans_juge = [w for w, j in disponibilites.items() if not j]
                if wods_sans_juge:
                    st.error(f"Aucun juge pour : {', '.join(wods_sans_juge)}")
                    return

                planning = assign_judges_equitable(schedule, judges, disponibilites, rotation)

                st.subheader("📊 Analyse des assignations")
                total_par_juge = {j: len(v) for j, v in planning.items()}
                cible = sum(total_par_juge.values()) // len(judges)

                for j, c in sorted(total_par_juge.items(), key=lambda x: x[1], reverse=True):
                    ecart = c - cible
                    st.write(f"{'✅' if abs(ecart) <= 1 else '⚠️'} {j}: {c} créneaux ({ecart:+d})")

                pdf_juges = generate_pdf_tableau(planning)
                pdf_heats = generate_heat_pdf(planning)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
                    pdf_juges.output(tmp1.name)
                    with open(tmp1.name, "rb") as f:
                        st.download_button("📘 Télécharger planning par juge", f, "planning_juges.pdf")
                    os.unlink(tmp1.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
                    pdf_heats.output(tmp2.name)
                    with open(tmp2.name, "rb") as f:
                        st.download_button("📗 Télécharger planning par heat", f, "planning_heats.pdf")
                    os.unlink(tmp2.name)

                st.success("✅ Plannings générés avec succès !")

        except Exception as e:
            st.error("Erreur lors du traitement :")
            st.code(traceback.format_exc())
    else:
        st.info("Veuillez uploader le fichier de planning et saisir les juges.")


if __name__ == "__main__":
    main()
