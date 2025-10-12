# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:48:50 2025

@author: felima
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from typing import Dict, List
from collections import defaultdict
import traceback

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄 Copyright © 2025 Felix Marcellin")


# ---------------------------------------------------------------------------
# Helper: test de chevauchement entre deux intervalles (start/end)
# ---------------------------------------------------------------------------
def overlaps(a_start, a_end, b_start, b_end) -> bool:
    """
    Retourne True si les intervalles [a_start, a_end) et [b_start, b_end) se chevauchent.
    Fonction compatible avec pandas.Timestamp ou des strings formatées.
    """
    try:
        # Convert to pandas Timestamp si nécessaire
        a_s = pd.to_datetime(a_start)
        a_e = pd.to_datetime(a_end)
        b_s = pd.to_datetime(b_start)
        b_e = pd.to_datetime(b_end)
    except Exception:
        # En dernier recours, comparer comme strings (moins fiable)
        return not (a_end <= b_start or a_start >= b_end)

    return not (a_e <= b_s or a_s >= b_e)


# ---------------------------------------------------------------------------
# 🧾 Génération du PDF par juge
# ---------------------------------------------------------------------------
def generate_pdf_tableau(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)

    for juge, creneaux in planning.items():
        if not creneaux:
            continue

        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Nom de la compétition", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning: {juge}", 0, 1, 'C')
        pdf.ln(10)

        col_widths = [30, 10, 15, 50, 25, 40]
        headers = ["Heure", "Lane", "WOD", "Athlete", "Division", "Emplacement"]

        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])
            start_time = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end_time = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']

            data = [
                f"{start_time} - {end_time}",
                str(c['lane']),
                c['wod'],
                c['athlete'],
                c['division'],
                c['location']
            ]

            for val, width in zip(data, col_widths):
                pdf.cell(width, 10, str(val), border=1, align='C', fill=True)
            pdf.ln()

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf


# ---------------------------------------------------------------------------
# 🔥 Génération du PDF par HEAT
# ---------------------------------------------------------------------------
def generate_heat_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))

    for juge, creneaux in planning.items():
        for c in creneaux:
            start = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']
            key = (start, end, c['wod'], c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = sorted(heat_map.items(), key=lambda x: x[0][0])

    for i in range(0, len(heats), 2):
        pdf.add_page()
        col_width = 90
        row_height = 8
        spacing = 15

        for j in range(2):
            if i + j >= len(heats):
                break

            (start, end, wod, location), lanes = heats[i + j]
            x_position = 10 + j * (col_width + spacing)

            pdf.set_font("Arial", 'B', 10)
            pdf.set_xy(x_position, 15)
            pdf.cell(col_width, row_height, f"HEAT: {start} - {end}", border=1, align='C', fill=True)

            pdf.set_font("Arial", '', 9)
            pdf.set_xy(x_position, 15 + row_height)
            pdf.cell(col_width, row_height, f"WOD: {wod}", border=1, align='C')

            pdf.set_xy(x_position, 15 + 2 * row_height)
            pdf.cell(col_width, row_height, f"Location: {location}", border=1, align='C')

            pdf.set_font("Arial", 'B', 9)
            pdf.set_xy(x_position, 15 + 3 * row_height)
            pdf.cell(col_width / 2, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width / 2, row_height, "Juge", border=1, align='C', fill=True)

            pdf.set_font("Arial", '', 9)
            for k, lane in enumerate(sorted(lanes)):
                y_position = 15 + (4 + k) * row_height
                pdf.set_xy(x_position, y_position)
                pdf.cell(col_width / 2, row_height, str(lane), border=1, align='C')
                pdf.cell(col_width / 2, row_height, lanes[lane], border=1, align='C')

    return pdf


# ---------------------------------------------------------------------------
# 🚀 Application principale Streamlit
# ---------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.header("Import des fichiers")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])

        st.header("Saisie des juges")
        input_method = st.radio(
            "Méthode de saisie des juges",
            options=["Fichier CSV", "Saisie manuelle"],
            index=0
        )

        judges = []
        if input_method == "Fichier CSV":
            judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
            if judges_file:
                judges = pd.read_csv(judges_file, header=None, encoding='latin1')[0].dropna().tolist()
        else:
            judges_text = st.text_area(
                "Saisir les noms des juges (un par ligne)",
                value="Juge 1\nJuge 2\nJuge 3",
                height=150,
                help="Entrez un nom de juge par ligne"
            )
            judges = [j.strip() for j in judges_text.split('\n') if j.strip()]

            if judges:
                st.write("Juges saisis:")
                st.write(judges)

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            st.subheader("Aperçu du planning importé")
            st.dataframe(schedule.head())

            required_columns = ['Workout', 'Lane', 'Competitor', 'Division', 'Workout Location', 'Heat Start Time', 'Heat End Time']
            if not
