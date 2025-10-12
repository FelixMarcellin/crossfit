# -*- coding: utf-8 -*-
"""
Updated on Oct 2025
Author: Felix Marcellin
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
st.title("Planning Juges by Crossfit Amiens 🦄")


# ---------------------------------------------------------------------------
# Helper: test de chevauchement entre deux intervalles
# ---------------------------------------------------------------------------
def overlaps(a_start, a_end, b_start, b_end) -> bool:
    """Retourne True si les intervalles [a_start, a_end) et [b_start, b_end) se chevauchent."""
    try:
        a_s, a_e = pd.to_datetime(a_start), pd.to_datetime(a_end)
        b_s, b_e = pd.to_datetime(b_start), pd.to_datetime(b_end)
    except Exception:
        return not (a_end <= b_start or a_start >= b_end)
    return not (a_e <= b_s or a_s >= b_e)


# ---------------------------------------------------------------------------
# PDF par juge
# ---------------------------------------------------------------------------
def generate_pdf_tableau(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    pdf = FPDF()
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

        col_widths = [25, 10, 10, 45, 25, 35, 30]
        headers = ["Heure", "Lane", "Heat #", "Athlete", "Division", "WOD", "Emplacement"]

        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 9)
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_font("Arial", '', 8)
        row_colors = [(255, 255, 255), (240, 240, 240)]

        for i, c in enumerate(creneaux):
            pdf.set_fill_color(*row_colors[i % 2])
            s = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            e = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']

            data = [
                f"{s}-{e}",
                str(c['lane']),
                str(c['heat']),
                c['athlete'],
                c['division'],
                c['wod'],
                c['location']
            ]
            for val, w in zip(data, col_widths):
                pdf.cell(w, 8, str(val), border=1, align='C', fill=True)
            pdf.ln()

        pdf.ln(6)
        pdf.set_font("Arial", 'I', 8)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total: {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf


# ---------------------------------------------------------------------------
# PDF par HEAT
# ---------------------------------------------------------------------------
def generate_heat_pdf(planning: Dict[str, List[Dict[str, any]]]) -> FPDF:
    heat_map = defaultdict(lambda: defaultdict(str))

    for juge, creneaux in planning.items():
        for c in creneaux:
            start = c['start'].strftime('%H:%M') if hasattr(c['start'], 'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'], 'strftime') else c['end']
            key = (c['wod'], c['heat'], start, end, c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 9)

    heats = sorted(heat_map.items(), key=lambda x: (x[0][2], x[0][0], x[0][1]))  # tri par heure, wod, heat

    for (wod, heat, start, end, location), lanes in heats:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f"{wod} - Heat #{heat} ({start}-{end})", 0, 1, 'C')
        pdf.set_font("Arial", '', 9)
        pdf.cell(0, 6, f"Location: {location}", 0, 1, 'C')
        pdf.ln(4)

        pdf.set_font("Arial", 'B', 9)
        pdf.cell(30, 8, "Lane", 1, 0, 'C', True)
        pdf.cell(60, 8, "Juge", 1, 1, 'C', True)

        pdf.set_font("Arial", '', 9)
        for lane, juge in sorted(lanes.items()):
            pdf.cell(30, 8, str(lane), 1, 0, 'C')
            pdf.cell(60, 8, juge, 1, 1, 'C')

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

            required_columns = [
                'Workout', 'Lane', 'Competitor', 'Division', 'Workout Location',
                'Heat Start Time', 'Heat End Time', 'Heat #'
            ]
            if not all(col in schedule.columns for col in required_columns):
                st.error("Erreur: Colonnes manquantes.")
                st.write("Colonnes requises:", required_columns)
                st.write("Colonnes trouvées:", list(schedule.columns))
                return

            # Nettoyage
            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            wods = sorted(schedule['Workout'].unique())

            st.header("Disponibilité des Juges par WOD")
            disponibilites = {wod: set() for wod in wods}
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"WOD: {wod}"):
                        disponibilites[wod] = set(st.multiselect(
                            f"Sélection pour {wod}",
                            judges,
                            key=f"dispo_{wod}"
                        ))

            # Option de rotation
            rotation_freq = st.selectbox(
                "Changer de juge tous les ... heats",
                options=[1, 2],
                index=0,
                help="1 = changement à chaque heat, 2 = tous les 2 heats consécutifs"
            )

            if st.button("Générer les plannings"):
                planning = {juge: [] for juge in judges}

                grouped = schedule.groupby(['Workout', 'Heat #', 'Heat Start Time', 'Heat End Time', 'Workout Location'])
                heats = sorted(grouped.groups.keys(), key=lambda x: (x[2], x[0], x[1]))  # tri par heure

                for idx, (wod, heat, start, end, location) in enumerate(heats):
                    juges_dispo = list(disponibilites[wod])
                    if not juges_dispo:
                        st.error(f"Aucun juge pour {wod}!")
                        continue

                    start_pos = (idx // rotation_freq) % len(juges_dispo)
                    assigned = False

                    for offset in range(len(juges_dispo)):
                        candidate = juges_dispo[(start_pos + offset) % len(juges_dispo)]

                        # vérifier si le juge a déjà un créneau qui se chevauche
                        conflict = any(
                            overlaps(c['start'], c['end'], start, end) for c in planning[candidate]
                        )
                        if conflict:
                            continue

                        # ✅ Affecter une seule lane de ce heat à ce juge
                        for _, row in grouped.get_group((wod, heat, start, end, location)).iterrows():
                            planning[candidate].append({
                                'wod': wod,
                                'heat': heat,
                                'lane': row['Lane'],
                                'athlete': row['Competitor'],
                                'division': row['Division'],
                                'location': location,
                                'start': start,
                                'end': end
                            })
                            break  # une seule ligne (lane) par heat

                        assigned = True
                        break

                    if not assigned:
                        st.error(f"Aucun juge libre pour {wod} - Heat {heat} ({start}-{end}) à {location}")

                pdf_juges = generate_pdf_tableau({k: v for k, v in planning.items() if v})
                pdf_heats = generate_heat_pdf({k: v for k, v in planning.items() if v})

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_j:
                    pdf_juges.output(tmp_j.name)
                    with open(tmp_j.name, "rb") as f:
                        st.download_button("📘 Télécharger planning par juge", f, "planning_juges.pdf")
                    os.unlink(tmp_j.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_h:
                    pdf_heats.output(tmp_h.name)
                    with open(tmp_h.name, "rb") as f:
                        st.download_button("🔥 Télécharger planning par heat", f, "planning_heats.pdf")
                    os.unlink(tmp_h.name)

                st.success("✅ PDF générés avec succès !")

                st.header("Récapitulatif des affectations")
                for juge, creneaux in planning.items():
                    if creneaux:
                        with st.expander(f"{juge} ({len(creneaux)} créneaux)"):
                            st.table(pd.DataFrame(creneaux))

        except Exception as e:
            st.error("Erreur lors du traitement :")
            st.code(traceback.format_exc())
    else:
        st.info("Veuillez uploader le fichier de planning et saisir les juges pour commencer.")


if __name__ == "__main__":
    main()
