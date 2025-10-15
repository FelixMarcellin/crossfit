# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 7.7 : Corrections alignement et format
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from collections import defaultdict
import traceback
import re


# ========================
# CONFIG STREAMLIT
# ========================
st.set_page_config(page_title="Planning Juges - Crossfit Amiens", layout="wide")
st.title("🏋️‍♂️ Planning Juges - Crossfit Amiens 🦄 (Version 7.7)")


# ========================
# FONCTION NETTOYAGE TEXTE ROBUSTE
# ========================
def clean_text(text):
    """Nettoie le texte des caractères spéciaux problématiques de manière robuste"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Liste complète des remplacements
    replacements = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
        'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n', 'ß': 'ss',
        'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A',
        'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
        'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O',
        'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
        'Ç': 'C', 'Ñ': 'N',
        'œ': 'oe', 'æ': 'ae', '€': 'E', '£': 'GBP',
        '§': 'S', 'µ': 'u', '°': 'deg', '²': '2', '³': '3'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Supprimer tout caractère non-ASCII restant
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text


# ========================
# PDF PAR JUGE (CORRIGÉ - portrait et alignement)
# ========================
def generate_pdf_tableau(planning: dict, competition_name: str) -> FPDF:
    """Génère le PDF par juge en mode portrait avec alignement corrigé"""
    pdf = FPDF(orientation='P')  # MODE PORTRAIT
    pdf.set_auto_page_break(auto=True, margin=10)
    
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
        
        # En-tête
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, clean_text(competition_name), 0, 1, 'C')
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Planning : {clean_text(juge)}", 0, 1, 'C')
        pdf.ln(5)

        # En-têtes de colonnes CORRIGÉES - "Lane" au lieu de "Ln"
        headers = ["Heure", "Lane", "WOD", "Heat", "Athlete", "Division"]
        # Largeurs optimisées pour portrait
        col_widths = [22, 10, 18, 12, 65, 25]
        
        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 8)
        
        # DESSINER LA PREMIÈRE LIGNE CORRECTEMENT
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 6, h, 1, 0, 'C', fill=True)
        pdf.ln()
        
        # Contenu du tableau
        pdf.set_font("Arial", '', 7)
        row_colors = [(255, 255, 255), (240, 240, 240)]
        
        for i, c in enumerate(creneaux):
            # Vérifier si on dépasse la page
            if pdf.get_y() > 270:  # Bas de page en portrait
                pdf.add_page()
                # Ré-afficher les en-têtes
                pdf.set_font("Arial", 'B', 8)
                pdf.set_fill_color(211, 211, 211)
                x_start = pdf.get_x()
                y_start = pdf.get_y()
                for h, w in zip(headers, col_widths):
                    pdf.cell(w, 6, h, 1, 0, 'C', fill=True)
                pdf.ln()
                pdf.set_font("Arial", '', 7)
            
            pdf.set_fill_color(*row_colors[i % 2])
            
            # FORMATAGE DES DONNÉES - HEURE SUR UNE SEULE LIGNE
            start = clean_text(str(c['start']))[:5]  # HH:MM
            end = clean_text(str(c['end']))[:5]      # HH:MM
            
            # Tronquer les textes longs
            athlete_name = clean_text(str(c['athlete']))
            if len(athlete_name) > 30:
                athlete_name = athlete_name[:30] + "..."
                
            division_name = clean_text(str(c['division']))
            if len(division_name) > 15:
                division_name = division_name[:15] + "..."
            
            wod_name = clean_text(str(c['wod']))
            if len(wod_name) > 10:
                wod_name = wod_name[:10] + ".."
            
            heat_name = clean_text(str(c['heat']))
            if len(heat_name) > 8:
                heat_name = heat_name[:8] + ".."
            
            # HEURE SUR UNE SEULE LIGNE : "HH:MM-HH:MM"
            vals = [
                f"{start}-{end}",  # UNE SEULE LIGNE
                clean_text(str(c['lane'])),
                wod_name,
                heat_name,
                athlete_name,
                division_name
            ]
            
            # Dessiner chaque cellule sur la même ligne
            for v, w in zip(vals, col_widths):
                pdf.cell(w, 5, v, 1, 0, 'C', fill=True)
            pdf.ln()

        # Pied de page
        pdf.ln(3)
        pdf.set_font("Arial", 'I', 7)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 5, f"Total : {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf


# ========================
# PDF PAR HEAT (4 tableaux par page)
# ========================
def generate_heat_pdf(planning: dict, competition_name: str) -> FPDF:
    """Génère le PDF par heat (4 tableaux par page - grand espacement)"""
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (c['wod'], c['heat'], c['start'], c['end'], c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = sorted(heat_map.items(), key=lambda x: (x[0][0], x[0][1]))

    for i in range(0, len(heats), 4):
        pdf.add_page()

        # Nom de la compétition
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_text(competition_name), 0, 1, 'C')
        pdf.ln(4)

        col_width = 85
        row_height = 8
        spacing_x = 15
        
        # Grand espacement entre les rangées
        y_positions = [25, 120]

        for j in range(4):
            if i + j >= len(heats):
                break

            (wod, heat, start, end, loc), lanes = heats[i + j]
            
            col = j % 2
            row = j // 2
            
            x = 10 + col * (col_width + spacing_x)
            y_start = y_positions[row]

            # En-tête du bloc heat
            pdf.set_font("Arial", 'B', 10)
            pdf.set_xy(x, y_start)
            header_text = clean_text(f"{wod} | {heat} | {start}-{end}")
            pdf.cell(col_width, row_height, header_text, border=1, align='C')
            
            # En-tête du tableau
            pdf.set_xy(x, y_start + row_height)
            pdf.set_font("Arial", 'B', 9)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(col_width / 2, row_height, "Lane", border=1, align='C', fill=True)
            pdf.cell(col_width / 2, row_height, "Juge", border=1, align='C', fill=True)

            # Contenu du tableau
            pdf.set_font("Arial", '', 9)
            pdf.set_fill_color(255, 255, 255)
            
            for lane_num, (lane, juge_name) in enumerate(sorted(lanes.items())):
                y_pos = y_start + row_height * 2 + lane_num * row_height
                pdf.set_xy(x, y_pos)
                pdf.cell(col_width / 2, row_height, clean_text(str(lane)), border=1, align='C')
                pdf.cell(col_width / 2, row_height, clean_text(juge_name), border=1, align='C')

    return pdf


# ========================
# HELPERS
# ========================
def extract_heat_number(heat_str):
    if pd.isna(heat_str):
        return 0
    if isinstance(heat_str, (int, float)):
        return int(heat_str)
    numbers = re.findall(r'\d+', str(heat_str))
    return int(numbers[0]) if numbers else 0


# ========================
# ATTRIBUTION ÉQUILIBRÉE + 2-ON/2-OFF
# ========================
def assign_judges_equitable(schedule, judges, disponibilites, rotation):
    """Attribution équilibrée avec blocs 2-on / 2-off prioritaires"""
    planning = {j: [] for j in judges}
    df = schedule.copy()
    df['heat_num'] = df['Heat #'].apply(extract_heat_number)
    df = df.sort_values(['Heat Start Time', 'Workout', 'heat_num', 'Lane']).reset_index(drop=True)

    heats = []
    for (wod, heat_num, start, end), g in df.groupby(['Workout', 'heat_num', 'Heat Start Time', 'Heat End Time']):
        heats.append({'wod': wod, 'heat_num': heat_num, 'start': start, 'end': end, 'rows': g.to_dict('records')})

    n_judges = len(judges)
    target = len(df) // n_judges if n_judges > 0 else 0

    state = {j: {'last': -999, 'on': 0, 'rest': 0, 'count': 0} for j in judges}
    ON, REST = 2, 2

    for idx, heat in enumerate(heats):
        wod = heat['wod']
        dispo = disponibilites.get(wod, judges)
        used = set()

        for row in heat['rows']:
            best, best_score = None, 9999
            for j in dispo:
                if j in used:
                    continue
                s = state[j]
                # Éviter doublon même heat
                if any(c['wod'] == wod and c['heat_num'] == heat['heat_num'] for c in planning[j]):
                    continue
                score = 0
                if s['last'] == idx - 1 and s['on'] > 0:
                    score -= 100  # continuer bloc
                if s['rest'] > 0:
                    score += 50  # éviter ceux en repos
                score += max(0, s['count'] - target)
                score += (s['count'] - target) * 2
                if score < best_score:
                    best_score, best = score, j
            if best is None:
                best = min(judges, key=lambda j: state[j]['count'])

            planning[best].append({
                'wod': clean_text(str(wod)),
                'lane': clean_text(str(row['Lane'])),
                'athlete': clean_text(str(row['Competitor'])),
                'division': clean_text(str(row['Division'])),
                'location': clean_text(str(row['Workout Location'])),
                'start': clean_text(str(row['Heat Start Time'])),
                'end': clean_text(str(row['Heat End Time'])),
                'heat': clean_text(str(row['Heat #'])),
                'heat_num': heat['heat_num']
            })
            used.add(best)

            # Mise à jour état
            s = state[best]
            if s['last'] == idx - 1 and s['on'] > 0:
                s['on'] -= 1
                if s['on'] == 0:
                    s['rest'] = REST
            else:
                s['on'] = ON - 1
                s['rest'] = 0
            s['last'] = idx
            s['count'] += 1

        # décrémenter repos
        for j in judges:
            if state[j]['rest'] > 0 and state[j]['last'] != idx:
                state[j]['rest'] -= 1

    # Équilibrage final
    avg = sum(state[j]['count'] for j in judges) / len(judges)
    over = [j for j in judges if state[j]['count'] > avg + 1]
    under = [j for j in judges if state[j]['count'] < avg - 1]
    for j_over in over:
        for j_under in under:
            if state[j_over]['count'] <= avg + 1 or state[j_under]['count'] >= avg - 1:
                continue
            if planning[j_over]:
                c = planning[j_over].pop()
                planning[j_under].append(c)
                state[j_over]['count'] -= 1
                state[j_under]['count'] += 1

    return planning


# ========================
# MAIN STREAMLIT
# ========================
def main():
    with st.sidebar:
        st.header("📂 Fichiers d'entrée")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])

        st.header("🏋️‍♀️ Nom de la compétition")
        competition_name = st.text_input("Nom à afficher sur les PDF", "Unicorn and the Beast 2025")

        st.header("👩‍⚖️ Juges")
        judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
        if judges_file:
            judges_df = pd.read_csv(judges_file, header=None, encoding='latin1')
            judges = [clean_text(str(j)) for j in judges_df[0].dropna().tolist()]
        else:
            judges_text = st.text_area("Saisir les juges (un par ligne)", "Juge 1\nJuge 2\nJuge 3")
            judges = [clean_text(j.strip()) for j in judges_text.split('\n') if j.strip()]

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            required = ['Workout', 'Lane', 'Competitor', 'Division', 'Workout Location',
                        'Heat Start Time', 'Heat End Time', 'Heat #']
            if not all(c in schedule.columns for c in required):
                st.error("⚠️ Colonnes manquantes dans le fichier Excel.")
                return

            # Nettoyer les données dès la lecture
            for col in ['Workout', 'Competitor', 'Division', 'Workout Location', 'Heat #']:
                if col in schedule.columns:
                    schedule[col] = schedule[col].apply(lambda x: clean_text(str(x)) if pd.notna(x) else "")
            
            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            wods = sorted(schedule['Workout'].dropna().unique())
            
            st.header("📅 Disponibilités des juges")
            disponibilites = {}
            cols = st.columns(3)
            for i, wod in enumerate(wods):
                with cols[i % 3]:
                    with st.expander(f"{wod}"):
                        select_all = st.checkbox(f"Tout sélectionner ({wod})", key=f"sel_{wod}")
                        if select_all:
                            disponibilites[wod] = judges
                        else:
                            disponibilites[wod] = st.multiselect("Juges disponibles", judges, key=f"multi_{wod}")

            if st.button("🦄 Générer le planning"):
                planning = assign_judges_equitable(schedule, judges, disponibilites, 2)

                st.subheader("📊 Équilibre des assignations")
                counts = {j: len(planning[j]) for j in judges}
                total = sum(counts.values())
                target = total // len(judges)
                for j in sorted(counts, key=counts.get, reverse=True):
                    ecart = counts[j] - target
                    emoji = "✅" if abs(ecart) <= 1 else "⚠️"
                    st.write(f"{emoji} {j}: {counts[j]} créneaux ({ecart:+d})")

                try:
                    pdf_juges = generate_pdf_tableau(planning, competition_name)
                    pdf_heats = generate_heat_pdf(planning, competition_name)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t1:
                        pdf_juges.output(t1.name)
                        with open(t1.name, "rb") as f:
                            st.download_button("📘 Télécharger planning par juge", f, "planning_juges.pdf")
                        os.unlink(t1.name)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t2:
                        pdf_heats.output(t2.name)
                        with open(t2.name, "rb") as f:
                            st.download_button("📗 Télécharger planning par heat", f, "planning_heats.pdf")
                        os.unlink(t2.name)
                    
                    st.success("✅ Planning généré avec succès !")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération des PDF: {str(e)}")
                    st.info("💡 Essayez de simplifier les noms des juges ou des athlètes")

        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")
            st.info("💡 Vérifiez que le fichier Excel est valide et ne contient pas de caractères spéciaux problématiques")

    else:
        st.info("👉 Veuillez importer un fichier Excel et saisir la liste des juges.")


if __name__ == "__main__":
    main()