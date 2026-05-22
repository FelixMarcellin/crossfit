# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 8.0 : Choix flexible du système on/off
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
st.title("🏋️‍♂️ Planning Juges - Crossfit Amiens 🦄 (f.marcellin)")


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
# PDF AVEC FILIGRANE
# ========================
class WatermarkPDF(FPDF):
    def __init__(self, logo_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logo_path = logo_path

    def add_watermark(self):
        """Ajoute un filigrane centré AU-DESSUS du contenu"""
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                logo_width = 120
                x = (210 - logo_width) / 2
                y = (297 - logo_width) / 2

                if hasattr(self, "set_alpha"):
                    self.set_alpha(0.5)

                self.image(self.logo_path, x=x, y=y, w=logo_width)

                if hasattr(self, "set_alpha"):
                    self.set_alpha(1)

            except Exception as e:
                print(f"Erreur filigrane : {e}")

# ========================
# PDF PAR JUGE (LARGEUR OPTIMISÉE)
# ========================
def generate_pdf_tableau(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    """Génère le PDF par juge avec largeur optimisée pour toute la feuille"""
    pdf = WatermarkPDF(logo_path=logo_path, orientation='P')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Largeur totale disponible (A4 portrait : 210mm - marges)
    total_width = 180
    
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

        creneaux = sorted(creneaux, key=lambda c: parse_time(c.get('start', '')))

        pdf.add_page()
        pdf.add_watermark()
        
        # En-tête
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, clean_text(competition_name), 0, 1, 'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Planning : {clean_text(juge)}", 0, 1, 'C')
        pdf.ln(8)

        # En-têtes de colonnes avec largeurs optimisées
        headers = ["Heure", "Lane", "WOD", "Heat", "Athlete", "Division"]
        
        # Largeurs proportionnelles à l'importance des colonnes
        col_widths = [
            total_width * 0.12,  # Heure : 12%
            total_width * 0.08,  # Lane : 8%
            total_width * 0.12,  # WOD : 12%
            total_width * 0.10,  # Heat : 10%
            total_width * 0.40,  # Athlete : 40% (plus large)
            total_width * 0.18   # Division : 18%
        ]
        
        # Convertir en entiers
        col_widths = [int(w) for w in col_widths]
        
        pdf.set_fill_color(211, 211, 211)
        pdf.set_font("Arial", 'B', 10)
        
        # Dessiner la ligne d'en-tête
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 8, h, 1, 0, 'C', fill=True)
        pdf.ln()
        
        # Contenu du tableau
        pdf.set_font("Arial", '', 9)
        row_colors = [(255, 255, 255), (240, 240, 240)]
        
        for i, c in enumerate(creneaux):
            # Vérifier si on dépasse la page
            if pdf.get_y() > 260:
                pdf.add_page()
                pdf.add_watermark()
                # Ré-afficher les en-têtes
                pdf.set_font("Arial", 'B', 10)
                pdf.set_fill_color(211, 211, 211)
                for h, w in zip(headers, col_widths):
                    pdf.cell(w, 8, h, 1, 0, 'C', fill=True)
                pdf.ln()
                pdf.set_font("Arial", '', 9)
            
            pdf.set_fill_color(*row_colors[i % 2])
            
            # Formatage des données
            start = clean_text(str(c['start']))[:5]
            end = clean_text(str(c['end']))[:5]
            
            # Tronquer moins agressivement grâce à la largeur augmentée
            athlete_name = clean_text(str(c['athlete']))
            if len(athlete_name) > 45:
                athlete_name = athlete_name[:45] + "..."
                
            division_name = clean_text(str(c['division']))
            if len(division_name) > 25:
                division_name = division_name[:25] + "..."
            
            wod_name = clean_text(str(c['wod']))
            if len(wod_name) > 15:
                wod_name = wod_name[:15] + "..."
            
            heat_name = clean_text(str(c['heat']))
            if len(heat_name) > 12:
                heat_name = heat_name[:12] + "..."
            
            vals = [
                f"{start}-{end}",
                clean_text(str(c['lane'])),
                wod_name,
                heat_name,
                athlete_name,
                division_name
            ]
            
            # Dessiner chaque cellule
            for v, w in zip(vals, col_widths):
                pdf.cell(w, 7, v, 1, 0, 'C', fill=True)
            pdf.ln()

        # Pied de page
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 9)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0, 8, f"Total : {len(creneaux)} créneaux sur {total_wods} WODs", 0, 1)

    return pdf


# ========================
# PDF PAR HEAT (4 tableaux par page)
# ========================
def generate_heat_pdf(planning: dict, competition_name: str, logo_path=None) -> FPDF:
    """Génère le PDF par heat (4 tableaux par page - grand espacement)"""
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            key = (c['wod'], c['heat'], c['start'], c['end'], c['location'])
            heat_map[key][int(c['lane'])] = juge

    pdf = WatermarkPDF(logo_path=logo_path)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 10)

    heats = sorted(heat_map.items(), key=lambda x: (x[0][0], x[0][1]))

    for i in range(0, len(heats), 4):
        pdf.add_page()
        pdf.add_watermark()

        # Nom de la compétition
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_text(competition_name), 0, 1, 'C')
        pdf.ln(4)

        # Largeurs adaptées à la largeur totale
        col_width = 90  # Augmenté pour utiliser plus d'espace
        row_height = 8
        spacing_x = 10
        
        current_y = 25
        
        page_heats = heats[i:i+4]
        
        for j, ((wod, heat, start, end, loc), lanes) in enumerate(page_heats):

            col = j % 2
    
            x = 10 + col * (col_width + spacing_x)
            y_start = current_y
    
            # Header bloc
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

            if col == 1 or j == len(heats[i:i+4]) - 1:
                block_height = (len(lanes) + 2) * row_height + 12
                current_y += block_height

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
# ATTRIBUTION ÉQUILIBRÉE AVEC SYSTÈME ON/OFF FLEXIBLE
# ========================
def assign_judges_equitable(schedule, judges, disponibilites, rotation_config):
    """Attribution équilibrée avec système on/off flexible"""
    planning = {j: [] for j in judges}
    df = schedule.copy()
    df['heat_num'] = df['Heat #'].apply(extract_heat_number)
    df = df.sort_values(['Heat Start Time', 'Workout', 'heat_num', 'Lane']).reset_index(drop=True)

    heats = []
    for (wod, heat_num, start, end), g in df.groupby(['Workout', 'heat_num', 'Heat Start Time', 'Heat End Time']):
        heats.append({'wod': wod, 'heat_num': heat_num, 'start': start, 'end': end, 'rows': g.to_dict('records')})

    n_judges = len(judges)
    target = len(df) // n_judges if n_judges > 0 else 0

    # Configuration flexible du système on/off
    ON = rotation_config['on']
    REST = rotation_config['off']
    
    state = {j: {'last': -999, 'on': 0, 'rest': 0, 'count': 0} for j in judges}

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
                # Priorité pour continuer un bloc en cours
                if s['last'] == idx - 1 and s['on'] > 0:
                    score -= 100  # continuer bloc
                # Pénalité pour ceux en repos
                if s['rest'] > 0:
                    score += 50  # éviter ceux en repos
                # Équilibrage du nombre total de créneaux
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

            # Mise à jour état avec système flexible
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

        # décrémenter repos pour tous les juges
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

        st.header("⚙️ Configuration du roulement")
        rotation_system = st.selectbox(
            "Système de roulement",
            options=[
                {"name": "1-on/1-off", "on": 1, "off": 1},
                {"name": "2-on/2-off", "on": 2, "off": 2},
                {"name": "3-on/3-off", "on": 3, "off": 3},
                {"name": "2-on/1-off", "on": 2, "off": 1},
                {"name": "3-on/2-off", "on": 3, "off": 2}
            ],
            format_func=lambda x: x["name"]
        )
        
        st.info(f"🔄 Système sélectionné : {rotation_system['name']}")
        st.header("🖼️ Logo filigrane")
        logo_file = st.file_uploader("Uploader un logo", type=["png", "jpg", "jpeg"])

        logo_path = None
        if logo_file:
            temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_logo.write(logo_file.read())
            temp_logo.close()
            logo_path = temp_logo.name


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
                planning = assign_judges_equitable(schedule, judges, disponibilites, rotation_system)

                st.subheader("📊 Équilibre des assignations")
                counts = {j: len(planning[j]) for j in judges}
                total = sum(counts.values())
                target = total // len(judges)
                
                # Afficher le système utilisé
                st.info(f"**Système de roulement :** {rotation_system['name']}")
                
                for j in sorted(counts, key=counts.get, reverse=True):
                    ecart = counts[j] - target
                    emoji = "✅" if abs(ecart) <= 1 else "⚠️"
                    st.write(f"{emoji} {j}: {counts[j]} créneaux ({ecart:+d})")

                try:
                    pdf_juges = generate_pdf_tableau(planning, competition_name, logo_path)
                    pdf_heats = generate_heat_pdf(planning, competition_name, logo_path)
                    
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