# -*- coding: utf-8 -*-
"""
Planning Juges équilibré - Crossfit Amiens
Version 5 : Priorité 2 heats consécutifs puis 2 repos (flexible si impossible)
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os
from collections import defaultdict
import traceback
import re

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄 - Version 5 (2 on / 2 off priorité)")


# ---------------------------
# Utilitaires PDF
# ---------------------------
def generate_pdf_tableau(planning: dict) -> FPDF:
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


def generate_heat_pdf(planning: dict) -> FPDF:
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


# ---------------------------
# Helpers
# ---------------------------
def extract_heat_number(heat_str):
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


# ---------------------------
# Attribution V5 (priorité 2 on / 2 off, flexible si impossible)
# ---------------------------
def assign_judges_equitable(schedule: pd.DataFrame, judges: list, disponibilites: dict, rotation: int):
    """
    Priorité stricte : 2 heats consécutifs puis 2 de repos si possible.
    Si impossible, autorise 1 ou 3 (ou plus si nécessaire) en minimisant ces cas.
    Garantit qu'une ligne = un juge (si WOD a au moins un juge disponible).
    """
    # init planning vide
    planning = {j: [] for j in judges}

    # préparer et trier les heats globalement (ordre chronologique)
    df = schedule.copy()
    df['heat_num'] = df['Heat #'].apply(extract_heat_number)

    # Convertir times si possible pour un tri fiable
    for col in ['Heat Start Time', 'Heat End Time']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col]).dt.time
            except Exception:
                pass

    # Nous construisons une liste de heats (globalement ordonnée)
    df = df.sort_values(['Heat Start Time', 'Workout', 'heat_num', 'Lane']).reset_index(drop=True)

    heats = []
    for (wod, heat_num, start, end), group in df.groupby(['Workout', 'heat_num', 'Heat Start Time', 'Heat End Time']):
        heats.append({
            'wod': wod,
            'heat_num': heat_num,
            'start': start,
            'end': end,
            'rows': group.to_dict('records')
        })

    total_heats = len(heats)
    if total_heats == 0:
        return planning

    # Etat par juge pour gérer séquences
    state = {}
    for j in judges:
        state[j] = {
            'mode': 'idle',       # 'idle' / 'on' / 'rest'
            'remaining_on': 0,    # si en on-block, combien de heats restants (incluant le courant)
            'remaining_rest': 0,  # si en repos, combien de heats restants
            'last_heat_idx': -999,# dernier index de heat servi
            'assign_count': 0     # nombre de lignes assignées
        }

    # Contrôles prioritaires
    PREFERRED_ON = 2  # priorité : 2 heats consécutifs
    PREFERRED_REST = 2

    # On parcourt les heats chronologiquement
    for idx, heat in enumerate(heats):
        wod = heat['wod']
        rows = heat['rows']

        # liste des juges déjà utilisés pour ce heat (on interdit duplication)
        used_in_this_heat = set()

        # pour chaque ligne (lane) du heat, on doit assigner un juge
        for row in rows:
            # candidats = juges disponibles pour ce WOD
            candidats = [j for j in disponibilites.get(wod, []) if j in judges and j not in used_in_this_heat]
            if not candidats:
                # aucun juge dispo pour ce WOD : on ne peut pas assigner -> on loggue un fallback
                # (dans main on empêche cette situation avant de lancer)
                continue

            # Construire scoring des candidats
            scored = []
            for j in candidats:
                s = state[j]
                score = 0
                reasons = []

                # 1) Priorité très forte : juge qui était sur le heat précédent (idx-1)
                #    et qui est en 'on' avec remaining_on > 0 -> on veut le garder pour obtenir la 2ème heat consécutive.
                if s['last_heat_idx'] == idx - 1 and s['mode'] == 'on' and s['remaining_on'] > 0:
                    score -= 100  # très prioritaire
                    reasons.append("continue_on_block")

                # 2) Si juge est idle or just finished rest (remaining_rest==0) -> peut démarrer un bloc on
                if s['mode'] == 'idle' or (s['mode'] == 'rest' and s['remaining_rest'] == 0):
                    score -= 50
                    reasons.append("ready_to_start")

                # 3) Désavantage pour ceux encore en repos (mais with remaining_rest > 0)
                if s['mode'] == 'rest' and s['remaining_rest'] > 0:
                    score += 50
                    reasons.append("still_resting")

                # 4) Charge actuelle : on préfère juges moins chargés
                score += s['assign_count'] * 2

                # 5) Si juge a déjà servi plusieurs heats consécutifs récemment (remaining_on small) on favorise repos
                # (no extra adjustment here; handled by remaining_on/remaining_rest logic)

                scored.append((score, j, reasons))

            # Trier par score (plus petit = mieux), puis par assign_count pour briser égalité
            scored.sort(key=lambda x: (x[0], state[x[1]]['assign_count'], x[1]))

            # Choix du meilleur candidat
            best_score, best_j, best_reasons = scored[0]

            # Si le meilleur candidat est en repos strict et qu'il reste d'autres candidats "acceptables", essayer le suivant
            # (on veut minimiser assignations pendant rest si possible)
            if 'still_resting' in best_reasons and len(scored) > 1:
                # chercher un candidat non 'still_resting'
                alt = None
                for sc in scored:
                    if 'still_resting' not in sc[2]:
                        alt = sc
                        break
                if alt is not None:
                    best_score, best_j, best_reasons = alt

            # Enfin, on attribue best_j à cette ligne
            assigned_j = best_j
            planning[assigned_j].append({
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
            used_in_this_heat.add(assigned_j)

            # Mettre à jour l'état du juge
            s = state[assigned_j]

            # Si il venait du heat précédent (idx-1) et était en 'on' -> on continue le block
            if s['last_heat_idx'] == idx - 1 and s['mode'] == 'on' and s['remaining_on'] > 0:
                s['remaining_on'] -= 1
                s['last_heat_idx'] = idx
                s['assign_count'] += 1
                # si remaining_on devient 0 -> passer en rest
                if s['remaining_on'] == 0:
                    s['mode'] = 'rest'
                    s['remaining_rest'] = PREFERRED_REST
            else:
                # Il démarre (ou redémarre) un bloc on ici
                # Si il était en rest avec remaining_rest>0, on diminue remaining_rest (exception case)
                if s['mode'] == 'rest' and s['remaining_rest'] > 0:
                    # On autorise cette affectation uniquement si aucun autre candidat "valide" existait.
                    # Ici on considère que l'algorithme de scoring a choisi best_j en dernier recours.
                    # Réinitialiser la séquence : on le considère démarrant un nouveau bloc on (flexibilité)
                    s['mode'] = 'on'
                    s['remaining_on'] = PREFERRED_ON - 1  # on consomme 1 heat maintenant
                    s['remaining_rest'] = 0
                    s['last_heat_idx'] = idx
                    s['assign_count'] += 1
                    # s['remaining_on'] peut être 1 (s'il doit encore faire 1 heat consecutif)
                    if s['remaining_on'] == 0:
                        s['mode'] = 'rest'
                        s['remaining_rest'] = PREFERRED_REST
                else:
                    # cas normal : démarre un bloc on
                    s['mode'] = 'on'
                    s['remaining_on'] = PREFERRED_ON - 1  # après avoir pris ce heat
                    s['last_heat_idx'] = idx
                    s['assign_count'] += 1
                    if s['remaining_on'] == 0:
                        s['mode'] = 'rest'
                        s['remaining_rest'] = PREFERRED_REST

        # Fin du heat : décrémenter remaining_rest pour tous les juges en repos (car un heat a passé)
        for j in judges:
            if state[j]['mode'] == 'rest' and state[j]['remaining_rest'] > 0:
                # Si le juge n'a été attribué sur le heat courant, alors ce heat compte comme repos
                if state[j]['last_heat_idx'] != idx:
                    state[j]['remaining_rest'] -= 1
                    if state[j]['remaining_rest'] <= 0:
                        state[j]['mode'] = 'idle'
                        state[j]['remaining_rest'] = 0

    # Post-check : s'assurer que chaque ligne a été assignée ; si non, faire une passe de rattrapage (fallback)
    # Construire index de toutes les lignes pour vérifier
    all_rows = df.to_dict('records')
    assigned_pairs = set()
    for j, rows in planning.items():
        for r in rows:
            key = (r['wod'], r['heat_num'], r['lane'])
            assigned_pairs.add(key)

    # Pour toute ligne non assignée, assigner au juge le moins chargé disponible pour le WOD (respect interdit double dans même heat)
    for row in all_rows:
        key = (row['Workout'], extract_heat_number(row['Heat #']), row['Lane'])
        if key not in assigned_pairs:
            wod = row['Workout']
            candidats = [j for j in disponibilites.get(wod, []) if j in judges]
            # filtrer ceux déjà sur ce heat
            candidats = [j for j in candidats if not any(c['wod'] == wod and c['heat_num'] == key[1] for c in planning[j])]
            if not candidats:
                # dernier recours : tous les juges mais pas ceux déjà sur ce heat
                candidats = [j for j in judges if not any(c['wod'] == wod and c['heat_num'] == key[1] for c in planning[j])]
            # choisir le moins chargé
            candidats.sort(key=lambda j: len(planning[j]))
            if candidats:
                chosen = candidats[0]
                planning[chosen].append({
                    'wod': wod,
                    'lane': row['Lane'],
                    'athlete': row['Competitor'],
                    'division': row['Division'],
                    'location': row['Workout Location'],
                    'start': row['Heat Start Time'],
                    'end': row['Heat End Time'],
                    'heat': row['Heat #'],
                    'heat_num': key[1]
                })
                assigned_pairs.add(key)

    return planning


# ---------------------------
# MAIN STREAMLIT
# ---------------------------
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

        st.header("Paramètres de rotation")
        rotation = st.radio("Mode d'attribution",
                           options=[1, 2],
                           index=1,
                           format_func=lambda x: "1 heat consécutif max" if x == 1 else "2 heats consécutifs + 2 repos")

        st.markdown("**Remarque** : la logique V5 priorise strictement 2-on/2-off quand c'est possible. "
                    "Si le planning/les disponibilités empêchent cela, le moteur pourra exceptionnellement attribuer 1 ou 3 consécutifs.")

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')

            required_columns = ['Workout', 'Lane', 'Competitor', 'Division',
                                'Workout Location', 'Heat Start Time', 'Heat End Time', 'Heat #']
            if not all(col in schedule.columns for col in required_columns):
                st.error("Erreur: Colonnes manquantes.")
                st.write("Colonnes trouvées:", list(schedule.columns))
                return

            # Nettoyage / preview
            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'] = schedule['Workout'].fillna("WOD Inconnu")
            schedule['heat_num'] = schedule['Heat #'].apply(extract_heat_number)

            st.subheader("Aperçu du planning chargé")
            st.dataframe(schedule[['Workout', 'Heat #', 'Lane', 'Competitor', 'Heat Start Time']].head(30))

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
                # vérifier qu'aucun WOD n'est sans juge
                wods_sans_juge = [w for w, j in disponibilites.items() if not j]
                if wods_sans_juge:
                    st.error(f"Les WODs suivants n'ont aucun juge sélectionné : {', '.join(wods_sans_juge)}")
                    return

                planning = assign_judges_equitable(schedule, judges, disponibilites, rotation)

                # Afficher résumé
                st.subheader("📊 Résumé des assignations")
                totals = {j: len(planning[j]) for j in judges}
                total_lines = sum(totals.values())
                cible = total_lines // len(judges) if len(judges) > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Assignations par juge :")
                    for j in sorted(totals, key=lambda x: totals[x], reverse=True):
                        st.write(f"{j}: {totals[j]} créneaux (écart {totals[j]-cible:+d})")

                with col2:
                    # compter séquences consécutives réelles par juge
                    seqs = {}
                    for j in judges:
                        seq_count = 0
                        entries = sorted(planning[j], key=lambda x: (x['wod'], extract_heat_number(x['heat'])))
                        for k in range(len(entries) - 1):
                            if entries[k]['wod'] == entries[k+1]['wod'] and extract_heat_number(entries[k+1]['heat']) - extract_heat_number(entries[k]['heat']) == 1:
                                seq_count += 1
                        seqs[j] = seq_count
                    st.write("Séquences consécutives par juge (estimation) :")
                    for j in sorted(seqs, key=lambda x: seqs[x], reverse=True):
                        st.write(f"{j}: {seqs[j]}")

                # Générer PDFs
                pdf_juges = generate_pdf_tableau(planning)
                pdf_heats = generate_heat_pdf(planning)

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
        st.info("Veuillez uploader le fichier de planning et saisir la liste des juges.")


if __name__ == "__main__":
    main()
