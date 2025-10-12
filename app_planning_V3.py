# -*- coding: utf-8 -*-
"""
Optimized Heat/Judge assignment
"""

import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile, os
from collections import defaultdict
import traceback

st.set_page_config(page_title="Planning Juges by Crossfit Amiens 🦄", layout="wide")
st.title("Planning Juges by Crossfit Amiens 🦄")

# ---------------- PDF Functions ---------------- #

def generate_pdf_tableau(planning):
    pdf = FPDF(orientation='P')
    pdf.set_auto_page_break(True, margin=15)
    for juge, creneaux in planning.items():
        if not creneaux: continue
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0,10,"Nom de la compétition",0,1,'C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0,10,f"Planning: {juge}",0,1,'C')
        pdf.ln(10)
        col_widths = [30,10,15,50,25,40]
        headers = ["Heure","Lane","WOD","Athlete","Division","Emplacement"]
        pdf.set_fill_color(211,211,211)
        pdf.set_font("Arial",'B',10)
        for h,w in zip(headers,col_widths):
            pdf.cell(w,10,h,1,0,'C',fill=True)
        pdf.ln()
        pdf.set_font("Arial",'',9)
        colors = [(255,255,255),(240,240,240)]
        for i,c in enumerate(creneaux):
            pdf.set_fill_color(*colors[i%2])
            start = c['start'].strftime('%H:%M') if hasattr(c['start'],'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'],'strftime') else c['end']
            data = [f"{start} - {end}",str(c['lane']),c['wod'],c['athlete'],c['division'],c['location']]
            for val,w in zip(data,col_widths):
                pdf.cell(w,10,str(val),1,0,'C',fill=True)
            pdf.ln()
        pdf.ln(5)
        pdf.set_font("Arial",'I',10)
        total_wods = len({c['wod'] for c in creneaux})
        pdf.cell(0,8,f"Total: {len(creneaux)} creneaux sur {total_wods} WODs",0,1)
    return pdf

def generate_heat_pdf(planning):
    heat_map = defaultdict(lambda: defaultdict(str))
    for juge, creneaux in planning.items():
        for c in creneaux:
            start = c['start'].strftime('%H:%M') if hasattr(c['start'],'strftime') else c['start']
            end = c['end'].strftime('%H:%M') if hasattr(c['end'],'strftime') else c['end']
            key = (c['wod'],c['heat'],start,end,c['location'])
            heat_map[key][int(c['lane'])] = juge
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_font("Arial",'',10)
    heats = sorted(heat_map.items(), key=lambda x: (x[0][0], x[0][1]))
    for i in range(0,len(heats),2):
        pdf.add_page()
        col_width=90; row_height=8; spacing=15
        for j in range(2):
            if i+j>=len(heats): break
            (wod,heat,start,end,loc), lanes = heats[i+j]
            x = 10 + j*(col_width+spacing)
            pdf.set_font("Arial",'B',10)
            pdf.set_xy(x,15)
            pdf.cell(col_width,row_height,f"WOD: {wod} - Heat {heat}",1,0,'C',fill=True)
            pdf.set_font("Arial",'',9)
            pdf.set_xy(x,15+row_height)
            pdf.cell(col_width,row_height,f"Heure: {start} - {end}",1,0,'C')
            pdf.set_xy(x,15+2*row_height)
            pdf.cell(col_width,row_height,f"Emplacement: {loc}",1,0,'C')
            pdf.set_font("Arial",'B',9)
            pdf.set_xy(x,15+3*row_height)
            pdf.cell(col_width/2,row_height,"Lane",1,0,'C',fill=True)
            pdf.cell(col_width/2,row_height,"Juge",1,0,'C',fill=True)
            pdf.set_font("Arial",'',9)
            for k,lane in enumerate(sorted(lanes)):
                y = 15 + (4+k)*row_height
                pdf.set_xy(x,y)
                pdf.cell(col_width/2,row_height,str(lane),1,0,'C')
                pdf.cell(col_width/2,row_height,lanes[lane],1,0,'C')
    return pdf

# ---------------- MAIN APP ---------------- #

def main():
    with st.sidebar:
        st.header("Importer le planning")
        schedule_file = st.file_uploader("Planning (Excel)", type=["xlsx"])
        st.header("Juges")
        input_method = st.radio("Méthode de saisie des juges", ["CSV","Manuelle"], index=0)
        judges=[]
        if input_method=="CSV":
            judges_file = st.file_uploader("Liste des juges (CSV)", type=["csv"])
            if judges_file:
                judges = pd.read_csv(judges_file, header=None, encoding='latin1')[0].dropna().tolist()
        else:
            text = st.text_area("Saisir les juges (1 par ligne)", value="Juge 1\nJuge 2\nJuge 3", height=150)
            judges = [j.strip() for j in text.split("\n") if j.strip()]
        st.header("Rotation")
        rotation_heats = st.selectbox("Nombre de heats consécutifs par juge", [1,2], index=0)

    if schedule_file and judges:
        try:
            schedule = pd.read_excel(schedule_file, engine='openpyxl')
            st.subheader("Aperçu")
            st.dataframe(schedule.head())
            required = ['Workout','Heat #','Lane','Competitor','Division','Workout Location','Heat Start Time','Heat End Time']
            if not all(c in schedule.columns for c in required):
                st.error("Colonnes manquantes"); return
            schedule = schedule[~schedule['Competitor'].str.contains('EMPTY LANE', na=False)]
            schedule['Workout'].fillna("WOD Inconnu", inplace=True)
            wods = sorted(schedule['Workout'].unique())
            st.header("Sélection des juges par WOD")
            disponibilites = {}
            cols = st.columns(3)
            for i,wod in enumerate(wods):
                with cols[i%3]:
                    with st.expander(wod):
                        select_all = st.checkbox(f"Tout sélectionner", key=f"select_all_{wod}")
                        if select_all: sel = judges
                        else: sel = st.multiselect(f"Sélection pour {wod}", judges, key=f"wod_{wod}")
                        disponibilites[wod] = sel

            if st.button("Générer planning"):
                planning = {j:[] for j in judges}
                for wod in wods:
                    data_wod = schedule[schedule['Workout']==wod].sort_values(['Heat #','Lane'])
                    heats = sorted(data_wod['Heat #'].unique())
                    available_judges = disponibilites[wod]
                    if not available_judges: st.error(f"Aucun juge pour {wod}"); continue
                    heat_idx=0
                    while heat_idx<len(heats):
                        for i,j in enumerate(available_judges):
                            for r in range(rotation_heats):
                                if heat_idx+r>=len(heats): break
                                heat_num = heats[heat_idx+r]
                                lines = data_wod[data_wod['Heat #']==heat_num]
                                # Assigner un juge différent pour chaque ligne
                                judge_idx=0
                                for _, row in lines.iterrows():
                                    planning[available_judges[judge_idx%len(available_judges)]].append({
                                        'wod':wod,'heat':heat_num,'lane':row['Lane'],
                                        'athlete':row['Competitor'],'division':row['Division'],
                                        'location':row['Workout Location'],'start':row['Heat Start Time'],'end':row['Heat End Time']
                                    })
                                    judge_idx+=1
                        heat_idx += rotation_heats*len(available_judges)

                pdf_j = generate_pdf_tableau({k:v for k,v in planning.items() if v})
                pdf_h = generate_heat_pdf({k:v for k,v in planning.items() if v})

                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as f:
                    pdf_j.output(f.name)
                    with open(f.name,"rb") as ff:
                        st.download_button("Télécharger planning par juge", ff, "planning_juges.pdf","application/pdf")
                    os.unlink(f.name)

                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as f:
                    pdf_h.output(f.name)
                    with open(f.name,"rb") as ff:
                        st.download_button("Télécharger planning par heat", ff, "planning_heats.pdf","application/pdf")
                    os.unlink(f.name)

                st.success("PDF générés avec succès!")
                st.header("Récapitulatif")
                for juge, c in planning.items():
                    if c:
                        with st.expander(f"{juge} ({len(c)} créneaux)"):
                            st.table(pd.DataFrame(c))

        except Exception:
            st.error("Erreur:")
            st.code(traceback.format_exc())
    else:
        st.info("Uploader le fichier et saisir les juges pour commencer.")

if __name__=="__main__":
    main()
