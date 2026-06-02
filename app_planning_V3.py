def assign_judges_equitable(schedule, judges, disponibilites, rotation_config):
    planning = {j: [] for j in judges}
    warnings_list = []

    # Tri chronologique exact des Heats
    df = schedule.copy()
    df["heat_num"] = df["Heat #"].apply(extract_heat_number)
    df = df.sort_values(["Heat Start Time", "heat_num", "Lane"]).reset_index(drop=True)

    heats = []
    for (wod, heat_num, start, end), g in df.groupby(["Workout", "heat_num", "Heat Start Time", "Heat End Time"]):
        heats.append({
            "wod": wod, "heat_num": heat_num, "start": start, "end": end,
            "rows": sorted(g.to_dict("records"), key=lambda r: int(float(r["Lane"])))
        })

    ON_TARGET = rotation_config["on"]   # ex: 3
    OFF_TARGET = rotation_config["off"] # ex: 3
    MAX_ALLOWED = ON_TARGET + 1         # Sécurité absolue : 4 Heats max !

    # État individuel de chaque juge
    state = {
        j: {
            "status": "AVAILABLE",       # AVAILABLE, ON, OFF
            "heats_sans_pause": 0,       # LE VERROU : Compte TOUS les heats depuis le dernier vrai repos
            "consecutive_off": 0,        # Nombre de Heats consécutifs de repos réels
            "total_count": 0,            # Compteur global pour équilibrer
            "current_lane": None         # Ligne actuellement assignée
        } for j in judges
    }

    # Dictionnaire des affectations de lignes : { lane_number: nom_du_juge }
    lane_assignments = {}

    for heat in heats:
        wod = heat["wod"]
        heat_label = f"Heat {heat['heat_num']} ({heat['start']})"
        dispo = disponibilites.get(wod, judges)
        if not dispo: dispo = judges

        # --- ÉTAPE 1 : VÉRIFICATION DES BLOCAGES STRICTS ---
        for j in judges:
            # Si le juge a atteint ou dépassé la limite absolue (ex: 4 heats bossés ou enchaînés)
            # OU s'il a fait ses 3 heats et qu'on doit le couper
            if state[j]["heats_sans_pause"] >= MAX_ALLOWED or (state[j]["status"] == "ON" and state[j]["heats_sans_pause"] >= ON_TARGET):
                state[j]["status"] = "OFF"
                state[j]["heats_sans_pause"] = 0
                state[j]["consecutive_off"] = 0
                
                # Libération immédiate de sa ligne
                old_lane = state[j]["current_lane"]
                if old_lane and old_lane in lane_assignments and lane_assignments[old_lane] == j:
                    del lane_assignments[old_lane]
                state[j]["current_lane"] = None

            # Fin de repos réglementaire (il doit avoir fait TOUT son temps mort)
            elif state[j]["status"] == "OFF" and state[j]["consecutive_off"] >= OFF_TARGET:
                state[j]["status"] = "AVAILABLE"
                state[j]["consecutive_off"] = 0
                state[j]["heats_sans_pause"] = 0

        # Liste des juges qui vont travailler sur ce Heat précis
        judges_working_this_heat = set()

        # --- ÉTAPE 2 : ASSIGNATION DES LIGNES ---
        for row in heat["rows"]:
            lane = str(int(float(row["Lane"])))
            chosen_judge = None

            # 1. Priorité 1 : Conserver le juge sur SA ligne (S'IL N'EST PAS EN REPOS FORCÉ)
            if lane in lane_assignments:
                prev_judge = lane_assignments[lane]
                if (prev_judge in dispo and 
                    state[prev_judge]["status"] in ["AVAILABLE", "ON"] and 
                    prev_judge not in judges_working_this_heat and 
                    state[prev_judge]["heats_sans_pause"] < MAX_ALLOWED):
                    
                    chosen_judge = prev_judge

            # 2. Priorité 2 : Trouver un nouveau juge frais
            if chosen_judge is None:
                # Candidats valides : dispo, pas encore pris ce heat, et PAS en statut OFF
                candidates = [j for j in judges if j in dispo and j not in judges_working_this_heat and state[j]["status"] in ["AVAILABLE", "ON"]]
                
                # Tri : Priorité absolue à ceux qui reviennent de repos (AVAILABLE) et qui ont le moins de Heats au total
                candidates = sorted(candidates, key=lambda j: (
                    0 if state[j]["status"] == "AVAILABLE" else 1, 
                    state[j]["heats_sans_pause"], 
                    state[j]["total_count"]
                ))
                
                # Si pénurie critique sur le terrain (pas assez de juges dispos)
                if not candidates:
                    # On pioche d'abord chez les AVAILABLE hors-dispo, et en dernier recours absolu chez ceux en OFF
                    candidates = [j for j in judges if j not in judges_working_this_heat]
                    candidates = sorted(candidates, key=lambda j: (
                        0 if state[j]["status"] == "AVAILABLE" else 1,
                        state[j]["heats_sans_pause"],
                        state[j]["total_count"]
                    ))

                if candidates:
                    chosen_judge = candidates[0]
                    
                    # S'il change de ligne, on nettoie l'ancienne
                    old_lane = state[chosen_judge]["current_lane"]
                    if old_lane and old_lane in lane_assignments and lane_assignments[old_lane] == chosen_judge:
                        del lane_assignments[old_lane]
                    
                    # On l'affecte
                    lane_assignments[lane] = chosen_judge
                    state[chosen_judge]["current_lane"] = lane
                    state[chosen_judge]["status"] = "ON"
                else:
                    chosen_judge = "SANS JUGE"
                    warnings_list.append(f"⚠️ {wod} | {heat_label} | Couloir {lane} : Aucun juge disponible !")

            # Enregistrement du créneau
            if chosen_judge != "SANS JUGE":
                planning[chosen_judge].append({
                    "wod": clean_text(str(row["Workout"])),
                    "lane": lane,
                    "athlete": clean_text(str(row["Competitor"])),
                    "division": clean_text(str(row["Division"])),
                    "start": clean_text(str(row["Heat Start Time"])),
                    "end": clean_text(str(row["Heat End Time"])),
                    "heat": clean_text(str(row["Heat #"])),
                    "heat_num": heat["heat_num"]
                })
                judges_working_this_heat.add(chosen_judge)

        # --- ÉTAPE 3 : MISE À JOUR DES COMPTEURS (POST-HEAT) ---
        for j in judges:
            if j in judges_working_this_heat:
                state[j]["heats_sans_pause"] += 1
                state[j]["consecutive_off"] = 0
                state[j]["total_count"] += 1
            else:
                # Le juge n'a pas travaillé sur ce heat
                if state[j]["status"] == "ON":
                    # S'il était actif mais qu'il y a eu un trou/heat vide pour lui, 
                    # cela compte TOUT DE MÊME comme un heat écoulé dans son bloc d'activité
                    state[j]["heats_sans_pause"] += 1
                elif state[j]["status"] == "OFF":
                    state[j]["consecutive_off"] += 1
                else:
                    # Disponible (en attente)
                    state[j]["consecutive_off"] += 1

    return planning, warnings_list
