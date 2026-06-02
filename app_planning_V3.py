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

    ON_TARGET = rotation_config["on"]
    OFF_TARGET = rotation_config["off"]

    # État individuel de chaque juge
    state = {
        j: {
            "status": "AVAILABLE",  # AVAILABLE, ON, OFF
            "history": [],
            "consecutive_on": 0,    # Nombre de Heats consécutifs arbitrés
            "consecutive_off": 0,   # Nombre de Heats consécutifs de repos
            "total_count": 0,       # Compteur global pour équilibrer
            "current_lane": None    # Ligne actuellement assignée au juge
        } for j in judges
    }

    # Dictionnaire des affectations de lignes : { lane_number: nom_du_juge }
    lane_assignments = {}

    for heat in heats:
        wod = heat["wod"]
        heat_label = f"Heat {heat['heat_num']} ({heat['start']})"
        dispo = disponibilites.get(wod, judges)
        if not dispo: dispo = judges

        # --- ÉTAPE 1 : RESTE DE FORCE OU DÉPART EN REPOS ---
        for j in judges:
            # Sécurité anti-tunnel : Si le juge a atteint la limite max (ON_TARGET + 1), REPOS FORCÉ
            if state[j]["consecutive_on"] >= (ON_TARGET + 1):
                state[j]["status"] = "OFF"
                state[j]["consecutive_on"] = 0
                state[j]["consecutive_off"] = 0
                # On libère sa ligne
                if state[j]["current_lane"] in lane_assignments:
                    if lane_assignments[state[j]["current_lane"]] == j:
                        del lane_assignments[state[j]["current_lane"]]
                state[j]["current_lane"] = None

            # Fin de repos réglementaire
            elif state[j]["status"] == "OFF" and state[j]["consecutive_off"] >= OFF_TARGET:
                state[j]["status"] = "AVAILABLE"
                state[j]["consecutive_off"] = 0

        # Liste des juges sur le pont pour ce Heat précis
        judges_working_this_heat = set()

        # --- ÉTAPE 2 : ASSIGNATION DES LIGNES POUR CE HEAT ---
        for row in heat["rows"]:
            lane = str(int(float(row["Lane"])))
            chosen_judge = None

            # 1. Règle d'or : On garde le juge qui possède déjà cette ligne si disponible
            if lane in lane_assignments:
                prev_judge = lane_assignments[lane]
                if (prev_judge in dispo and 
                    state[prev_judge]["status"] != "OFF" and 
                    prev_judge not in judges_working_this_heat and 
                    state[prev_judge]["consecutive_on"] < (ON_TARGET + 1)):
                    
                    chosen_judge = prev_judge

            # 2. Si la ligne est vacante (ou juge en repos), on cherche un nouveau profil
            if chosen_judge is None:
                # Candidats : dispo sur le WOD, pas encore pris sur ce heat, et pas en OFF
                candidates = [j for j in judges if j in dispo and j not in judges_working_this_heat and state[j]["status"] in ["AVAILABLE", "ON"]]
                
                # Priorité absolue : 
                #  a) Ceux qui sortent de repos (AVAILABLE) plutôt que ceux déjà fatigués (ON)
                #  b) À statut égal, celui qui a le moins bossé aujourd'hui (total_count)
                candidates = sorted(candidates, key=lambda j: (0 if state[j]["status"] == "AVAILABLE" else 1, state[j]["total_count"]))
                
                # Si pénurie totale de juges (cas critique), on réquisitionne en priorité les "AVAILABLE" hors-dispo puis les "OFF"
                if not candidates:
                    candidates = [j for j in judges if j not in judges_working_this_heat]
                    candidates = sorted(candidates, key=lambda j: (0 if state[j]["status"] == "AVAILABLE" else 1, state[j]["consecutive_on"], state[j]["total_count"]))

                if candidates:
                    chosen_judge = candidates[0]
                    
                    # Si ce juge avait une ancienne ligne, on la nettoie
                    old_lane = state[chosen_judge]["current_lane"]
                    if old_lane and old_lane in lane_assignments and lane_assignments[old_lane] == chosen_judge:
                        del lane_assignments[old_lane]
                    
                    # On l'installe sur sa nouvelle ligne
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

        # --- ÉTAPE 3 : MISE À JOUR DES COMPTEURS POST-HEAT ---
        for j in judges:
            if j in judges_working_this_heat:
                state[j]["consecutive_on"] += 1
                state[j]["consecutive_off"] = 0
                state[j]["total_count"] += 1
            else:
                # Le juge ne travaille pas sur ce heat
                if state[j]["status"] == "ON":
                    # C'était un "trou" dans son bloc d'activité (ex: pas d'athlète dans sa lane)
                    # On considère que son bloc continue s'il n'a pas eu son quota de repos complet
                    state[j]["consecutive_on"] += 0.5 # Compte comme une demi-charge ou une pause légère
                    if state[j]["consecutive_on"] >= ON_TARGET:
                        state[j]["status"] = "OFF"
                        state[j]["consecutive_on"] = 0
                        state[j]["consecutive_off"] = 1
                elif state[j]["status"] == "OFF":
                    state[j]["consecutive_off"] += 1
                else:
                    # Un juge AVAILABLE qui attend accumule du repos
                    state[j]["consecutive_off"] += 1

    return planning, warnings_list
