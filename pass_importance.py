import json
import os
import numpy as np
import pandas as pd

MATCHES_FILE = "matches_2_27.json"   # list of matches 
EVENTS_DIR = "."                     
LOOK_AHEAD = 5                       # max events to scan ahead inside same possession
TIME_AHEAD_SECONDS = 30              # only credit shots within 30s

# Pass classification
def classify_pass(row):
    dx = row['end_x'] - row['start_x']
    dy = row['end_y'] - row['start_y']
    length = np.sqrt(dx**2 + dy**2)

    if (row['start_y'] < 18 or row['start_y'] > 62) and (row['end_x'] > 102):
        return "Cross"
    elif row['start_x'] > 102 and dx < -5:
        return "Cutback"
    elif dx > 10 and row['end_x'] > 102 and abs(row['end_y']-40) < 20:
        return "Through Ball"
    elif abs(dy) > 30 and length > 30:
        return "Switch of Play"
    elif row['end_x'] > row['start_x'] + 0.25*(120 - row['start_x']):
        return "Progressive"
    elif dx > 5:
        return "Forward"
    elif dx < -5:
        return "Backward"
    else:
        return "Lateral"

def time_to_seconds(minute, second):
    try:
        return int(minute) * 60 + int(second)
    except Exception:
        return None

# Event analyzing
def extract_passes_from_match(match_file):
    with open(match_file, "r") as f:
        events = json.load(f)

    passes = []
    for i, e in enumerate(events):
        if e.get("type", {}).get("name") != "Pass":
            continue

        pass_obj = e.get("pass", {})
        # skip incomplete passes 
        if "outcome" in pass_obj:
            continue

        # Coordinates
        loc = e.get("location")
        end_loc = pass_obj.get("end_location")
        if not loc or not end_loc or len(loc) < 2 or len(end_loc) < 2:
            continue

        start_x, start_y = float(loc[0]), float(loc[1])
        end_x, end_y = float(end_loc[0]), float(end_loc[1])

        team_id = e.get("team", {}).get("id")
        team_name = e.get("team", {}).get("name")
        player_name = e.get("player", {}).get("name")
        possession_id = e.get("possession")
        m = e.get("minute", 0)
        s = e.get("second", 0)
        t0 = time_to_seconds(m, s)

        next_shot_xg = 0.0
        leads_to_shot = 0

        # Scan ahead inside same possession for a shot
        for j in range(1, LOOK_AHEAD + 1):
            if i + j >= len(events):
                break
            ev = events[i + j]

            # Stop if possession changes
            if ev.get("possession") != possession_id:
                break

            if ev.get("type", {}).get("name") == "Shot" and ev.get("team", {}).get("id") == team_id:
                if TIME_AHEAD_SECONDS is not None and t0 is not None:
                    t_ev = time_to_seconds(ev.get("minute", 0), ev.get("second", 0))
                    if t_ev is not None and (t_ev - t0) > TIME_AHEAD_SECONDS:
                        break
                shot_obj = ev.get("shot", {})
                xg = float(shot_obj.get("statsbomb_xg", 0.0))
                next_shot_xg = xg
                leads_to_shot = 1
                break

        passes.append({
            "minute": m,
            "second": s,
            "player": player_name,
            "team": team_name,
            "team_id": team_id,
            "possession": possession_id,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "next_shot_xg": next_shot_xg,
            "leads_to_shot": leads_to_shot
        })
    return passes

# Structural scoring 
def zone(x):
    # defensive: 0-40, middle: 40-80, final: 80-120
    if x < 40: return 0
    if x < 80: return 1
    return 2

def centrality_bonus(y):
    # Reward central targets more than touchline
    # center lane = [30,50], half-spaces = [20,30] & [50,60]
    if 30 <= y <= 50:
        return 1.0
    if 20 <= y < 30 or 50 < y <= 60:
        return 0.6
    if 10 <= y < 20 or 60 < y <= 70:
        return 0.3
    return 0.1

PASS_TYPE_WEIGHT = {
    "Through Ball":    1.00,
    "Cutback":         0.95,
    "Cross":           0.85,
    "Switch of Play":  0.70,
    "Progressive":     0.65,
    "Forward":         0.55,
    "Lateral":         0.20,
    "Backward":        0.10
}

def transition_value(start_x, end_x):
    # Bonus for area of pass
    zs = zone(start_x)
    ze = zone(end_x)
    dz = ze - zs
    if dz >= 2:            
        return 1.0
    elif dz == 1:          
        return 0.7
    elif dz == 0:
        return 0.2 if end_x > start_x + 5 else 0.05
    else:
        return -0.2 if dz == -1 else -0.35

def to_striker_bonus(end_x, end_y):
    if end_x >= 85 and abs(end_y - 40) <= 15:
        return 0.5
    if end_x >= 95 and abs(end_y - 40) <= 20:
        return 0.7
    return 0.0

def compute_structural_importance(row):
    # Pass type 
    ptype = row["pass_type"]
    base = PASS_TYPE_WEIGHT.get(ptype, 0.4)

    # Direction & progression
    dx = row["dx"]
    length = row["length"]
    prog = 0.0
    if dx > 0:
        # reward forward distance 
        prog = min(dx / 40.0, 0.6)  
    elif dx < 0:
        prog = max(dx / 60.0, -0.5)  


    trans = transition_value(row["start_x"], row["end_x"])

    final_third_bonus = 0.35 if row["final_third"] == 1 else 0.0
    box_entry_bonus   = 0.45 if row["box_entry"] == 1 else 0.0

    cent = centrality_bonus(row["end_y"]) * 0.4  

    # Special boosts
    special = 0.0
    if ptype == "Through Ball":
        special += 0.4
    if ptype == "Cutback":
        special += 0.35
    if ptype == "Cross" and row["end_x"] > 102:
        special += 0.25
    if ptype == "Switch of Play":
        if length >= 30 and abs(row["dy"]) >= 25:
            special += 0.25

    # "to striker" bonus
    striker = to_striker_bonus(row["end_x"], row["end_y"])

    lateral_pen = 0.0
    if ptype == "Lateral" and row["start_x"] < 60:
        lateral_pen = -0.15

    structural = (
        base + prog + trans + final_third_bonus + box_entry_bonus +
        cent + special + striker + lateral_pen
    )

    return max(structural, 0.0)

def normalize_series(s, eps=1e-9):
    return (s - s.min()) / (s.max() - s.min() + eps)


if __name__ == "__main__":
    # Load matches list
    with open(MATCHES_FILE, "r") as f:
        matches = json.load(f)

    all_passes = []

    # Loop through all matches
    for m in matches:
        match_id = m.get("match_id")
        event_file = os.path.join(EVENTS_DIR, f"{match_id}.json")
        if os.path.exists(event_file):
            all_passes.extend(extract_passes_from_match(event_file))
        else:
            print(f"⚠️ Event file missing for match {match_id}")

    # Build DataFrame
    df = pd.DataFrame(all_passes)
    if df.empty:
        raise ValueError("No passes extracted. Check your EVENTS_DIR path and files.")

    df["dx"] = df.end_x - df.start_x
    df["dy"] = df.end_y - df.start_y
    df["length"] = np.sqrt(df.dx**2 + df.dy**2)

    # Categorical features 
    df["progressive"] = (df.end_x > df.start_x + 0.25*(120 - df.start_x)).astype(int)
    df["final_third"] = (df.end_x > 80).astype(int)
    df["box_entry"] = ((df.end_x > 102) & (df.end_y.between(18,62))).astype(int)
    df["cross"] = (((df.start_y < 18) | (df.start_y > 62)) & (df.end_x > 102)).astype(int)
    df["through"] = ((df.dx > 10) & (df.end_x > 102) & (abs(df.end_y-40)<20)).astype(int)
    df["cutback"] = ((df.start_x > 102) & (df.dx < -5)).astype(int)

    # Pass type label
    df["pass_type"] = df.apply(classify_pass, axis=1)

    # Structural score 
    df["structural_score"] = df.apply(compute_structural_importance, axis=1)
    struct_norm = normalize_series(df["structural_score"])

    # Optional light xG bonus 
    xg_cap = 1  
    df["xg_norm"] = np.clip(df["next_shot_xg"] / xg_cap, 0, 1)

    # Final score
    df["importance_raw"] = 0.85 * struct_norm + 0.15 * df["xg_norm"]

    # Final normalization 
    df["importance_norm"] = normalize_series(df["importance_raw"])

    # Nice timestamp
    df["time"] = (df["minute"] * 60 + df["second"]).astype(int)

    # Print sample
    print("\nPasses with time, type, importance, and next shot xG:")
    cols = ["time","player","team","pass_type","importance_norm","next_shot_xg"]
    print(df[cols].head(20).to_string(index=False))

    # Save results
    out_path = "pass_importance_output.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
