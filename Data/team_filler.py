import pandas as pd
import numpy as np
import os
import time

import statsapi  # pip install MLB-StatsAPI

IN_PATH  = "pitcher_year_to_year_emd_with_siera_and_stuffplus.csv"
OUT_PATH = "pitcher_year_to_year_emd_with_siera_and_stuffplus_teamfilled.csv"

SLEEP_SEC = 0.10            # be nice to the API
CACHE_PLAYER_IDS = "name_to_personid.csv"

def is_missing_team(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan" or s in ["- - -", "---", "--", "-", "None"]

def load_id_cache(path=CACHE_PLAYER_IDS):
    if os.path.exists(path):
        c = pd.read_csv(path)
        c["pitcher"] = c["pitcher"].astype(str).str.strip()
        c["personId"] = pd.to_numeric(c["personId"], errors="coerce").astype("Int64")
        return dict(zip(c["pitcher"], c["personId"]))
    return {}

def save_id_cache(cache, path=CACHE_PLAYER_IDS):
    out = pd.DataFrame({"pitcher": list(cache.keys()), "personId": list(cache.values())})
    out.to_csv(path, index=False)

def lookup_person_id(full_name: str, cache: dict):
    full_name = str(full_name).strip()
    if full_name in cache and not pd.isna(cache[full_name]):
        return int(cache[full_name])

    # returns list of candidate dicts (name, id, etc.)
    candidates = statsapi.lookup_player(full_name)  # :contentReference[oaicite:1]{index=1}
    if not candidates:
        cache[full_name] = pd.NA
        return None

    # prefer exact fullName match (case-insensitive)
    exact = [c for c in candidates if str(c.get("fullName", "")).strip().lower() == full_name.lower()]
    pick = exact[0] if exact else candidates[0]

    pid = pick.get("id", None)
    cache[full_name] = pid if pid is not None else pd.NA
    return pid

def team_for_pitcher_season(person_id: int, season: int):
    """
    Returns team abbreviation for that season.
    If multiple teams (trade), picks team with most gamesPlayed in the splits.
    """
    if person_id is None or pd.isna(person_id):
        return None

    # Use raw endpoint access for maximum control. :contentReference[oaicite:2]{index=2}
    # Hydrate yearByYear pitching splits for that season.
    hydrate = f"stats(group=[pitching],type=[yearByYear],season={int(season)})"
    data = statsapi.get("people", {"personIds": int(person_id), "hydrate": hydrate})

    people = data.get("people", [])
    if not people:
        return None

    stats = people[0].get("stats", [])
    if not stats:
        return None

    # Find yearByYear splits
    # Each split typically has: season, team, stat (including gamesPlayed)
    best_team = None
    best_gp = -1

    for block in stats:
        for split in block.get("splits", []):
            if int(split.get("season", 0)) != int(season):
                continue
            team = split.get("team", {})
            abbr = team.get("abbreviation", None)
            gp = split.get("stat", {}).get("gamesPlayed", None)
            gp = int(gp) if gp is not None else 0

            # some splits might be missing abbreviation; skip those
            if abbr is None:
                continue

            if gp > best_gp:
                best_gp = gp
                best_team = abbr

    return best_team

def fill_team_cols(df: pd.DataFrame):
    id_cache = load_id_cache()

    # build the set of pitcher-year pairs we actually need to look up
    needs = []
    for side in [("team_y1", "start_year"), ("team_y2", "end_year")]:
        tcol, ycol = side
        miss = df[tcol].apply(is_missing_team)
        needs += list(zip(df.loc[miss, "pitcher"], df.loc[miss, ycol]))
    needs = sorted(set((p, int(y)) for p, y in needs if pd.notna(p) and pd.notna(y)))

    print(f"[statsapi] pitcher-years to fill: {len(needs)}")

    # cache results of (pitcher, year) -> team
    team_cache = {}

    for i, (pname, yr) in enumerate(needs, 1):
        key = (pname, yr)
        if key in team_cache:
            continue

        pid = lookup_person_id(pname, id_cache)
        team = team_for_pitcher_season(pid, yr)

        team_cache[key] = team
        if i % 50 == 0 or i == len(needs):
            print(f"[statsapi] {i}/{len(needs)} filled (latest: {pname}, {yr} -> {team})")

        time.sleep(SLEEP_SEC)

    save_id_cache(id_cache)

    # apply fills
    for tcol, ycol in [("team_y1", "start_year"), ("team_y2", "end_year")]:
        miss = df[tcol].apply(is_missing_team)
        df.loc[miss, tcol] = [
            team_cache.get((p, int(y)), None)
            for p, y in zip(df.loc[miss, "pitcher"], df.loc[miss, ycol])
        ]

    return df

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["pitcher"] = df["pitcher"].astype(str).str.strip()

    # preserve originals
    if "team_y1_original" not in df.columns:
        df["team_y1_original"] = df["team_y1"]
    if "team_y2_original" not in df.columns:
        df["team_y2_original"] = df["team_y2"]

    df_filled = fill_team_cols(df)
    df_filled.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
