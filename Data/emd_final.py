import numpy as np
import pandas as pd
import glob
import os
import re
import time
import requests
from scipy.stats import wasserstein_distance

# -----------------------
# Config
# -----------------------
PITCH_CSV_PATH = "pitch.csv"

PITCHING_STATS_DIR = r"C:\Users\TateW\OneDrive\SeniorProject\YaleSeniorProject\AdvancedStats"
PITCHING_STATS_PATTERN = "pitching_stats_*.csv"

STUFF_STATS_DIR = r"C:\Users\TateW\OneDrive\SeniorProject\YaleSeniorProject\Stuff+"
STUFF_STATS_PATTERN = "pitcher_stuff+_*.csv"

SPEED_COL = "pitch_start_speed"
HB_COL = "break_horizontal"
VB_COL = "break_vertical_induced"
TIME_COL = "game_time"

PITCHER_COL = "pitcher"              # <-- NAME in your pitch.csv
PITCH_TYPE_COL = "pitch_type_code"

# MLB Stats API caching for gamePk -> teams
GAMEPK_TEAM_CACHE = "gamepk_team_map.csv"
MLB_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{gamepk}/feed/live"
API_SLEEP_SEC = 0.05

N_PROJ = 30
MAX_PITCH_SAMPLE = None
SEED = 0

# -----------------------
# Whitening + sliced Wasserstein
# -----------------------
def fit_whitener(X: np.ndarray, eps: float = 1e-6):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        mu = X.mean(axis=0) if X.shape[0] else np.zeros(X.shape[1])
        return mu, np.eye(X.shape[1])
    mu = X.mean(axis=0)
    Xc = X - mu
    C = np.cov(Xc, rowvar=False) + eps * np.eye(X.shape[1])
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu, W

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, dirs: np.ndarray) -> float:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.nan
    return float(np.mean([wasserstein_distance(X @ v, Y @ v) for v in dirs]))

# -----------------------
# MLB Stats API: gamePk -> home/away team abbreviation (with cache)
# -----------------------
def load_gamepk_team_cache(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        c = pd.read_csv(path)
        c["gamepk"] = pd.to_numeric(c["gamepk"], errors="coerce").astype("Int64")
        return c
    return pd.DataFrame(columns=["gamepk", "home_team", "away_team"])

def fetch_home_away_abbr(gamepk: int) -> tuple[str, str]:
    url = MLB_FEED_URL.format(gamepk=int(gamepk))
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    home = j["gameData"]["teams"]["home"]["abbreviation"]
    away = j["gameData"]["teams"]["away"]["abbreviation"]
    return home, away

def build_gamepk_team_map(gamepks: pd.Series, cache_path: str = GAMEPK_TEAM_CACHE) -> pd.DataFrame:
    cache = load_gamepk_team_cache(cache_path)
    have = set(cache["gamepk"].dropna().astype(int).tolist())

    unique_gamepks = (
        pd.to_numeric(pd.Series(gamepks), errors="coerce")
          .dropna()
          .astype(int)
          .unique()
          .tolist()
    )
    need = [g for g in unique_gamepks if g not in have]
    if not need:
        return cache[["gamepk", "home_team", "away_team"]].copy()

    rows = []
    for g in need:
        try:
            home, away = fetch_home_away_abbr(g)
            rows.append({"gamepk": g, "home_team": home, "away_team": away})
        except Exception:
            rows.append({"gamepk": g, "home_team": pd.NA, "away_team": pd.NA})

        if API_SLEEP_SEC:
            time.sleep(API_SLEEP_SEC)

    new = pd.DataFrame(rows)
    out = pd.concat([cache, new], ignore_index=True)
    out["gamepk"] = pd.to_numeric(out["gamepk"], errors="coerce").astype("Int64")
    out = out.drop_duplicates(subset=["gamepk"], keep="last")
    out.to_csv(cache_path, index=False)
    return out[["gamepk", "home_team", "away_team"]].copy()

# -----------------------
# Load pitching stats (Team + SIERA) from folder
#   IMPORTANT: pitch.csv pitcher is NAME, so we merge by Name+year
#   We keep an "ambiguous" flag if multiple rows exist for same Name+year.
# -----------------------
def load_pitching_stats(folder: str, pattern: str) -> pd.DataFrame:
    glob_path = os.path.join(folder, pattern)
    paths = sorted(glob.glob(glob_path))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_path}")

    frames = []
    for p in paths:
        m = re.search(r"(\d{4})", os.path.basename(p))
        if not m:
            continue
        year = int(m.group(1))
        df = pd.read_csv(p)

        needed = ["Name", "Team", "SIERA"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")

        out = df[["Name", "Team", "SIERA"]].copy()
        out["year"] = year
        out["Name"] = out["Name"].astype(str).str.strip()
        out["Team"] = out["Team"].astype(str).str.strip().replace({"- - -": pd.NA, "nan": pd.NA})
        frames.append(out)

    perf = pd.concat(frames, ignore_index=True)

    # mark ambiguous Name+year
    dup_ct = perf.groupby(["Name", "year"]).size().rename("n_rows").reset_index()
    perf = perf.merge(dup_ct, on=["Name", "year"], how="left")
    perf["ambiguous_pitching_stats"] = perf["n_rows"] > 1

    # deterministic pick (does NOT "fix" multi-team, but avoids random)
    perf = perf.sort_values(["Name", "year", "Team"]).drop_duplicates(subset=["Name", "year"], keep="first")
    perf = perf.drop(columns=["n_rows"])

    return perf

# -----------------------
# Load Stuff+ files (merge by Name+year for the same reason)
# -----------------------
def load_stuffplus_files(folder: str, pattern: str) -> pd.DataFrame:
    glob_path = os.path.join(folder, pattern)
    paths = sorted(glob.glob(glob_path))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_path}")

    frames = []
    for p in paths:
        m = re.search(r"(\d{4})", os.path.basename(p))
        if not m:
            continue
        year = int(m.group(1))

        df = pd.read_csv(p)
        needed = ["Name", "Team", "Stuff+", "Location+", "Pitching+"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")

        out = df[["Name", "Team", "Stuff+", "Location+", "Pitching+"]].copy()
        out["year"] = year
        out["Name"] = out["Name"].astype(str).str.strip()
        out["Team"] = out["Team"].astype(str).str.strip().replace({"- - -": pd.NA, "nan": pd.NA})
        frames.append(out)

    stuff = pd.concat(frames, ignore_index=True)

    dup_ct = stuff.groupby(["Name", "year"]).size().rename("n_rows").reset_index()
    stuff = stuff.merge(dup_ct, on=["Name", "year"], how="left")
    stuff["ambiguous_stuff_stats"] = stuff["n_rows"] > 1

    stuff = stuff.sort_values(["Name", "year", "Team"]).drop_duplicates(subset=["Name", "year"], keep="first")
    stuff = stuff.drop(columns=["n_rows"])

    return stuff

# -----------------------
# Main
# -----------------------
def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(PITCH_CSV_PATH)
    required = [PITCHER_COL, PITCH_TYPE_COL, TIME_COL, SPEED_COL, HB_COL, VB_COL, "gamepk", "is_top_inning"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in pitch file: {missing}")

    # year
    dt = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df["year"] = dt.dt.year
    df = df.dropna(subset=["year", SPEED_COL, HB_COL, VB_COL, PITCHER_COL, PITCH_TYPE_COL])
    df["year"] = df["year"].astype(int)

    # pitcher is NAME
    df["pitcher_name"] = df[PITCHER_COL].astype(str).str.strip()

    # --- Add home/away via MLB Stats API (cached) ---
    df["gamepk"] = pd.to_numeric(df["gamepk"], errors="coerce").astype("Int64")
    gmap = build_gamepk_team_map(df["gamepk"], cache_path=GAMEPK_TEAM_CACHE)
    df = df.merge(gmap, on="gamepk", how="left")

    # pitcher is on fielding team:
    # top inning -> home fields; bottom inning -> away fields
    df["pitcher_team_pitch"] = np.where(df["is_top_inning"].astype(bool), df["home_team"], df["away_team"])

    # team_start/team_most per pitcher-year (by pitch count)
    team_year = (
        df.dropna(subset=["pitcher_name", "year", "pitcher_team_pitch", TIME_COL])
          .sort_values(TIME_COL)
          .groupby(["pitcher_name", "year"], as_index=False)
          .agg(
              team_start=("pitcher_team_pitch", "first"),
              team_most=("pitcher_team_pitch", lambda s: s.value_counts().idxmax()),
          )
    )

    # --- EMD inputs (unchanged) ---
    use = df[["pitcher_name", "year", PITCH_TYPE_COL, SPEED_COL, HB_COL, VB_COL]].copy()
    pitch_types = sorted(use[PITCH_TYPE_COL].unique().tolist())

    dirs = rng.normal(size=(N_PROJ, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    rows = []
    for pitcher, g in use.groupby("pitcher_name", sort=False):
        years = np.sort(g["year"].unique())
        if len(years) < 2:
            continue

        year_to_X = {}
        for yr, gy in g.groupby("year", sort=False):
            X = gy[[SPEED_COL, HB_COL, VB_COL]].to_numpy(dtype=np.float32)
            if MAX_PITCH_SAMPLE is not None and X.shape[0] > MAX_PITCH_SAMPLE:
                idx = rng.choice(X.shape[0], size=MAX_PITCH_SAMPLE, replace=False)
                X = X[idx]
            year_to_X[int(yr)] = X

        means = (
            g.groupby(["year", PITCH_TYPE_COL])[[SPEED_COL, HB_COL, VB_COL]]
             .mean()
             .rename(columns={SPEED_COL: "velo", HB_COL: "hb", VB_COL: "vb"})
        )

        counts = g.groupby("year").size().to_dict()

        year_set = set(int(y) for y in years)
        for y1 in sorted(year_set):
            y2 = y1 + 1
            if y2 not in year_set:
                continue

            X1 = year_to_X.get(y1, np.empty((0, 3), dtype=np.float32))
            X2 = year_to_X.get(y2, np.empty((0, 3), dtype=np.float32))

            if X1.shape[0] == 0 or X2.shape[0] == 0 or (X1.shape[0] + X2.shape[0]) < 2:
                emd = np.nan
            else:
                X_all = np.vstack([X1, X2])
                mu, W = fit_whitener(X_all)
                X1w = (X1.astype(np.float64) - mu) @ W
                X2w = (X2.astype(np.float64) - mu) @ W
                emd = sliced_wasserstein(X1w, X2w, dirs)

            row = {
                "pitcher": pitcher,
                "start_year": y1,
                "end_year": y2,
                "emd_whitened_sliced": emd,
                "n_pitches_year1": int(counts.get(y1, 0)),
                "n_pitches_year2": int(counts.get(y2, 0)),
            }

            for pt in pitch_types:
                if (y1, pt) in means.index:
                    v1, h1, b1 = means.loc[(y1, pt), ["velo", "hb", "vb"]].tolist()
                else:
                    v1 = h1 = b1 = np.nan

                if (y2, pt) in means.index:
                    v2, h2, b2 = means.loc[(y2, pt), ["velo", "hb", "vb"]].tolist()
                else:
                    v2 = h2 = b2 = np.nan

                row[f"y1_velo_{pt}"] = v1
                row[f"y1_hb_{pt}"] = h1
                row[f"y1_vb_{pt}"] = b1
                row[f"y2_velo_{pt}"] = v2
                row[f"y2_hb_{pt}"] = h2
                row[f"y2_vb_{pt}"] = b2

            rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["pitcher", "start_year"]).reset_index(drop=True)

    # --- attach pitch-derived teams (these are the "correct" teams for your pitch data) ---
    out_df = out_df.merge(
        team_year.rename(columns={"year": "start_year", "team_start": "team_start_y1", "team_most": "team_most_y1"}),
        on=["pitcher", "start_year"],
        how="left",
    )
    out_df = out_df.merge(
        team_year.rename(columns={"year": "end_year", "team_start": "team_start_y2", "team_most": "team_most_y2"}),
        on=["pitcher", "end_year"],
        how="left",
    )

    # --- Merge SIERA + Team (year-specific) by Name+year ---
    perf = load_pitching_stats(PITCHING_STATS_DIR, PITCHING_STATS_PATTERN)

    out_df = out_df.merge(
        perf.rename(columns={"Name": "pitcher", "Team": "team_y1", "SIERA": "siera_y1", "year": "start_year"}),
        on=["pitcher", "start_year"],
        how="left",
    )
    out_df = out_df.merge(
        perf.rename(columns={"Name": "pitcher", "Team": "team_y2", "SIERA": "siera_y2", "year": "end_year"}),
        on=["pitcher", "end_year"],
        how="left",
    )

    # --- Merge Stuff+/Location+/Pitching+ by Name+year ---
    stuff = load_stuffplus_files(STUFF_STATS_DIR, STUFF_STATS_PATTERN)

    out_df = out_df.merge(
        stuff.rename(columns={
            "Name": "pitcher",
            "Team": "team_stuff_y1",
            "Stuff+": "stuff+_y1",
            "Location+": "location+_y1",
            "Pitching+": "pitching+_y1",
            "year": "start_year",
        }),
        on=["pitcher", "start_year"],
        how="left",
    )

    out_df = out_df.merge(
        stuff.rename(columns={
            "Name": "pitcher",
            "Team": "team_stuff_y2",
            "Stuff+": "stuff+_y2",
            "Location+": "location+_y2",
            "Pitching+": "pitching+_y2",
            "year": "end_year",
        }),
        on=["pitcher", "end_year"],
        how="left",
    )

    # diffs
    out_df["diff_siera"] = out_df["siera_y2"] - out_df["siera_y1"]
    out_df["diff_stuff+"] = out_df["stuff+_y2"] - out_df["stuff+_y1"]
    out_df["diff_location+"] = out_df["location+_y2"] - out_df["location+_y1"]
    out_df["diff_pitching+"] = out_df["pitching+_y2"] - out_df["pitching+_y1"]

    # useful flags: when Name-year was ambiguous in your source stats
    # (you can filter these out in team-change analyses if you want)
    if "ambiguous_pitching_stats" in perf.columns:
        # bring ambiguity flags in for y1/y2
        a1 = perf[["Name", "year", "ambiguous_pitching_stats"]].rename(
            columns={"Name": "pitcher", "year": "start_year", "ambiguous_pitching_stats": "ambig_perf_y1"}
        )
        a2 = perf[["Name", "year", "ambiguous_pitching_stats"]].rename(
            columns={"Name": "pitcher", "year": "end_year", "ambiguous_pitching_stats": "ambig_perf_y2"}
        )
        out_df = out_df.merge(a1, on=["pitcher", "start_year"], how="left")
        out_df = out_df.merge(a2, on=["pitcher", "end_year"], how="left")

    return out_df

if __name__ == "__main__":
    df_out = main()
    print(df_out.head())
    df_out.to_csv("pitcher_year_to_year_emd_with_siera_and_stuffplus.csv", index=False)
    print("\nSaved: pitcher_year_to_year_emd_with_siera_and_stuffplus.csv")
