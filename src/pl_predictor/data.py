from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import MATCHES_PATH, RAW_DIR, ensure_directories

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/football-datasets/main/"
    "datasets/premier-league/season-{season_code}.csv"
)

CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
KEEP_COLUMNS = CORE_COLUMNS + STAT_COLUMNS


def season_code(start_year: int) -> str:
    """Convert 2000 to 0001 and 2026 to 2627."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year:04d}/{(start_year + 1) % 100:02d}"


def _fetch_bytes(
    url: str,
    timeout: int = 30,
    attempts: int = 3,
    retry_delay: float = 2.0,
) -> bytes:
    request = Request(url, headers={"User-Agent": "pl-predictor/0.1"})
    last_error: HTTPError | URLError | None = None
    retryable_http_codes = {408, 425, 429, 500, 502, 503, 504}

    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError) as exc:
            last_error = exc
            retryable = not isinstance(exc, HTTPError) or exc.code in retryable_http_codes
            if attempt == attempts or not retryable:
                break
            time.sleep(retry_delay * (2 ** (attempt - 1)))

    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def download_season(start_year: int, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    code = season_code(start_year)
    destination = raw_dir / f"{code}.csv"
    if destination.exists() and not force:
        return destination

    try:
        payload = _fetch_bytes(BASE_URL.format(season_code=code))
        if not payload.strip():
            raise RuntimeError(f"Downloaded an empty CSV for {season_label(start_year)}")
    except RuntimeError as exc:
        if destination.exists():
            print(
                f"{season_label(start_year)}: live download failed; "
                f"using cached detailed data ({exc})"
            )
            return destination
        raise
    destination.write_bytes(payload)
    return destination


def _current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _replace_with_complete_mirror(start_year: int, destination: Path) -> None:
    code = season_code(start_year)
    payload = _fetch_bytes(MIRROR_URL.format(season_code=code))
    if not payload.strip():
        raise RuntimeError(f"Complete-data mirror returned an empty CSV for {season_label(start_year)}")
    destination.write_bytes(payload)


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    return parsed


def normalize_season(path: Path, start_year: int) -> pd.DataFrame:
    # Read through BytesIO so malformed encodings in older files do not crash the pipeline.
    raw = path.read_bytes().decode("latin-1", errors="replace")
    frame = pd.read_csv(io.StringIO(raw), on_bad_lines="skip")
    frame.columns = frame.columns.str.strip()

    missing_core = sorted(set(CORE_COLUMNS) - set(frame.columns))
    if missing_core:
        raise ValueError(f"{path.name} is missing required columns: {missing_core}")

    for column in STAT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame[KEEP_COLUMNS].copy()
    frame["Date"] = _parse_date(frame["Date"])
    frame["HomeTeam"] = frame["HomeTeam"].astype("string").str.strip()
    frame["AwayTeam"] = frame["AwayTeam"].astype("string").str.strip()
    frame["FTR"] = frame["FTR"].astype("string").str.strip().str.upper()

    for column in ["FTHG", "FTAG", *STAT_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    frame = frame[frame["FTR"].isin(["H", "D", "A"])]
    frame["FTHG"] = frame["FTHG"].astype(int)
    frame["FTAG"] = frame["FTAG"].astype(int)
    frame.insert(0, "season_start", start_year)
    frame.insert(1, "season", season_label(start_year))
    return frame.reset_index(drop=True)


def build_match_dataset(
    start_season: int = 2000,
    end_season: int = 2026,
    force: bool = False,
    refresh_current: bool = False,
    output_path: Path = MATCHES_PATH,
) -> pd.DataFrame:
    if end_season < start_season:
        raise ValueError("end_season must be greater than or equal to start_season")

    ensure_directories()
    frames: list[pd.DataFrame] = []
    for start_year in range(start_season, end_season + 1):
        should_force = force or (refresh_current and start_year == end_season)
        path = download_season(start_year, force=should_force)
        season_frame = normalize_season(path, start_year)
        if start_year < _current_season_start() and len(season_frame) != 380:
            print(
                f"{season_label(start_year)}: primary file has {len(season_frame)} rows; "
                "trying complete-data mirror"
            )
            _replace_with_complete_mirror(start_year, path)
            season_frame = normalize_season(path, start_year)
            if len(season_frame) != 380:
                raise ValueError(
                    f"Completed season {season_label(start_year)} has {len(season_frame)} matches; "
                    "expected 380"
                )
        print(f"{season_label(start_year)}: {len(season_frame):>3} completed matches")
        frames.append(season_frame)

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    print(f"Saved {len(matches):,} matches to {output_path}")
    return matches
