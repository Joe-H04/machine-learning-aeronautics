import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://opensky-network.org/api"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
DEFAULT_AIRPORTS = ["EDDF", "EGLL", "LFPG", "EHAM", "LEMD", "LIRF", "LOWW", "LSZH", "EKCH", "EDDM"]
FLIGHT_COLUMNS = [
    "icao24", "firstSeen", "estDepartureAirport", "lastSeen", "estArrivalAirport", "callsign",
    "estDepartureAirportHorizDistance", "estDepartureAirportVertDistance",
    "estArrivalAirportHorizDistance", "estArrivalAirportVertDistance",
    "departureAirportCandidatesCount", "arrivalAirportCandidatesCount",
]
TRACK_COLUMNS = ["time", "latitude", "longitude", "baro_altitude", "true_track", "on_ground"]
TOKEN = {"value": None, "exp": 0.0}
SESSION = requests.Session()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--airports", nargs="+", default=DEFAULT_AIRPORTS)
    p.add_argument("--hours", type=int, default=4)
    p.add_argument("--output", default="data/raw")
    p.add_argument("--no-tracks", action="store_true")
    return p.parse_args()


def auth_headers():
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return {}
    if time.time() >= TOKEN["exp"]:
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        TOKEN["value"] = data["access_token"]
        TOKEN["exp"] = time.time() + data.get("expires_in", 1800) - 60
    return {"Authorization": f"Bearer {TOKEN['value']}"}


def get(path, **params):
    try:
        r = SESSION.get(f"{BASE_URL}{path}", headers=auth_headers(), params=params, timeout=30)
        
        # Handle expired tokens
        if r.status_code == 401 and TOKEN["value"]:
            TOKEN["exp"] = 0
            r = SESSION.get(f"{BASE_URL}{path}", headers=auth_headers(), params=params, timeout=30)
            
        # Handle missing tracks or OpenSky internal server crashes gracefully
        if r.status_code in (404, 500, 502, 503, 504):
            if r.status_code >= 500:
                print(f"      -> Skipped: OpenSky Server Error {r.status_code}")
            return None
            
        r.raise_for_status()
        return r.json()
        
    except requests.exceptions.RequestException as e:
        print(f"      -> Skipped: Network/Timeout Error")
        return None


def windows(start, end, step=7200):
    while start < end:
        yield start, min(start + step, end)
        start += step


def collect_flights(airports, start, end, output_dir):
    keep = {a.upper() for a in airports}
    chunks = []
    for begin, finish in windows(start, end):
        df = pd.DataFrame(get("/flights/all", begin=begin, end=finish) or [], columns=FLIGHT_COLUMNS)
        if df.empty:
            continue
        arrivals = df[df["estArrivalAirport"].isin(keep)].copy()
        if not arrivals.empty:
            arrivals["airport"] = arrivals["estArrivalAirport"]
            arrivals["direction"] = "arrivals"
            chunks.append(arrivals)
        departures = df[df["estDepartureAirport"].isin(keep)].copy()
        if not departures.empty:
            departures["airport"] = departures["estDepartureAirport"]
            departures["direction"] = "departures"
            chunks.append(departures)
    flights = pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset=["icao24", "firstSeen", "lastSeen", "airport", "direction"]
    ) if chunks else pd.DataFrame()
    if not flights.empty:
        out = output_dir / "flights"
        out.mkdir(parents=True, exist_ok=True)
        flights.to_parquet(out / f"flights_{datetime.now():%Y%m%d_%H%M%S}.parquet", index=False)
    return flights


def collect_tracks(flights, output_dir):
    out = output_dir / "tracks"
    out.mkdir(parents=True, exist_ok=True)
    
    unique_flights = flights.drop_duplicates(subset=["icao24", "firstSeen", "lastSeen"])
    total_flights = len(unique_flights)
    print(f"\nDownloading {total_flights} flight trajectories...")
    
    for i, (_, row) in enumerate(unique_flights.iterrows(), 1):
        icao24 = str(row["icao24"]).lower()
        first_seen = int(row["firstSeen"])
        print(f"[{i}/{total_flights}] Fetching track for {icao24}...")
        
        # If 'get' returns None due to an error, it safely falls back to an empty {}
        track = get("/tracks/all", icao24=icao24, time=first_seen) or {}
        
        if not track.get("path"):
            continue
            
        df = pd.DataFrame(track["path"], columns=TRACK_COLUMNS)
        df["icao24"] = icao24
        df["callsign"] = track.get("callsign") or str(row.get("callsign") or "").strip()
        df["flight_start"] = track.get("startTime")
        df["flight_end"] = track.get("endTime")
        
        # Save the successful track
        df.to_parquet(out / f"{icao24}_{first_seen}.parquet", index=False)


def main():
    args = parse_args()
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    output_dir = Path(args.output)
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(hours=args.hours)
    print(f"Window: {start:%Y-%m-%d %H:%M:%S UTC} -> {end:%Y-%m-%d %H:%M:%S UTC}")
    flights = collect_flights(args.airports, int(start.timestamp()), int(end.timestamp()), output_dir)
    print(f"Matched airport records: {len(flights)}")
    if not flights.empty and not args.no_tracks:
        collect_tracks(flights, output_dir)


if __name__ == "__main__":
    main()