# machine-learning-aero

A machine learning project for aviation data analysis, starting with flight data ingestion from the OpenSky Network API.

## Data Pipeline

`data_pipeline.py` fetches live aircraft state data from the [OpenSky Network](https://opensky-network.org/) for a geographic bounding box over Europe (Switzerland region) and saves it locally as a CSV.

**Output:** `raw_flight_data.csv` — one row per aircraft with fields: ICAO24, callsign, origin country, position, altitude, velocity, and more.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python data_pipeline.py
```
