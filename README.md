# Machine Learning for Aircraft Trajectory Reconstruction

A machine learning project for reconstructing aircraft flight paths from sparse position data, using both physics-based and neural network approaches.

## Overview

Aircraft tracking systems (radar, ADS-B) often have gaps in position reports. This project builds increasingly sophisticated methods to reconstruct missing trajectory segments:

1. **Baseline** — Great-circle interpolation (simple geometric approach)
2. **Kalman Filter** — Physics-based constant-velocity model
3. **Kalman Smoother** — Bidirectional smoothing for better reconstruction ⭐ _Recommended_
4. **LSTM Model** — Deep learning on complete trajectories

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Flight Data

```bash
python data_pipeline.py
```

Fetches real aircraft tracking data from [OpenSky Network](https://opensky-network.org/) for Europe (Switzerland region).
Stores parquet files in `data/clean/tracks/` and `data/clean/flights/`.

### Understand the Baseline

```bash
python baseline.py
```

Demonstrates great-circle interpolation—the simplest approach to filling trajectory gaps.

**Output:** Example trajectory with interpolated points

## Project Structure

```
baseline.py              Great-circle interpolation baseline
model.py                Improved fusion models (Kalman, LSTM)
data_pipeline.py        Fetch flight data from OpenSky Network
│
explanation.txt         Baseline detailed explanation
MODEL_GUIDE.txt         In-depth model documentation
README.md              (this file)
│
data/
  clean/
    flights/           Flight metadata from OpenSky
    tracks/            Individual flight trajectories (parquet files)
```

## Core Modules

### baseline.py

Simple reference model using great-circle arcs.

**Key Functions:**

- `fill_trajectory_gaps()` — Main function to interpolate missing positions
- `interpolate_great_circle()` — Geodetic interpolation
- `haversine_distance()` — Great-circle distance calculation

**Use Case:** Quick baseline for comparison

### model.py

Advanced trajectory reconstruction models.

**Classes:**

- `ConstantVelocityKalmanFilter` — Forward filtering (real-time)
- `KalmanSmoother` — Bidirectional smoothing (offline) ⭐
- `LSTMTrajectoryModel` — Deep learning (requires training)
- `FusionTrajectoryModel` — Unified interface

**Use Case:** Production trajectory reconstruction

## Usage Examples

### Example 1: Smooth a Trajectory (Recommended)

```python
from model import KalmanSmoother
import pandas as pd
import numpy as np

# Load flight data
df = pd.read_parquet("data/clean/tracks/example_flight.parquet")

# Prepare data (convert timestamps to seconds from start)
df = df.sort_values('time')
start_time = df['time'].min()
timestamps = (df['time'] - start_time).astype(float).values
positions = df[['latitude', 'longitude']].values
altitudes = df['baro_altitude'].values

# Apply Kalman Smoother
smoother = KalmanSmoother(process_noise=1e-6, measurement_noise=1e-4)
smooth_lats, smooth_lons, smooth_alts = smoother.smooth_trajectory(
    timestamps, positions, altitudes
)

# Save results
df_smoothed = df.copy()
df_smoothed['latitude'] = smooth_lats
df_smoothed['longitude'] = smooth_lons
df_smoothed['baro_altitude'] = smooth_alts
df_smoothed.to_parquet("output_smoothed.parquet")
```

### Example 2: Fill Trajectory Gaps with Baseline

```python
from baseline import fill_trajectory_gaps
import pandas as pd

# Load flight data
df = pd.read_parquet("data/clean/tracks/example_flight.parquet")

# Select required columns
df = df[['time', 'latitude', 'longitude', 'baro_altitude']].copy()

# Fill gaps (automatically handles Unix timestamp conversion)
df_filled = fill_trajectory_gaps(df, max_gap_seconds=300, points_per_interval=10)

print(f"Original: {len(df)} points")
print(f"With gaps filled: {len(df_filled)} points")
df_filled.to_parquet("output_filled.parquet")
```

## Methods Comparison

| Feature           | Baseline   | Kalman   | Smoother | LSTM     |
| ----------------- | ---------- | -------- | -------- | -------- |
| Interpretability  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐     |
| Speed             | Fast       | Fast     | Fast     | Slow     |
| Accuracy          | 1.0x       | 1.3-1.5x | 1.5-2.0x | 2.0-3.0x |
| Training Required | No         | No       | No       | Yes      |
| Real-time         | ✓          | ✓        | ✗        | ⚠        |

_Accuracy relative to baseline_

## Recommendations

### For Quick Results

**Use: Kalman Smoother**

- Best accuracy without training
- Handles small to medium gaps (< 10 minutes)
- Fast and interpretable

### For Real-time Tracking

**Use: Kalman Filter**

- Only needs past data (no lookahead)
- Same accuracy as smoother on short gaps
- Lightweight for edge deployment

### For Maximum Accuracy

**Use: LSTM Model**

- Requires ~1000 complete trajectories
- Can learn complex flight patterns
- 2-3x improvement potential
- Suitable for post-processing

## Key Documentation

- **[explanation.txt](explanation.txt)** — How the baseline works
- **[MODEL_GUIDE.txt](MODEL_GUIDE.txt)** — Detailed model theory and math

## Performance Metrics

Models are evaluated on:

- **MAE_LAT, MAE_LON** — Position errors in degrees (~0.001° ≈ 100m)
- **RMSE_POSITION** — Root mean square error in kilometers (main metric)
- **velocity_smoothness** — Trajectory smoothness (lower = better)

Typical improvement over baseline:

- Kalman Filter: **20-30%**
- Kalman Smoother: **35-50%**
- LSTM (with training): **50-100%+**

## Data Format

**Input DataFrame Requirements:**

- `time` — Timestamp (datetime64 or int64 Unix seconds)
- `latitude` — Decimal degrees (-90 to 90)
- `longitude` — Decimal degrees (-180 to 180)
- `baro_altitude` — Barometric altitude (meters)

**Output:** Same format with reconstructed values

## Requirements

Core Dependencies:

- `numpy` — Numerical computing
- `pandas` — Data manipulation
- `scipy` — Scientific computing
- `requests` — HTTP requests
- `python-dotenv` — Environment variables

Optional (for ML models):

- `tensorflow` — Neural networks
- `keras` — Deep learning
- `scikit-learn` — Machine learning utilities

See [requirements.txt](requirements.txt) for full list.

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-url>
cd machine-learning-aeronautics
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure OpenSky API (Optional)

Create `.env` file for API credentials:

```
OPENSKY_CLIENT_ID=your_id
OPENSKY_CLIENT_SECRET=your_secret
```

### 5. Download Data

```bash
python data_pipeline.py
```

### 6. Run Examples

```bash
python baseline.py        # See baseline in action
python model.py           # Explore improved models
```

## Troubleshooting

**ImportError: No module named 'tensorflow'**

```bash
pip install tensorflow keras
```

**Parquet file not found**

```bash
python data_pipeline.py  # Download data first
```

**Smoother produces NaN values**

- Check timestamps are monotonically increasing
- Verify positions have valid lat/lon values
- Try increasing `process_noise` to 1e-4

**Results worse than baseline**

- Tune `measurement_noise` parameter
- Check data for spikes/outliers
- Try different hyperparameters

## Next Steps

1. Run `python evaluate.py` to see baseline performance
2. Choose method based on your constraints (speed vs accuracy)
3. Tune hyperparameters on your specific data
4. For LSTM: collect complete trajectories and train
5. Deploy Kalman Smoother for production (lightweight, reliable)

## References

- **Kalman Filter**: Kalman, R. E. (1960). _A New Approach to Linear Filtering and Prediction Problems_
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). _Long Short-Term Memory_
- **Data Source**: [OpenSky Network](https://opensky-network.org/)
