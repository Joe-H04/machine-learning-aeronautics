"""
Improved Fusion Models for Aircraft Trajectory Reconstruction

Implements three approaches beyond the baseline:
1. Constant-Velocity Kalman Filter - Physics-based state estimation
2. Kalman Smoother - Bidirectional filtering for better reconstruction
3. LSTM Sequence Model - Machine learning approach for complex patterns

These models reconstruct missing trajectory segments better than naive great-circle
interpolation by accounting for velocity dynamics and temporal patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Try importing ML libraries; graceful fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@dataclass
class TrajectoryMetrics:
    """Metrics for evaluating trajectory reconstruction quality."""
    mae_lat: float  # Mean Absolute Error in latitude
    mae_lon: float  # Mean Absolute Error in longitude
    mae_alt: float  # Mean Absolute Error in altitude
    rmse_position: float  # Root Mean Square Error in position (km)
    velocity_smoothness: float  # Measure of velocity continuity


class ConstantVelocityKalmanFilter:
    """
    Kalman Filter with constant-velocity motion model.
    
    Assumes aircraft maintain constant velocity between observations.
    Better than interpolation because it accounts for velocity dynamics.
    
    State vector: [lat, lon, alt, v_lat, v_lon, v_alt]
    where v_* are velocities (degrees/sec and m/sec)
    """
    
    def __init__(self, process_noise: float = 1e-6, measurement_noise: float = 1e-5):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: How much we expect velocity to change (Q matrix)
            measurement_noise: How much we trust position measurements (R matrix)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = None
        self.covariance = None
    
    def initialize_state(self, lat: float, lon: float, alt: float):
        """Initialize filter with first observation."""
        self.state = np.array([lat, lon, alt, 0, 0, 0], dtype=float)
        self.covariance = np.eye(6) * 0.1
    
    def predict(self, dt: float) -> np.ndarray:
        """
        Prediction step: extrapolate position based on velocity.
        
        Args:
            dt: Time step (seconds)
        
        Returns:
            Predicted state
        """
        # Motion model: position = position + velocity * dt
        F = np.eye(6)
        F[0, 3] = dt  # lat += v_lat * dt
        F[1, 4] = dt  # lon += v_lon * dt
        F[2, 5] = dt  # alt += v_alt * dt
        
        self.state = F @ self.state
        
        # Update covariance: P = F * P * F^T + Q
        Q = np.eye(6) * self.process_noise
        Q[3:6, 3:6] *= 100  # Higher uncertainty in velocity changes
        self.covariance = F @ self.covariance @ F.T + Q
        
        return self.state
    
    def update(self, measurement: np.ndarray):
        """
        Update step: correct state based on measurement.
        
        Args:
            measurement: [lat, lon, alt] observation
        """
        # Only measure position, not velocity
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # Measurement residual
        y = measurement - H @ self.state
        
        # Kalman gain
        R = np.eye(3) * self.measurement_noise
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
    
    def filter_trajectory(self, timestamps: np.ndarray, positions: np.ndarray,
                         altitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter a complete trajectory.
        
        Args:
            timestamps: Array of timestamps (seconds from start)
            positions: Nx2 array of [lat, lon] positions
            altitudes: Array of altitudes
        
        Returns:
            Filtered (latitudes, longitudes, altitudes)
        """
        filtered_lats = []
        filtered_lons = []
        filtered_alts = []
        
        for i, (lat, lon, alt) in enumerate(zip(positions[:, 0], positions[:, 1], altitudes)):
            if i == 0:
                self.initialize_state(lat, lon, alt)
            else:
                dt = timestamps[i] - timestamps[i - 1]
                self.predict(dt)
            
            self.update(np.array([lat, lon, alt]))
            
            filtered_lats.append(self.state[0])
            filtered_lons.append(self.state[1])
            filtered_alts.append(self.state[2])
        
        return np.array(filtered_lats), np.array(filtered_lons), np.array(filtered_alts)


class KalmanSmoother:
    """
    Rauch-Tuch-Striebel Smoother.
    
    Improvements over filter:
    • Bidirectional: uses past AND future observations
    • Better for gap reconstruction: can use context from both sides
    • More accurate: processes all data multiple times
    
    Process:
    1. Forward pass: filter (past → future)
    2. Backward pass: smooth (future → past)
    """
    
    def __init__(self, process_noise: float = 1e-6, measurement_noise: float = 1e-5):
        self.kf = ConstantVelocityKalmanFilter(process_noise, measurement_noise)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def smooth_trajectory(self, timestamps: np.ndarray, positions: np.ndarray,
                         altitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply smoother to trajectory.
        
        Args:
            timestamps: Array of timestamps (seconds from start)
            positions: Nx2 array of [lat, lon] positions
            altitudes: Array of altitudes
        
        Returns:
            Smoothed (latitudes, longitudes, altitudes)
        """
        n = len(timestamps)
        
        # Forward pass: standard Kalman filter
        states_fwd = []
        covs_fwd = []
        
        for i, (lat, lon, alt) in enumerate(zip(positions[:, 0], positions[:, 1], altitudes)):
            if i == 0:
                self.kf.initialize_state(lat, lon, alt)
            else:
                dt = timestamps[i] - timestamps[i - 1]
                self.kf.predict(dt)
            
            self.kf.update(np.array([lat, lon, alt]))
            
            states_fwd.append(self.kf.state.copy())
            covs_fwd.append(self.kf.covariance.copy())
        
        states_fwd = np.array(states_fwd)
        covs_fwd = np.array(covs_fwd)
        
        # Backward pass: smooth
        states_smooth = states_fwd.copy()
        
        for i in range(n - 2, -1, -1):
            dt = timestamps[i + 1] - timestamps[i]
            
            # Predicted next state
            F = np.eye(6)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            
            state_pred_next = F @ states_fwd[i]
            
            # Smoother gain
            Q = np.eye(6) * self.process_noise
            Q[3:6, 3:6] *= 100
            cov_pred_next = F @ covs_fwd[i] @ F.T + Q
            
            D = covs_fwd[i] @ F.T @ np.linalg.inv(cov_pred_next)
            
            # Smoothed state
            states_smooth[i] = states_fwd[i] + D @ (states_smooth[i + 1] - state_pred_next)
        
        return (states_smooth[:, 0], states_smooth[:, 1], states_smooth[:, 2])


def create_lstm_model(sequence_length: int = 20, output_length: int = 5,
                      feature_dim: int = 4) -> 'keras.Model':
    """
    Create LSTM sequence model for trajectory prediction.
    
    Architecture:
    • Input: Sequences of past positions/velocities
    • LSTM layers: Learn temporal patterns
    • Dense output: Predict future positions
    
    Args:
        sequence_length: Number of past timesteps
        output_length: Number of future timesteps to predict
        feature_dim: Number of features (lat, lon, alt, time_delta)
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
    
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, 
                   input_shape=(sequence_length, feature_dim)),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_length * 3),  # Predict lat, lon, alt for each future step
        layers.Reshape((output_length, 3))
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


class LSTMTrajectoryModel:
    """LSTM-based trajectory reconstruction model."""
    
    def __init__(self, sequence_length: int = 20):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM model")
        
        self.sequence_length = sequence_length
        self.model = create_lstm_model(sequence_length)
        self.fitted = False
    
    def prepare_training_data(self, trajectories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                             output_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for LSTM.
        
        Args:
            trajectories: List of (positions, altitudes, times) tuples
            output_length: How many steps to predict
        
        Returns:
            (X_train, y_train) ready for model training
        """
        X, y = [], []
        
        for positions, altitudes, times in trajectories:
            if len(positions) < self.sequence_length + output_length:
                continue
            
            for i in range(len(positions) - self.sequence_length - output_length):
                # Input: past sequence
                seq_in = []
                for j in range(self.sequence_length):
                    idx = i + j
                    seq_in.append([
                        positions[idx, 0],  # lat
                        positions[idx, 1],  # lon
                        altitudes[idx],
                        times[idx + 1] - times[idx] if idx < len(times) - 1 else 0
                    ])
                
                # Output: future sequence
                seq_out = []
                for j in range(output_length):
                    idx = i + self.sequence_length + j
                    seq_out.append([
                        positions[idx, 0],  # lat
                        positions[idx, 1],  # lon
                        altitudes[idx]
                    ])
                
                X.append(seq_in)
                y.append(seq_out)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             epochs: int = 50, batch_size: int = 32, verbose: int = 0):
        """Train the LSTM model."""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                      verbose=verbose, validation_split=0.2)
        self.fitted = True
    
    def predict_gap(self, before: np.ndarray, after: np.ndarray,
                   times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict trajectory to fill a gap.
        
        Args:
            before: Trajectory segment before gap
            after: Trajectory segment after gap
            times: Times for the gap
        
        Returns:
            (predicted_lats, predicted_lons, predicted_alts)
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")
        
        # Use data before gap as input
        input_seq = before[-self.sequence_length:]
        input_data = np.array([[
            input_seq[i, 0],
            input_seq[i, 1],
            input_seq[i, 2] if len(input_seq[i]) > 2 else 0,
            1  # time delta
        ] for i in range(len(input_seq))])
        
        predictions = self.model.predict(input_data[np.newaxis, :], verbose=0)
        
        return predictions[0, :, 0], predictions[0, :, 1], predictions[0, :, 2]


class FusionTrajectoryModel:
    """
    Main fusion model combining multiple approaches.
    
    Provides interface to:
    • Baseline: Great-circle interpolation
    • Kalman Filter: Physics-based filtering
    • Kalman Smoother: Bidirectional smoothing
    • LSTM: Machine learning approach
    """
    
    def __init__(self, use_lstm: bool = False):
        self.kf = ConstantVelocityKalmanFilter()
        self.smoother = KalmanSmoother()
        self.lstm_model = None
        
        if use_lstm:
            try:
                self.lstm_model = LSTMTrajectoryModel()
            except ImportError:
                print("⚠ TensorFlow not available. LSTM model disabled.")
    
    def reconstruct_gap(self, before_lat: np.ndarray, before_lon: np.ndarray,
                       before_alt: np.ndarray, before_times: np.ndarray,
                       after_lat: np.ndarray, after_lon: np.ndarray,
                       after_alt: np.ndarray, after_times: np.ndarray,
                       gap_times: np.ndarray,
                       method: str = 'smoother') -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Reconstruct a gap in trajectory using specified method.
        
        Args:
            before_*: Trajectory data before gap
            after_*: Trajectory data after gap
            gap_times: Times to fill
            method: 'kalman', 'smoother', or 'lstm'
        
        Returns:
            Dict with reconstruction from each available method
        """
        results = {}
        
        # Combine context from before and after
        combined_times = np.concatenate([before_times, gap_times, after_times])
        combined_lats = np.concatenate([before_lat, np.zeros_like(gap_times), after_lat])
        combined_lons = np.concatenate([before_lon, np.zeros_like(gap_times), after_lon])
        combined_alts = np.concatenate([before_alt, np.zeros_like(gap_times), after_alt])
        
        combined_positions = np.column_stack([combined_lats, combined_lons])
        
        # Normalize times to seconds from start
        combined_times = (combined_times - combined_times[0]).total_seconds()
        
        # Kalman filter
        try:
            filtered_lats, filtered_lons, filtered_alts = self.kf.filter_trajectory(
                combined_times, combined_positions, combined_alts
            )
            gap_idx = len(before_times)
            results['kalman'] = (
                filtered_lats[gap_idx:gap_idx + len(gap_times)],
                filtered_lons[gap_idx:gap_idx + len(gap_times)],
                filtered_alts[gap_idx:gap_idx + len(gap_times)]
            )
        except Exception as e:
            print(f"Kalman filter error: {e}")
        
        # Kalman smoother
        if method == 'smoother':
            try:
                smoothed_lats, smoothed_lons, smoothed_alts = self.smoother.smooth_trajectory(
                    combined_times, combined_positions, combined_alts
                )
                gap_idx = len(before_times)
                results['smoother'] = (
                    smoothed_lats[gap_idx:gap_idx + len(gap_times)],
                    smoothed_lons[gap_idx:gap_idx + len(gap_times)],
                    smoothed_alts[gap_idx:gap_idx + len(gap_times)]
                )
            except Exception as e:
                print(f"Smoother error: {e}")
        
        return results
    
    def evaluate_reconstruction(self, true_positions: np.ndarray,
                              predicted_positions: np.ndarray) -> TrajectoryMetrics:
        """Evaluate reconstruction quality against ground truth."""
        lat_error = np.mean(np.abs(true_positions[:, 0] - predicted_positions[:, 0]))
        lon_error = np.mean(np.abs(true_positions[:, 1] - predicted_positions[:, 1]))
        
        # Simple Euclidean distance error (not great-circle, but for comparison)
        dist_error = np.mean(np.linalg.norm(
            true_positions - predicted_positions, axis=1
        ))
        
        # Velocity smoothness (lower is smoother)
        velocities = np.diff(predicted_positions, axis=0)
        velocity_changes = np.diff(velocities, axis=0)
        smoothness = np.mean(np.linalg.norm(velocity_changes, axis=1))
        
        return TrajectoryMetrics(
            mae_lat=lat_error,
            mae_lon=lon_error,
            mae_alt=0.0,  # Would need altitude data
            rmse_position=dist_error,
            velocity_smoothness=smoothness
        )


if __name__ == "__main__":
    print("Improved Fusion Models for Aircraft Trajectory Reconstruction")
    print("=" * 70)
    
    # Demonstration with synthetic data
    print("\n[1] Creating synthetic trajectory with gap...")
    
    # Before gap
    t_before = np.linspace(0, 300, 30)  # 30 points over 5 minutes
    lat_before = 47.5 + 0.01 * np.sin(t_before / 100)
    lon_before = 8.5 + 0.01 * np.cos(t_before / 100)
    alt_before = 10000 + 100 * np.sin(t_before / 150)
    
    # After gap (simulating 2 minutes of flight)
    t_after = t_before + 120
    lat_after = lat_before[-1] + 0.02 + 0.01 * np.sin(t_after / 100)
    lon_after = lon_before[-1] + 0.02 + 0.01 * np.cos(t_after / 100)
    alt_after = alt_before[-1] + 200 + 100 * np.sin(t_after / 150)
    
    print(f"  Before gap: {len(lat_before)} points")
    print(f"  After gap: {len(lat_after)} points")
    print(f"  Gap duration: 120 seconds")
    
    # Test Kalman filter
    print("\n[2] Testing Constant-Velocity Kalman Filter...")
    kf = ConstantVelocityKalmanFilter(process_noise=1e-6, measurement_noise=1e-4)
    
    combined_times = np.concatenate([t_before, t_before[-1] + np.linspace(1, 120, 50)])
    combined_lats = np.concatenate([lat_before, np.zeros(50)])
    combined_lons = np.concatenate([lon_before, np.zeros(50)])
    combined_alts = np.concatenate([alt_before, np.zeros(50)])
    
    combined_positions = np.column_stack([combined_lats, combined_lons])
    
    filt_lats, filt_lons, filt_alts = kf.filter_trajectory(combined_times, combined_positions, combined_alts)
    print(f"  Filtered trajectory: {len(filt_lats)} points")
    
    # Test Kalman smoother
    print("\n[3] Testing Kalman Smoother (RTS)...")
    smoother = KalmanSmoother(process_noise=1e-6, measurement_noise=1e-4)
    smooth_lats, smooth_lons, smooth_alts = smoother.smooth_trajectory(
        combined_times, combined_positions, combined_alts
    )
    print(f"  Smoothed trajectory: {len(smooth_lats)} points")
    
    print("\n[4] Model Comparison:")
    print("  ✓ Constant-Velocity Kalman Filter: Lightweight, physics-based")
    print("  ✓ Kalman Smoother: Better for gaps, bidirectional")
    print("  ✓ LSTM Model: Learns complex patterns (requires training data)")
    
    print("\n[5] Recommendations:")
    print("  • Start with Kalman Smoother for trajectory gaps")
    print("  • Use when: gaps are small (<5 min), need fast inference")
    print("  • Train LSTM on complete trajectories for better long-term predictions")
    print("  • Compare against baseline to quantify improvement")
