"""
GPS Position + Velocity Update for EKF.
"""

import numpy as np
from ekf.updates.base import UpdateBase


class GPSPositionVelocityUpdate(UpdateBase):
    """
    Update EKF with GPS position and velocity measurements.

    Measurement model: z = [px, py, pz, vx, vy, vz]^T
    State indices: position [4:7], velocity [7:10]
    """

    def __init__(self, R_position=None, R_velocity=None):
        super().__init__(state_dim=16)

        # Default measurement noise covariances
        if R_position is None:
            R_position = np.array([25, 25, 100])  # m^2
        if R_velocity is None:
            R_velocity = np.array([0.25, 0.25, 0.64])  # (m/s)^2

        self.R = np.diag(np.concatenate([R_position, R_velocity]))

    def compute_innovation(self, x, measurement):
        """
        Compute innovation for GPS position + velocity.

        Args:
            x: State vector (16x1)
            measurement: dict with 'position' (3x1) and 'velocity' (3x1)

        Returns:
            tuple: (y, h) innovation and predicted measurement
        """
        position = measurement['position']
        velocity = measurement['velocity']

        # Measurement vector z (6x1)
        z = np.vstack([position, velocity])

        # Predicted measurement h(x) = [p, v]
        h = np.vstack([
            x[4:7],   # position
            x[7:10]   # velocity
        ])

        # Innovation
        y = z - h

        return y, h

    def compute_jacobian(self, x, measurement=None):
        """
        Compute Jacobian H for GPS position + velocity.
        H is constant: identity blocks at position and velocity indices.

        Args:
            x: State vector (16x1)
            measurement: Not used for this update

        Returns:
            np.ndarray: H matrix (6x16)
        """
        H = np.zeros((6, self.state_dim))
        H[0:3, 4:7] = np.eye(3)   # dh_position/d_position = I
        H[3:6, 7:10] = np.eye(3)  # dh_velocity/d_velocity = I

        return H

    def get_measurement_noise(self):
        """Get measurement noise covariance R."""
        return self.R

    def validate_measurement(self, x, measurement):
        """Validate GPS measurement."""
        if measurement is None:
            return False
        if 'position' not in measurement or 'velocity' not in measurement:
            return False
        return True
