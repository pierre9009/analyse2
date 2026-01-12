"""
Magnetometer Heading Update for EKF (with B_NED estimation).
"""

import numpy as np
from ekf.updates.base import UpdateBase
from ekf.utils import Utils


class HeadingMagUpdate(UpdateBase):
    """
    Update EKF with magnetometer measurement.

    This update estimates both the quaternion orientation and the
    local magnetic field vector B_NED simultaneously.

    Measurement model: z = R^T @ B_NED_normalized (body frame)
    where R is body->NED rotation matrix.
    """

    def __init__(self, R_mag=None, max_innovation_deg=30.0, max_quaternion_gain=0.1):
        super().__init__(state_dim=19)

        # Default measurement noise covariance
        if R_mag is None:
            R_mag = (20 * np.pi / 180) ** 2  # (20 degrees)^2

        self.R_scalar = R_mag
        self.R = np.diag([R_mag, R_mag, R_mag])
        self.max_innovation = max_innovation_deg * np.pi / 180  # radians
        self.max_quaternion_gain = max_quaternion_gain

        # Cache for innovation check
        self._last_innovation_norm = None

    def compute_innovation(self, x, measurement):
        """
        Compute innovation for magnetometer.

        Args:
            x: State vector (19x1)
            measurement: Magnetometer reading (3x1) in body frame

        Returns:
            tuple: (y, h) innovation and predicted measurement
        """
        # Normalize measurement
        mag_norm = np.linalg.norm(measurement)
        if mag_norm < 1e-6:
            return None, None
        mag_n = measurement / mag_norm

        # Get estimated B_NED from state
        B_NED_est = x[16:19]
        B_NED_norm = np.linalg.norm(B_NED_est)
        if B_NED_norm < 1e-6:
            return None, None
        B_NED_n = B_NED_est / B_NED_norm

        # Predicted measurement: h(x) = R^T @ B_NED_normalized
        q = x[0:4]
        R = Utils.quaternion_to_rotation_matrix(q)
        h = R.T @ B_NED_n  # NED -> body
        h = h / np.linalg.norm(h)

        # Measurement vector
        z = mag_n

        # Innovation
        y = z - h

        # Cache innovation norm for validation
        self._last_innovation_norm = np.linalg.norm(y)

        return y, h

    def compute_jacobian(self, x, measurement=None):
        """
        Compute Jacobian H for magnetometer using finite differences.

        Computes derivatives w.r.t. quaternion (3x4) and B_NED (3x3).

        Args:
            x: State vector (19x1)
            measurement: Magnetometer reading (needed for Jacobian computation)

        Returns:
            np.ndarray: H matrix (3x19)
        """
        q = x[0:4]
        B_NED_est = x[16:19]

        # Normalize B_NED
        B_NED_norm = np.linalg.norm(B_NED_est)
        if B_NED_norm < 1e-6:
            return np.zeros((3, self.state_dim))
        B_NED_n = B_NED_est / B_NED_norm

        # Compute reference h value
        R = Utils.quaternion_to_rotation_matrix(q)
        h = R.T @ B_NED_n
        h = h.flatten() / np.linalg.norm(h)

        q_flat = q.flatten()
        B_flat = B_NED_n.flatten()
        epsilon = 1e-7

        # Jacobian dh/dq (3x4) via finite differences
        H_q = np.zeros((3, 4))
        for i in range(4):
            q_plus = q_flat.copy()
            q_plus[i] += epsilon
            q_plus = q_plus / np.linalg.norm(q_plus)

            R_plus = Utils.quaternion_to_rotation_matrix(q_plus.reshape(4, 1))
            h_plus = R_plus.T @ B_NED_n
            h_plus = h_plus.flatten() / np.linalg.norm(h_plus)

            H_q[:, i] = (h_plus - h) / epsilon

        # Jacobian dh/dB_NED (3x3) via finite differences
        H_B = np.zeros((3, 3))
        for i in range(3):
            B_plus = B_flat.copy()
            B_plus[i] += epsilon
            B_plus = B_plus / np.linalg.norm(B_plus)

            h_plus = R.T @ B_plus.reshape(3, 1)
            h_plus = h_plus.flatten() / np.linalg.norm(h_plus)

            H_B[:, i] = (h_plus - h) / epsilon

        # Full Jacobian H (3x19)
        H = np.zeros((3, self.state_dim))
        H[:, 0:4] = H_q      # dh/dq
        H[:, 16:19] = H_B    # dh/dB_NED

        return H

    def get_measurement_noise(self):
        """Get measurement noise covariance R."""
        return self.R

    def validate_measurement(self, x, measurement):
        """
        Validate magnetometer measurement.
        Checks for valid magnitude and innovation gating.
        """
        if measurement is None:
            return False

        mag_norm = np.linalg.norm(measurement)
        if mag_norm < 1e-6:
            return False

        B_NED_est = x[16:19]
        B_NED_norm = np.linalg.norm(B_NED_est)
        if B_NED_norm < 1e-6:
            return False

        return True

    def check_innovation_gate(self):
        """
        Check if innovation is within acceptable bounds.
        Must be called after compute_innovation().

        Returns:
            bool: True if innovation is acceptable
        """
        if self._last_innovation_norm is None:
            return False
        return self._last_innovation_norm <= self.max_innovation

    def get_quaternion_gain_limit(self):
        """Get the maximum allowed gain for quaternion updates."""
        return self.max_quaternion_gain

    def prepare_update(self, x, measurement):
        """
        Override prepare_update to add gating check and info for gain saturation.
        """
        if not self.validate_measurement(x, measurement):
            return None

        y, h = self.compute_innovation(x, measurement)
        if y is None:
            return None

        # Gating check
        if not self.check_innovation_gate():
            print(f"Mag innovation: {np.rad2deg(self._last_innovation_norm):.1f} deg > {np.rad2deg(self.max_innovation):.1f} deg -> skip")
            return None

        H = self.compute_jacobian(x, measurement)
        R = self.get_measurement_noise()

        return {
            'y': y,
            'H': H,
            'R': R,
            'h': h,
            'saturate_quaternion_gain': True,
            'max_quaternion_gain': self.max_quaternion_gain,
            'use_joseph_form': True,
            'normalize_B_NED': True
        }
