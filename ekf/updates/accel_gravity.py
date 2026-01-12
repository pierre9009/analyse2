"""
Accelerometer Gravity Update for EKF (Roll/Pitch correction).
"""

import numpy as np
from ekf.updates.base import UpdateBase
from ekf.utils import Utils

GRAVITY = 9.81


class AccelGravityUpdate(UpdateBase):
    """
    Update EKF with accelerometer measurement for roll/pitch correction.

    Measurement model: z = R^T @ [0, 0, -g]^T + b_accel + noise
    where R is body->NED rotation matrix, so R^T is NED->body.

    This update corrects roll and pitch angles via the quaternion,
    plus the accelerometer bias.
    """

    def __init__(self, R_accel=None, gravity_threshold=0.5):
        super().__init__(state_dim=19)

        # Default measurement noise covariance
        if R_accel is None:
            R_accel = np.array([0.5, 0.5, 0.5])  # (m/s^2)^2

        self.R = np.diag(R_accel)
        self.gravity_threshold = gravity_threshold  # Only update if |a| ~ g

    def compute_innovation(self, x, measurement):
        """
        Compute innovation for accelerometer gravity.

        Args:
            x: State vector (19x1)
            measurement: Accelerometer reading (3x1) in body frame

        Returns:
            tuple: (y, h) innovation and predicted measurement
        """
        q = x[0:4]
        b_accel = x[13:16]

        z = measurement

        # Predicted measurement: h(x) = R^T @ [0, 0, -g]^T + b_accel
        R_T = Utils.quaternion_to_rotation_matrix(q).T  # NED -> body
        h = R_T @ np.array([0, 0, -GRAVITY]).reshape((3, 1)) + b_accel

        # Innovation
        y = z - h

        return y, h

    def compute_jacobian(self, x, measurement=None):
        """
        Compute Jacobian H for accelerometer gravity.

        Analytical derivatives from h = R^T @ [0,0,-g]^T + b_accel:
        h1 = -2g*(q1*q3 - q0*q2) = 2g*q0*q2 - 2g*q1*q3
        h2 = -2g*(q2*q3 + q0*q1) = -2g*q0*q1 - 2g*q2*q3
        h3 = -g + 2g*(q1^2 + q2^2)

        Args:
            x: State vector (19x1)
            measurement: Not used

        Returns:
            np.ndarray: H matrix (3x19)
        """
        q = x[0:4]
        q0, q1, q2, q3 = q.flatten()

        # Jacobian dh/dq (3x4)
        H_q = np.array([
            [2*q2*GRAVITY, -2*q3*GRAVITY,  2*q0*GRAVITY, -2*q1*GRAVITY],
            [-2*q1*GRAVITY, -2*q0*GRAVITY, -2*q3*GRAVITY, -2*q2*GRAVITY],
            [0,  4*q1*GRAVITY,  4*q2*GRAVITY, 0]
        ])

        # Full Jacobian H (3x19):
        # [H_q(3x4), zeros(3x3), zeros(3x3), zeros(3x3), I(3x3), zeros(3x3)]
        #  quat      pos         vel         bg          ba         B_NED
        H = np.zeros((3, self.state_dim))
        H[:, 0:4] = H_q              # dh/dq
        H[:, 13:16] = np.eye(3)      # dh/db_accel = I

        return H

    def get_measurement_noise(self):
        """Get measurement noise covariance R."""
        return self.R

    def validate_measurement(self, x, measurement):
        """
        Validate accelerometer measurement.
        Only update if |accel| ~ g (low dynamic forces).
        """
        if measurement is None:
            return False

        accel_norm = np.linalg.norm(measurement)

        # Check if acceleration magnitude is close to gravity
        if abs(accel_norm - GRAVITY) > self.gravity_threshold:
            return False

        return True
