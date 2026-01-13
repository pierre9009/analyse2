"""
GPS Heading Update for EKF (Yaw correction from velocity vector).
"""

import numpy as np
from ekf.updates.base import UpdateBase


class HeadingGPSUpdate(UpdateBase):
    """
    Update EKF with GPS heading derived from velocity vector.

    Measurement model: z = arctan2(vy, vx)
    This corrects yaw angle only.

    Only valid when horizontal speed is sufficient (default > 2.5 m/s).
    """

    def __init__(self, R_heading=None, min_horizontal_speed=2.5):
        super().__init__(state_dim=16)

        # Default measurement noise covariance
        if R_heading is None:
            R_heading = (5 * np.pi / 180) ** 2  # (5 degrees)^2

        self.R = np.array([[R_heading]])
        self.min_horizontal_speed = min_horizontal_speed

    def compute_innovation(self, x, measurement):
        """
        Compute innovation for GPS heading.

        Args:
            x: State vector (16x1)
            measurement: GPS velocity (3x1) in NED frame

        Returns:
            tuple: (y, h) innovation and predicted measurement
        """
        v_gps = measurement

        # Measured heading from GPS velocity
        z_heading = np.arctan2(v_gps[1], v_gps[0])
        z_heading = z_heading.reshape((1, 1))

        # Predicted heading from quaternion
        q = x[0:4]
        q0, q1, q2, q3 = q.flatten()

        h_heading = np.arctan2(
            2 * (q0 * q3 + q1 * q2),
            (q0**2 + q1**2 - q2**2 - q3**2)
        )
        h_heading = np.array([[h_heading]])

        # Innovation with angle wrap-around
        y = z_heading - h_heading

        # Wrap to [-pi, pi]
        if y > np.pi:
            y -= 2 * np.pi
        elif y < -np.pi:
            y += 2 * np.pi

        return y, h_heading

    def compute_jacobian(self, x, measurement=None):
        """
        Compute Jacobian H for GPS heading.

        Analytical derivative of yaw = arctan2(num, den) where:
        num = 2*(q0*q3 + q1*q2)
        den = q0^2 + q1^2 - q2^2 - q3^2

        Args:
            x: State vector (16x1)
            measurement: Not used

        Returns:
            np.ndarray: H matrix (1x16)
        """
        q = x[0:4]
        q0, q1, q2, q3 = q.flatten()

        num = 2 * (q0 * q3 + q1 * q2)
        den = q0**2 + q1**2 - q2**2 - q3**2

        # Derivative of arctan2(num, den) = (den*dnum - num*dden) / (num^2 + den^2)
        denom = num**2 + den**2

        # d_num/d_q
        dnum_dq0 = 2 * q3
        dnum_dq1 = 2 * q2
        dnum_dq2 = 2 * q1
        dnum_dq3 = 2 * q0

        # d_den/d_q
        dden_dq0 = 2 * q0
        dden_dq1 = 2 * q1
        dden_dq2 = -2 * q2
        dden_dq3 = -2 * q3

        # d_yaw/d_q
        dyaw_dq0 = (den * dnum_dq0 - num * dden_dq0) / denom
        dyaw_dq1 = (den * dnum_dq1 - num * dden_dq1) / denom
        dyaw_dq2 = (den * dnum_dq2 - num * dden_dq2) / denom
        dyaw_dq3 = (den * dnum_dq3 - num * dden_dq3) / denom

        H = np.zeros((1, self.state_dim))
        H[0, 0] = dyaw_dq0
        H[0, 1] = dyaw_dq1
        H[0, 2] = dyaw_dq2
        H[0, 3] = dyaw_dq3

        return H

    def get_measurement_noise(self):
        """Get measurement noise covariance R."""
        return self.R

    def validate_measurement(self, x, measurement):
        """
        Validate GPS velocity for heading estimation.
        Only valid if horizontal speed > threshold.
        """
        if measurement is None:
            return False

        v_gps = measurement
        v_horizontal = np.sqrt(v_gps[0]**2 + v_gps[1]**2)

        if v_horizontal < self.min_horizontal_speed:
            return False

        return True
