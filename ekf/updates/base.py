"""
Base class for EKF measurement updates.
"""

import numpy as np
from abc import ABC, abstractmethod


class UpdateBase(ABC):
    """
    Abstract base class for EKF measurement updates.

    Each update type must implement:
    - compute_innovation(): returns (y, h) where y = z - h
    - compute_jacobian(): returns H matrix
    - get_measurement_noise(): returns R matrix
    """

    def __init__(self, state_dim=19):
        self.state_dim = state_dim

    @abstractmethod
    def compute_innovation(self, x, measurement):
        """
        Compute the innovation vector y = z - h(x).

        Args:
            x: State vector (19x1)
            measurement: Measurement data (format depends on update type)

        Returns:
            tuple: (y, h) where y is the innovation and h is the predicted measurement
        """
        pass

    @abstractmethod
    def compute_jacobian(self, x, measurement=None):
        """
        Compute the measurement Jacobian H = dh/dx.

        Args:
            x: State vector (19x1)
            measurement: Optional measurement data (some updates need it)

        Returns:
            np.ndarray: H matrix (measurement_dim x state_dim)
        """
        pass

    @abstractmethod
    def get_measurement_noise(self):
        """
        Get the measurement noise covariance matrix R.

        Returns:
            np.ndarray: R matrix (measurement_dim x measurement_dim)
        """
        pass

    def get_measurement_dim(self):
        """
        Get the dimension of the measurement vector.

        Returns:
            int: Measurement dimension
        """
        return self.get_measurement_noise().shape[0]

    def validate_measurement(self, x, measurement):
        """
        Validate measurement before processing.
        Override in subclasses for specific validation logic.

        Args:
            x: State vector (19x1)
            measurement: Measurement data

        Returns:
            bool: True if measurement is valid, False otherwise
        """
        return True

    def prepare_update(self, x, measurement):
        """
        Prepare all data needed for Kalman update.

        Args:
            x: State vector (19x1)
            measurement: Measurement data

        Returns:
            dict: Contains 'y' (innovation), 'H' (Jacobian), 'R' (noise covariance)
                  or None if measurement is invalid
        """
        if not self.validate_measurement(x, measurement):
            return None

        y, h = self.compute_innovation(x, measurement)
        H = self.compute_jacobian(x, measurement)
        R = self.get_measurement_noise()

        return {
            'y': y,
            'H': H,
            'R': R,
            'h': h
        }
