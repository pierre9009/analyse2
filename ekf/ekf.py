import numpy as np
from ekf.utils import Utils
from ekf.updates import (
    GPSPositionVelocityUpdate,
    AccelGravityUpdate,
    HeadingGPSUpdate,
)

GRAVITY = 9.81


class EKF:
    """
    Extended Kalman Filter for navigation.

    State vector: [q(4), p(3), v(3), b_gyro(3), b_accel(3)]
    Total: 16 states

    - q: Quaternion [w, x, y, z] for orientation
    - p: Position in NED frame
    - v: Velocity in NED frame
    - b_gyro: Gyroscope bias
    - b_accel: Accelerometer bias
    """

    def __init__(self, initialization_duration=30.0, sample_rate=100):
        self.isInitialized = False
        self.n_samples_needed = int(initialization_duration * sample_rate)
        self._calib_gyro = []
        self._calib_accel = []
        self._calib_gps = []

        # State vector (16x1)
        self.x = np.zeros((16, 1))

        # Covariance matrix (16x16)
        self.P = np.diag([
            0.01, 0.01, 0.01, 0.01,     # quaternion
            25, 25, 100,                 # position
            0.01, 0.01, 0.01,            # velocity
            1e-4, 1e-4, 1e-4,            # gyro bias
            2.5e-3, 2.5e-3, 2.5e-3,      # accel bias
        ])

        # Process noise (16x16)
        self.Q = np.diag([
            1e-5, 1e-5, 1e-5, 1e-5,      # quaternion
            1e-2, 1e-2, 1e-2,            # position
            5e-3, 5e-3, 5e-3,            # velocity
            1e-7, 1e-7, 1e-7,            # gyro bias
            1e-8, 1e-8, 1e-8,            # accel bias
        ])

        # For quaternion continuity
        self._q_previous = None

        # Initialize update handlers
        self._gps_update = GPSPositionVelocityUpdate()
        self._accel_update = AccelGravityUpdate()
        self._heading_gps_update = HeadingGPSUpdate()

    def compute_initial_state(self, imu_data, gps_data=None):
        """
        Accumulate samples for calibration then initialize state.
        Estimates initial roll/pitch AND biases.

        Args:
            imu_data: dict with keys 'gyro' [gx,gy,gz], 'accel' [ax,ay,az]
            gps_data: optional dict with key 'position' [px,py,pz]

        Returns:
            float: calibration progress (0.0 to 1.0), or None if complete
        """
        if self.isInitialized:
            return None

        # Show message on first call
        if len(self._calib_gyro) == 0:
            print("Calibration starting (30s)...")
            print("Do not move the glider!")

        # Accumulate samples
        self._calib_gyro.append(imu_data['gyro'])
        self._calib_accel.append(imu_data['accel'])

        if gps_data is not None and 'position' in gps_data:
            self._calib_gps.append(gps_data['position'])

        n_samples = len(self._calib_gyro)

        # Check if calibration complete
        if n_samples < self.n_samples_needed:
            return n_samples / self.n_samples_needed

        # === COMPUTE INITIAL STATE ===

        gyro_data = np.array(self._calib_gyro)
        accel_data = np.array(self._calib_accel)

        # Check for movement during calibration
        gyro_std = np.std(gyro_data, axis=0)
        accel_std = np.std(accel_data, axis=0)

        if np.max(gyro_std) > 0.02:
            print(f"   Warning: Gyro moved during calibration (std={gyro_std})")
        if np.max(accel_std) > 0.15:
            print(f"   Warning: Accelerometer moved during calibration (std={accel_std})")

        # 1. GYRO BIAS (simple mean)
        b_gyro = np.mean(gyro_data, axis=0)

        # 2. INITIAL ORIENTATION from accelerometer
        accel_mean = np.mean(accel_data, axis=0)

        # Roll and Pitch from accelerometer (assumes stationary)
        gx, gy, gz = -accel_mean
        ax, ay, az = gx, gy, gz

        roll_0 = np.arctan2(ay, az)
        pitch_0 = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # Yaw init at 0 deg (no reference will be updated via gps)
        yaw_0 = 0

        # Initial quaternion
        q_0 = Utils.quaternion_from_euler(roll_0, pitch_0, yaw_0)
        self._enforce_quaternion_continuity()

        # 3. ACCELEROMETER BIAS
        R_0 = Utils.quaternion_to_rotation_matrix(q_0.reshape(4, 1))
        g_ned = np.array([0, 0, GRAVITY])
        g_body_expected = R_0.T @ g_ned

        accel_expected = -g_body_expected
        b_accel = accel_mean - accel_expected

        # 4. INITIAL POSITION
        if len(self._calib_gps) > 0:
            p_0 = np.mean(self._calib_gps, axis=0)
        else:
            p_0 = np.zeros(3)
            print("   Warning: No GPS during calibration, position = [0,0,0]")

        # 5. BUILD STATE VECTOR
        self.x = np.array([
            q_0[0], q_0[1], q_0[2], q_0[3],       # quaternion (4)
            p_0[0], p_0[1], p_0[2],               # position (3)
            0, 0, 0,                               # velocity (3)
            b_gyro[0], b_gyro[1], b_gyro[2],      # gyro bias (3)
            b_accel[0], b_accel[1], b_accel[2],   # accel bias (3)
        ]).reshape((16, 1))

        self.isInitialized = True

        # Display results
        print(f"Calibration complete!")
        print(f"   Initial orientation:")
        print(f"      Roll:  {np.rad2deg(roll_0):+7.2f} deg")
        print(f"      Pitch: {np.rad2deg(pitch_0):+7.2f} deg")
        print(f"      Yaw:   {np.rad2deg(yaw_0):+7.2f} deg")
        print(f"   Gyro bias:  [{b_gyro[0]:+.4f}, {b_gyro[1]:+.4f}, {b_gyro[2]:+.4f}] rad/s")
        print(f"   Accel bias: [{b_accel[0]:+.3f}, {b_accel[1]:+.3f}, {b_accel[2]:+.3f}] m/s^2")
        print(f"   Position:   [{p_0[0]:.2f}, {p_0[1]:.2f}, {p_0[2]:.2f}] m")

        return None

    def predict(self, imu_data, dt):
        """
        Propagate state by dt seconds using IMU data.

        Args:
            imu_data: dict with 'accel' and 'gyro' measurements
            dt: time step in seconds
        """
        # Extract states
        q = self.x[0:4]
        p = self.x[4:7]
        v = self.x[7:10]
        b_gyro = self.x[10:13]
        b_accel = self.x[13:16]

        accel_meas = np.array(imu_data['accel']).reshape((3, 1))
        omega_meas = np.array(imu_data['gyro']).reshape((3, 1))

        omega_body = omega_meas - b_gyro
        accel_body = accel_meas - b_accel

        # Compute Jacobian (16x16)
        F = Utils.compute_jacobian_F(q, omega_body, accel_body, dt)

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        # Propagate state

        # Quaternion
        dq = 0.5 * (Utils.skew_4x4(omega_body) @ q)
        q_new = q + dq * dt
        q_new = q_new / np.linalg.norm(q_new)

        # Velocity
        R = Utils.quaternion_to_rotation_matrix(q)
        accel_ned = R @ accel_body
        gravity_ned = np.array([0, 0, GRAVITY]).reshape((3, 1))
        v_new = v + (accel_ned + gravity_ned) * dt

        # Position
        p_new = p + v * dt

        # Biases remain constant
        b_gyro_new = b_gyro
        b_accel_new = b_accel

        self.x = np.vstack([q_new, p_new, v_new, b_gyro_new, b_accel_new])
        self.lock_yaw()
        self._enforce_quaternion_continuity()

    def update(self, imu_data, gps_data=None, phase="glide"):
        """
        Apply measurement updates based on available data and flight phase.

        Args:
            imu_data: dict with 'accel', 'gyro'
            gps_data: optional dict with 'position' and 'velocity'
            phase: flight phase ("ascension", "drop", "glide")
        """
        if not self.isInitialized:
            return

        if phase == "ascension":
            self._update_ascension(imu_data, gps_data)
        elif phase == "drop":
            self._update_drop(imu_data, gps_data)
        elif phase == "glide":
            self._update_glide(imu_data, gps_data)
            self.lock_yaw()

    def _update_ascension(self, imu_data, gps_data):
        """Update sequence for ascension phase."""
        # GPS position + velocity
        if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
            measurement = {
                'position': np.array(gps_data['position']).reshape((3, 1)),
                'velocity': np.array(gps_data['velocity']).reshape((3, 1))
            }
            self._apply_update(self._gps_update, measurement)

        # Accelerometer gravity (roll/pitch)
        if imu_data is not None and 'accel' in imu_data:
            accel_meas = np.array(imu_data['accel']).reshape((3, 1))
            self._apply_update(self._accel_update, accel_meas)

    def _update_drop(self, imu_data, gps_data):
        """Update sequence for drop phase (priority on roll/pitch)."""
        # Accelerometer gravity (priority for roll/pitch)
        if imu_data is not None and 'accel' in imu_data:
            accel_meas = np.array(imu_data['accel']).reshape((3, 1))
            self._apply_update(self._accel_update, accel_meas)

        # GPS position + velocity
        if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
            measurement = {
                'position': np.array(gps_data['position']).reshape((3, 1)),
                'velocity': np.array(gps_data['velocity']).reshape((3, 1))
            }
            self._apply_update(self._gps_update, measurement)

    def _update_glide(self, imu_data, gps_data):
        """Update sequence for glide phase."""
        # GPS position + velocity
        if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
            measurement = {
                'position': np.array(gps_data['position']).reshape((3, 1)),
                'velocity': np.array(gps_data['velocity']).reshape((3, 1))
            }
            self._apply_update(self._gps_update, measurement)

        # Accelerometer gravity (roll/pitch)
        if imu_data is not None and 'accel' in imu_data:
            accel_meas = np.array(imu_data['accel']).reshape((3, 1))
            self._apply_update(self._accel_update, accel_meas)

        # Heading: GPS if moving fast
        if gps_data is not None and 'velocity' in gps_data:
            v_gps = np.array(gps_data['velocity']).reshape((3, 1))
            v_horizontal = np.sqrt(v_gps[0]**2 + v_gps[1]**2)

            if v_horizontal > 2.5: # 2.5 m/s
                self._apply_update(self._heading_gps_update, v_gps)


    def _apply_update(self, update_handler, measurement):
        """
        Apply a single measurement update using the Kalman filter equations.

        Args:
            update_handler: An instance of UpdateBase subclass
            measurement: The measurement data (format depends on update type)
        """
        # Get update data (H, y, R) from handler
        update_data = update_handler.prepare_update(self.x, measurement)

        if update_data is None:
            return

        y = update_data['y']
        H = update_data['H']
        R = update_data['R']

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I_KH = np.eye(16) - K @ H

        self.P = I_KH @ self.P

        # Normalize quaternion
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])

        # Enforce quaternion continuity
        self._enforce_quaternion_continuity()

    def _enforce_quaternion_continuity(self):
        """Force quaternion to stay in the same hemisphere (q = -q)."""
        q_current = self.x[0:4].flatten()

        if self._q_previous is None:
            self._q_previous = q_current.copy()
            return

        # If dot < 0, we're in the opposite hemisphere
        if np.dot(q_current, self._q_previous) < 0:
            self.x[0:4] = -self.x[0:4]
            q_current = -q_current

        self._q_previous = q_current.copy()

    def remove_yaw_from_quaternion(self):
        q = self.x[0:4]

        roll, pitch, yaw = Utils.quaternion_to_euler(q)

        quaternion = Utils.quaternion_from_euler(roll, pitch, 0.0)

        self.x[0] = quaternion[3]
        self.x[1] = quaternion[0]
        self.x[2] = quaternion[1]
        self.x[3] = quaternion[2]
        return
    
    def lock_yaw(self):
        self.x[6] = 0.0
        self.P[6, 6] = 0.0
        self.remove_yaw_from_quaternion()
