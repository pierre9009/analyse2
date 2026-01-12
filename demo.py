"""
Demo mode for indoor presentation (no GPS, no magnetometer heading).

This script runs a simplified EKF that:
- Estimates roll/pitch from accelerometer
- Propagates yaw from gyroscope only (will drift but OK for short demo)
- Locks position and velocity to zero (no GPS)
- No magnetometer heading update (building interference)

Usage: Run this on the glider in a building to demo attitude estimation.
"""

import time
import numpy as np
import rerun as rr
from ekf.ekf import EKF
from ekf.imu_api import ImuReader
from ekf.utils import Utils
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class DemoEKF(EKF):
    """
    Simplified EKF for indoor demo.

    - Roll/pitch from accelerometer
    - Yaw from gyroscope integration only (no magnetometer)
    - Position/velocity locked to zero
    """

    def __init__(self, initialization_duration=5.0, sample_rate=100):
        super().__init__(initialization_duration, sample_rate)

        # Override covariance for demo mode
        # Large uncertainty on position/velocity (we don't care)
        # Small uncertainty on attitude (we estimate this)
        self.P = np.diag([
            0.01, 0.01, 0.01, 0.01,     # quaternion (estimate)
            1e6, 1e6, 1e6,               # position (locked, infinite uncertainty)
            1e6, 1e6, 1e6,               # velocity (locked, infinite uncertainty)
            1e-4, 1e-4, 1e-4,            # gyro bias (estimate)
            2.5e-3, 2.5e-3, 2.5e-3,      # accel bias (estimate)
            0.01, 0.01, 0.01             # B_NED (not used but keep for compatibility)
        ])

        # Reduce process noise for demo (more stable)
        self.Q = np.diag([
            1e-6, 1e-6, 1e-6, 1e-6,      # quaternion
            0, 0, 0,                      # position (locked)
            0, 0, 0,                      # velocity (locked)
            1e-8, 1e-8, 1e-8,            # gyro bias
            1e-9, 1e-9, 1e-9,            # accel bias
            0, 0, 0                       # B_NED (not updated)
        ])

    def compute_initial_state(self, imu_data, gps_data=None):
        """
        Simplified initialization for demo mode.
        - Uses accelerometer for roll/pitch
        - Sets yaw to 0 (no magnetometer reference)
        - Position/velocity = 0
        """
        if self.isInitialized:
            return None

        if len(self._calib_gyro) == 0:
            print("Calibration demo mode (5s)...")
            print("   Keep the glider stationary!")

        self._calib_gyro.append(imu_data['gyro'])
        self._calib_accel.append(imu_data['accel'])

        n_samples = len(self._calib_gyro)

        if n_samples < self.n_samples_needed:
            return n_samples / self.n_samples_needed

        # === COMPUTE INITIAL STATE ===
        gyro_data = np.array(self._calib_gyro)
        accel_data = np.array(self._calib_accel)

        # Gyro bias
        b_gyro = np.mean(gyro_data, axis=0)

        # Roll/Pitch from accelerometer
        accel_mean = np.mean(accel_data, axis=0)
        gx, gy, gz = -accel_mean
        ax, ay, az = gx, gy, gz

        roll_0 = np.arctan2(ay, az)
        pitch_0 = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # YAW = 0 (no magnetometer reference in building)
        yaw_0 = 0.0

        # Initial quaternion
        q_0 = Utils.quaternion_from_euler(roll_0, pitch_0, yaw_0)
        self._enforce_quaternion_continuity()

        # Accelerometer bias
        GRAVITY = 9.81
        R_0 = Utils.quaternion_to_rotation_matrix(q_0.reshape(4, 1))
        g_ned = np.array([0, 0, GRAVITY])
        g_body_expected = R_0.T @ g_ned
        accel_expected = -g_body_expected
        b_accel = accel_mean - accel_expected

        # Position/velocity = 0 (no GPS)
        p_0 = np.zeros(3)
        v_0 = np.zeros(3)

        # Build state vector
        self.x = np.array([
            q_0[0], q_0[1], q_0[2], q_0[3],
            p_0[0], p_0[1], p_0[2],
            v_0[0], v_0[1], v_0[2],
            b_gyro[0], b_gyro[1], b_gyro[2],
            b_accel[0], b_accel[1], b_accel[2],
            self.mag_ref_init[0, 0], self.mag_ref_init[1, 0], self.mag_ref_init[2, 0]
        ]).reshape((19, 1))

        self.isInitialized = True

        print(f"Calibration complete!")
        print(f"   Initial orientation:")
        print(f"      Roll:  {np.rad2deg(roll_0):+7.2f} deg")
        print(f"      Pitch: {np.rad2deg(pitch_0):+7.2f} deg")
        print(f"      Yaw:   {np.rad2deg(yaw_0):+7.2f} deg (fixed to 0)")
        print(f"   Gyro bias:  [{b_gyro[0]:+.4f}, {b_gyro[1]:+.4f}, {b_gyro[2]:+.4f}] rad/s")
        print(f"   Accel bias: [{b_accel[0]:+.3f}, {b_accel[1]:+.3f}, {b_accel[2]:+.3f}] m/s^2")
        print(f"   Position/Velocity: LOCKED to 0 (no GPS)")

        return None

    def predict(self, imu_data, dt):
        """
        Predict step with position/velocity locked to zero.
        Only propagates quaternion and biases.
        """
        # Extract states
        q = self.x[0:4]
        b_gyro = self.x[10:13]
        b_accel = self.x[13:16]
        B_NED = self.x[16:19]

        accel_meas = np.array(imu_data['accel']).reshape((3, 1))
        omega_meas = np.array(imu_data['gyro']).reshape((3, 1))

        omega_body = omega_meas - b_gyro
        accel_body = accel_meas - b_accel

        # Compute Jacobian (19x19)
        F = Utils.compute_jacobian_F_extended(q, omega_body, accel_body, dt)

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        # Propagate quaternion only (yaw will drift but that's OK)
        dq = 0.5 * (Utils.skew_4x4(omega_body) @ q)
        q_new = q + dq * dt
        q_new = q_new / np.linalg.norm(q_new)

        # Position/velocity stay at zero
        p_new = np.zeros((3, 1))
        v_new = np.zeros((3, 1))

        # Biases remain constant
        b_gyro_new = b_gyro
        b_accel_new = b_accel
        B_NED_new = B_NED

        self.x = np.vstack([q_new, p_new, v_new, b_gyro_new, b_accel_new, B_NED_new])

        self._enforce_quaternion_continuity()

    def update_demo(self, imu_data):
        """
        Demo update: only accelerometer for roll/pitch.
        No magnetometer (building interference).
        No GPS (indoor).
        """
        if not self.isInitialized:
            return

        # Only accelerometer gravity update (roll/pitch)
        if imu_data is not None and 'accel' in imu_data:
            accel_meas = np.array(imu_data['accel']).reshape((3, 1))
            self._apply_update(self._accel_update, accel_meas)

        # Force position/velocity to zero after update
        self.x[4:7] = 0  # position
        self.x[7:10] = 0  # velocity


def main():
    #windows(cmd): ipconfig ,linux: ifconfig
    PC_IP = config['network']['IP']

    rr.init("Glider_DEMO", spawn=False)

    print(f"Tentative de connexion a {PC_IP}:9876...")
    rr.connect_grpc(url=f"rerun+http://{PC_IP}:9876/proxy")
    print("Connecte!")

    # Setup 3D view
    rr.log(
        "world",
        rr.ViewCoordinates(xyz=rr.components.ViewCoordinates.FRD),
        static=True
    )
    rr.log("world/glider/mesh", rr.Asset3D(path="./mesh/planeur.glb"), static=True)

    # === LOGS ATTITUDE ===
    rr.log("attitude/roll", rr.SeriesLines(colors=[[255, 0, 0]], names=["Roll (deg)"]), static=True)
    rr.log("attitude/pitch", rr.SeriesLines(colors=[[0, 255, 0]], names=["Pitch (deg)"]), static=True)
    rr.log("attitude/yaw", rr.SeriesLines(colors=[[0, 0, 255]], names=["Yaw (deg)"]), static=True)

    # === LOGS BIAIS ===
    rr.log("debug/bias/gyro_x", rr.SeriesLines(colors=[[255, 100, 0]], names=["Bias Gyro X"]), static=True)
    rr.log("debug/bias/gyro_y", rr.SeriesLines(colors=[[255, 150, 0]], names=["Bias Gyro Y"]), static=True)
    rr.log("debug/bias/gyro_z", rr.SeriesLines(colors=[[255, 200, 0]], names=["Bias Gyro Z"]), static=True)
    rr.log("debug/bias/accel_x", rr.SeriesLines(colors=[[100, 255, 0]], names=["Bias Accel X"]), static=True)
    rr.log("debug/bias/accel_y", rr.SeriesLines(colors=[[150, 255, 0]], names=["Bias Accel Y"]), static=True)
    rr.log("debug/bias/accel_z", rr.SeriesLines(colors=[[200, 255, 0]], names=["Bias Accel Z"]), static=True)

    # === LOGS DONNEES BRUTES ===
    rr.log("debug/accel_raw_norm", rr.SeriesLines(colors=[[0, 200, 255]], names=["Accel Norm"]), static=True)
    rr.log("debug/gyro_x", rr.SeriesLines(colors=[[255, 50, 50]], names=["Gyro X"]), static=True)
    rr.log("debug/gyro_y", rr.SeriesLines(colors=[[50, 255, 50]], names=["Gyro Y"]), static=True)
    rr.log("debug/gyro_z", rr.SeriesLines(colors=[[50, 50, 255]], names=["Gyro Z"]), static=True)

    # === LOGS PERFORMANCE ===
    rr.log("debug/performance/dt", rr.SeriesLines(colors=[[255, 128, 0]], names=["dt (ms)"]), static=True)

    imu = ImuReader(port="/dev/ttyS0", baudrate=115200)
    ekf = DemoEKF(initialization_duration=5.0, sample_rate=100)

    last_time = None
    step = 0

    print("===========================================")
    print("   DEMO MODE - Indoor presentation")
    print("   - Roll/Pitch: Accelerometer")
    print("   - Yaw: Gyro only (will drift)")
    print("   - Position/Velocity: LOCKED to 0")
    print("   - No magnetometer (building interference)")
    print("===========================================")
    print("")
    print("Starting system...")

    with imu:
        while True:
            data = imu.read(timeout=0.1)
            if data is None:
                continue

            current_time = time.time()

            if last_time is None:
                last_time = current_time
                continue

            dt = current_time - last_time

            if dt > 0.05 or dt < 0.001:
                print(f"dt anormal: {dt*1000:.1f}ms")
                last_time = current_time
                continue

            last_time = current_time

            # Mapping capteurs
            accel = np.array([data['ax'], -data['ay'], -data['az']])
            gyro = np.array([data['gx'], -data['gy'], -data['gz']])

            imu_data = {'accel': accel, 'gyro': gyro}

            # Calibration
            if not ekf.isInitialized:
                ekf.compute_initial_state(imu_data)
                step += 1
                continue

            rr.set_time("step", sequence=step)
            step += 1

            # EKF Predict
            ekf.predict(imu_data, dt)

            # EKF Update (demo mode - accel only)
            ekf.update_demo(imu_data)

            # Logging (25 Hz)
            if step % 4 == 0:
                log_to_rerun(ekf, data, dt)


def log_to_rerun(ekf, raw_data, dt):
    """Send data to Rerun visualization."""

    q = ekf.x[0:4].flatten()
    pos = np.zeros(3)  # Always zero in demo mode
    bg = ekf.x[10:13].flatten()
    ba = ekf.x[13:16].flatten()

    roll, pitch, yaw = Utils.quaternion_to_euler(q)

    # === 1. 3D VISUALIZATION ===
    rr_quat = rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])
    rr.log("world/glider", rr.Transform3D(translation=pos, rotation=rr_quat))

    # === 2. ATTITUDE (degrees) ===
    rr.log("attitude/roll", rr.Scalars([float(np.degrees(roll))]))
    rr.log("attitude/pitch", rr.Scalars([float(np.degrees(pitch))]))
    rr.log("attitude/yaw", rr.Scalars([float(np.degrees(yaw))]))

    # === 3. BIASES ===
    rr.log("debug/bias/gyro_x", rr.Scalars([float(bg[0])]))
    rr.log("debug/bias/gyro_y", rr.Scalars([float(bg[1])]))
    rr.log("debug/bias/gyro_z", rr.Scalars([float(bg[2])]))
    rr.log("debug/bias/accel_x", rr.Scalars([float(ba[0])]))
    rr.log("debug/bias/accel_y", rr.Scalars([float(ba[1])]))
    rr.log("debug/bias/accel_z", rr.Scalars([float(ba[2])]))

    # === 4. RAW DATA ===
    accel_norm = np.sqrt(raw_data['ax']**2 + raw_data['ay']**2 + raw_data['az']**2)
    rr.log("debug/accel_raw_norm", rr.Scalars([float(accel_norm)]))
    rr.log("debug/gyro_x", rr.Scalars([float(raw_data['gx'])]))
    rr.log("debug/gyro_y", rr.Scalars([float(raw_data['gy'])]))
    rr.log("debug/gyro_z", rr.Scalars([float(raw_data['gz'])]))

    # === 5. PERFORMANCE ===
    rr.log("debug/performance/dt", rr.Scalars([float(dt * 1000)]))


if __name__ == "__main__":
    main()
