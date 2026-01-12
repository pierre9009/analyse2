"""
EKF Glide Phase Validation Test

Simulates a 60-second glide flight with:
- Ground truth trajectory generation
- Noisy sensor simulation (IMU, magnetometer, GPS)
- EKF estimation
- Rerun visualization for comparison

Convention: NED (North-East-Down) world frame, FRD (Front-Right-Down) body frame
"""

import time
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from ekf.ekf import EKF
from ekf.utils import Utils

# =============================================================================
# CONSTANTS
# =============================================================================

GRAVITY = 9.81
DT = 0.01  # 100 Hz
CALIBRATION_DURATION = 5.0  # seconds
FLIGHT_DURATION = 60.0  # seconds

# Grenoble approximate coordinates (in local NED, we start at origin)
INITIAL_POSITION = np.array([0.0, 0.0, -1000.0])  # 1000m altitude (negative in NED)

# Flight parameters
VELOCITY_NORTH = 20.0  # m/s towards North
VELOCITY_DOWN = 1.0    # m/s descent rate
ROLL_AMPLITUDE = 5.0   # degrees
ROLL_FREQUENCY = 0.1   # Hz (one cycle every 10 seconds)

# Sensor noise parameters (realistic values)
GYRO_NOISE_STD = 0.01      # rad/s
ACCEL_NOISE_STD = 0.1      # m/s²
MAG_NOISE_STD = 0.02       # normalized units
GPS_POS_NOISE_STD = 2.5    # m
GPS_VEL_NOISE_STD = 0.3    # m/s

# Sensor biases (to be injected and estimated)
TRUE_GYRO_BIAS = np.array([0.01, -0.005, 0.008])   # rad/s
TRUE_ACCEL_BIAS = np.array([0.05, -0.03, 0.02])    # m/s²

# Magnetic field reference (Grenoble region, normalized)
DECLINATION_DEG = 2.85
INCLINATION_DEG = 61.16
D = np.radians(DECLINATION_DEG)
I = np.radians(INCLINATION_DEG)
MAG_FIELD_NED = np.array([np.cos(I) * np.cos(D), np.cos(I) * np.sin(D), np.sin(I)])


# =============================================================================
# GROUND TRUTH GENERATION
# =============================================================================

def generate_ground_truth(t):
    """
    Generate ground truth state at time t.

    Args:
        t: Time in seconds

    Returns:
        dict with 'position', 'velocity', 'roll', 'pitch', 'yaw', 'quaternion',
                   'omega_body' (angular velocity in body frame)
    """
    # Position: linear motion towards North with descent
    position = INITIAL_POSITION + np.array([
        VELOCITY_NORTH * t,  # North
        0.0,                  # East
        VELOCITY_DOWN * t     # Down (positive = descending)
    ])

    # Velocity in NED
    velocity = np.array([VELOCITY_NORTH, 0.0, VELOCITY_DOWN])

    # Attitude: sinusoidal roll, constant pitch for glide, yaw = 0 (North)
    roll = np.radians(ROLL_AMPLITUDE) * np.sin(2 * np.pi * ROLL_FREQUENCY * t)
    pitch = np.radians(-3.0)  # Slight nose-down for glide
    yaw = 0.0  # Heading North

    # Angular velocity in body frame (derivative of Euler angles, simplified)
    roll_rate = np.radians(ROLL_AMPLITUDE) * 2 * np.pi * ROLL_FREQUENCY * np.cos(2 * np.pi * ROLL_FREQUENCY * t)
    pitch_rate = 0.0
    yaw_rate = 0.0

    # Convert Euler rates to body angular velocity (simplified for small angles)
    # omega_body = [p, q, r] in FRD
    omega_body = np.array([
        roll_rate - yaw_rate * np.sin(pitch),
        pitch_rate * np.cos(roll) + yaw_rate * np.sin(roll) * np.cos(pitch),
        -pitch_rate * np.sin(roll) + yaw_rate * np.cos(roll) * np.cos(pitch)
    ])

    # Quaternion from Euler angles
    quaternion = Utils.quaternion_from_euler(roll, pitch, yaw)

    return {
        'position': position,
        'velocity': velocity,
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'quaternion': quaternion,
        'omega_body': omega_body
    }


def generate_sensor_measurements(truth, add_bias=True):
    """
    Generate noisy sensor measurements from ground truth.

    Args:
        truth: Ground truth dict from generate_ground_truth()
        add_bias: Whether to add sensor biases

    Returns:
        dict with 'imu' and 'gps' measurements
    """
    q = truth['quaternion']
    R = Utils.quaternion_to_rotation_matrix(q.reshape(4, 1))  # body -> NED

    # Gravity in NED frame
    g_ned = np.array([0, 0, GRAVITY])

    # Specific force (what accelerometer measures): f = a - g
    # In steady glide, a ≈ 0, so f ≈ -g
    # Transform to body frame
    accel_true_body = R.T @ (-g_ned)  # NED -> body

    # Add centrifugal effects from rotation (small for slow rotations)
    # Simplified: just use gravity projection

    # Add noise and bias
    gyro_bias = TRUE_GYRO_BIAS if add_bias else np.zeros(3)
    accel_bias = TRUE_ACCEL_BIAS if add_bias else np.zeros(3)

    gyro_meas = truth['omega_body'] + gyro_bias + np.random.normal(0, GYRO_NOISE_STD, 3)
    accel_meas = accel_true_body + accel_bias + np.random.normal(0, ACCEL_NOISE_STD, 3)

    # Magnetometer: transform magnetic field to body frame
    mag_body_true = R.T @ MAG_FIELD_NED
    mag_meas = mag_body_true + np.random.normal(0, MAG_NOISE_STD, 3)
    # Normalize (magnetometer often outputs normalized values)
    mag_meas = mag_meas / np.linalg.norm(mag_meas)

    # GPS measurements (at lower rate, but we generate at each step)
    gps_pos = truth['position'] + np.random.normal(0, GPS_POS_NOISE_STD, 3)
    gps_vel = truth['velocity'] + np.random.normal(0, GPS_VEL_NOISE_STD, 3)

    return {
        'imu': {
            'gyro': gyro_meas,
            'accel': accel_meas,
            'mag': mag_meas
        },
        'gps': {
            'position': gps_pos,
            'velocity': gps_vel
        }
    }


def generate_calibration_measurements():
    """
    Generate measurements for stationary calibration phase.
    Planeur at rest, measuring gravity and magnetic field.
    """
    # At rest: no rotation, measuring -g in body frame
    # Assume level attitude for calibration
    roll, pitch, yaw = 0.0, 0.0, 0.0
    q = Utils.quaternion_from_euler(roll, pitch, yaw)
    R = Utils.quaternion_to_rotation_matrix(q.reshape(4, 1))

    g_ned = np.array([0, 0, GRAVITY])
    accel_true = R.T @ (-g_ned)  # Should be [0, 0, -9.81] for level

    gyro_meas = TRUE_GYRO_BIAS + np.random.normal(0, GYRO_NOISE_STD, 3)
    accel_meas = accel_true + TRUE_ACCEL_BIAS + np.random.normal(0, ACCEL_NOISE_STD, 3)

    mag_body = R.T @ MAG_FIELD_NED
    mag_meas = mag_body + np.random.normal(0, MAG_NOISE_STD, 3)
    mag_meas = mag_meas / np.linalg.norm(mag_meas)

    return {
        'imu': {
            'gyro': gyro_meas,
            'accel': accel_meas,
            'mag': mag_meas
        },
        'gps': {
            'position': INITIAL_POSITION + np.random.normal(0, GPS_POS_NOISE_STD, 3)
        }
    }


# =============================================================================
# RERUN SETUP
# =============================================================================

def setup_rerun():
    """Initialize Rerun with organized blueprints."""
    rr.init("EKF_Glide_Validation", spawn=True)

    # Set world coordinate system
    rr.log(
        "world",
        rr.ViewCoordinates(xyz=rr.components.ViewCoordinates.FRD),
        static=True
    )

    # Load glider mesh for estimation visualization
    rr.log("world/glider_est/mesh", rr.Asset3D(path="./mesh/planeur.glb"), static=True)

    # === SERIES DEFINITIONS ===

    # Attitude - Estimation
    rr.log("attitude/roll/est", rr.SeriesLines(colors=[[255, 0, 0]], names=["Roll Est"]), static=True)
    rr.log("attitude/pitch/est", rr.SeriesLines(colors=[[0, 255, 0]], names=["Pitch Est"]), static=True)
    rr.log("attitude/yaw/est", rr.SeriesLines(colors=[[0, 0, 255]], names=["Yaw Est"]), static=True)

    # Attitude - Truth
    rr.log("attitude/roll/truth", rr.SeriesLines(colors=[[255, 100, 100]], names=["Roll Truth"]), static=True)
    rr.log("attitude/pitch/truth", rr.SeriesLines(colors=[[100, 255, 100]], names=["Pitch Truth"]), static=True)
    rr.log("attitude/yaw/truth", rr.SeriesLines(colors=[[100, 100, 255]], names=["Yaw Truth"]), static=True)

    # Position - Estimation
    rr.log("position/x/est", rr.SeriesLines(colors=[[255, 0, 0]], names=["X Est"]), static=True)
    rr.log("position/y/est", rr.SeriesLines(colors=[[0, 255, 0]], names=["Y Est"]), static=True)
    rr.log("position/z/est", rr.SeriesLines(colors=[[0, 0, 255]], names=["Z Est"]), static=True)

    # Position - Truth
    rr.log("position/x/truth", rr.SeriesLines(colors=[[255, 100, 100]], names=["X Truth"]), static=True)
    rr.log("position/y/truth", rr.SeriesLines(colors=[[100, 255, 100]], names=["Y Truth"]), static=True)
    rr.log("position/z/truth", rr.SeriesLines(colors=[[100, 100, 255]], names=["Z Truth"]), static=True)

    # Velocity
    rr.log("velocity/norm/est", rr.SeriesLines(colors=[[255, 0, 0]], names=["V Est"]), static=True)
    rr.log("velocity/norm/truth", rr.SeriesLines(colors=[[255, 100, 100]], names=["V Truth"]), static=True)

    # Biases - Gyro
    rr.log("bias/gyro/x/est", rr.SeriesLines(colors=[[255, 0, 0]], names=["bg_x Est"]), static=True)
    rr.log("bias/gyro/y/est", rr.SeriesLines(colors=[[0, 255, 0]], names=["bg_y Est"]), static=True)
    rr.log("bias/gyro/z/est", rr.SeriesLines(colors=[[0, 0, 255]], names=["bg_z Est"]), static=True)
    rr.log("bias/gyro/x/truth", rr.SeriesLines(colors=[[255, 100, 100]], names=["bg_x True"]), static=True)
    rr.log("bias/gyro/y/truth", rr.SeriesLines(colors=[[100, 255, 100]], names=["bg_y True"]), static=True)
    rr.log("bias/gyro/z/truth", rr.SeriesLines(colors=[[100, 100, 255]], names=["bg_z True"]), static=True)

    # Biases - Accel
    rr.log("bias/accel/x/est", rr.SeriesLines(colors=[[255, 0, 0]], names=["ba_x Est"]), static=True)
    rr.log("bias/accel/y/est", rr.SeriesLines(colors=[[0, 255, 0]], names=["ba_y Est"]), static=True)
    rr.log("bias/accel/z/est", rr.SeriesLines(colors=[[0, 0, 255]], names=["ba_z Est"]), static=True)
    rr.log("bias/accel/x/truth", rr.SeriesLines(colors=[[255, 100, 100]], names=["ba_x True"]), static=True)
    rr.log("bias/accel/y/truth", rr.SeriesLines(colors=[[100, 255, 100]], names=["ba_y True"]), static=True)
    rr.log("bias/accel/z/truth", rr.SeriesLines(colors=[[100, 100, 255]], names=["ba_z True"]), static=True)

    # Error metrics
    rr.log("error/position_error", rr.SeriesLines(colors=[[255, 0, 255]], names=["Pos Error (m)"]), static=True)
    rr.log("error/attitude_error", rr.SeriesLines(colors=[[255, 128, 0]], names=["Att Error (deg)"]), static=True)

    # Raw sensors
    rr.log("debug/raw_sensors/accel_x", rr.SeriesLines(colors=[[255, 0, 0]], names=["Accel X"]), static=True)
    rr.log("debug/raw_sensors/accel_y", rr.SeriesLines(colors=[[0, 255, 0]], names=["Accel Y"]), static=True)
    rr.log("debug/raw_sensors/accel_z", rr.SeriesLines(colors=[[0, 0, 255]], names=["Accel Z"]), static=True)
    rr.log("debug/raw_sensors/gyro_x", rr.SeriesLines(colors=[[255, 100, 0]], names=["Gyro X"]), static=True)
    rr.log("debug/raw_sensors/gyro_y", rr.SeriesLines(colors=[[100, 255, 0]], names=["Gyro Y"]), static=True)
    rr.log("debug/raw_sensors/gyro_z", rr.SeriesLines(colors=[[0, 100, 255]], names=["Gyro Z"]), static=True)
    rr.log("debug/raw_sensors/mag_x", rr.SeriesLines(colors=[[200, 0, 200]], names=["Mag X"]), static=True)
    rr.log("debug/raw_sensors/mag_y", rr.SeriesLines(colors=[[0, 200, 200]], names=["Mag Y"]), static=True)
    rr.log("debug/raw_sensors/mag_z", rr.SeriesLines(colors=[[200, 200, 0]], names=["Mag Z"]), static=True)

    # Performance
    rr.log("performance/dt", rr.SeriesLines(colors=[[255, 128, 0]], names=["dt (ms)"]), static=True)
    rr.log("performance/predict_time", rr.SeriesLines(colors=[[255, 0, 128]], names=["Predict (ms)"]), static=True)
    rr.log("performance/update_time", rr.SeriesLines(colors=[[128, 0, 255]], names=["Update (ms)"]), static=True)

    # B_NED estimation
    rr.log("debug/B_NED/x", rr.SeriesLines(colors=[[255, 0, 0]], names=["B_x"]), static=True)
    rr.log("debug/B_NED/y", rr.SeriesLines(colors=[[0, 255, 0]], names=["B_y"]), static=True)
    rr.log("debug/B_NED/z", rr.SeriesLines(colors=[[0, 0, 255]], names=["B_z"]), static=True)

    # Create blueprint
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # Left: 3D view
            rrb.Spatial3DView(name="3D World", origin="world"),
            # Right: Multiple plots
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Roll", origin="attitude/roll"),
                    rrb.TimeSeriesView(name="Pitch", origin="attitude/pitch"),
                    rrb.TimeSeriesView(name="Yaw", origin="attitude/yaw"),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Position X", origin="position/x"),
                    rrb.TimeSeriesView(name="Position Y", origin="position/y"),
                    rrb.TimeSeriesView(name="Position Z", origin="position/z"),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Velocity", origin="velocity"),
                    rrb.TimeSeriesView(name="Position Error", origin="error"),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Gyro Bias", origin="bias/gyro"),
                    rrb.TimeSeriesView(name="Accel Bias", origin="bias/accel"),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Raw Sensors", origin="debug/raw_sensors"),
                    rrb.TimeSeriesView(name="Performance", origin="performance"),
                ),
            ),
            column_shares=[1, 3]
        ),
        collapse_panels=True
    )

    rr.send_blueprint(blueprint)


def log_to_rerun(step, ekf, truth, sensors, dt_real, t_predict, t_update):
    """Log all data to Rerun."""
    rr.set_time("step", sequence=step)

    # Extract EKF state
    q_est = ekf.x[0:4].flatten()
    pos_est = ekf.x[4:7].flatten()
    vel_est = ekf.x[7:10].flatten()
    bg_est = ekf.x[10:13].flatten()
    ba_est = ekf.x[13:16].flatten()
    B_NED_est = ekf.x[16:19].flatten()

    roll_est, pitch_est, yaw_est = Utils.quaternion_to_euler(q_est)

    # Extract truth
    pos_truth = truth['position']
    vel_truth = truth['velocity']
    roll_truth = truth['roll']
    pitch_truth = truth['pitch']
    yaw_truth = truth['yaw']
    q_truth = truth['quaternion']

    # === 3D VISUALIZATION ===

    # Estimated glider (with mesh)
    rr_quat_est = rr.Quaternion(xyzw=[q_est[1], q_est[2], q_est[3], q_est[0]])
    rr.log("world/glider_est", rr.Transform3D(translation=pos_est, rotation=rr_quat_est))

    # Truth reference frame (simple axes)
    rr_quat_truth = rr.Quaternion(xyzw=[q_truth[1], q_truth[2], q_truth[3], q_truth[0]])
    rr.log("world/glider_truth", rr.Transform3D(translation=pos_truth, rotation=rr_quat_truth))
    rr.log("world/glider_truth/axes", rr.Arrows3D(
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        radii=0.02
    ))

    # === ATTITUDE ===
    rr.log("attitude/roll/est", rr.Scalars([float(np.degrees(roll_est))]))
    rr.log("attitude/roll/truth", rr.Scalars([float(np.degrees(roll_truth))]))
    rr.log("attitude/pitch/est", rr.Scalars([float(np.degrees(pitch_est))]))
    rr.log("attitude/pitch/truth", rr.Scalars([float(np.degrees(pitch_truth))]))
    rr.log("attitude/yaw/est", rr.Scalars([float(np.degrees(yaw_est))]))
    rr.log("attitude/yaw/truth", rr.Scalars([float(np.degrees(yaw_truth))]))

    # === POSITION ===
    rr.log("position/x/est", rr.Scalars([float(pos_est[0])]))
    rr.log("position/x/truth", rr.Scalars([float(pos_truth[0])]))
    rr.log("position/y/est", rr.Scalars([float(pos_est[1])]))
    rr.log("position/y/truth", rr.Scalars([float(pos_truth[1])]))
    rr.log("position/z/est", rr.Scalars([float(pos_est[2])]))
    rr.log("position/z/truth", rr.Scalars([float(pos_truth[2])]))

    # === VELOCITY ===
    rr.log("velocity/norm/est", rr.Scalars([float(np.linalg.norm(vel_est))]))
    rr.log("velocity/norm/truth", rr.Scalars([float(np.linalg.norm(vel_truth))]))

    # === BIASES ===
    rr.log("bias/gyro/x/est", rr.Scalars([float(bg_est[0])]))
    rr.log("bias/gyro/y/est", rr.Scalars([float(bg_est[1])]))
    rr.log("bias/gyro/z/est", rr.Scalars([float(bg_est[2])]))
    rr.log("bias/gyro/x/truth", rr.Scalars([float(TRUE_GYRO_BIAS[0])]))
    rr.log("bias/gyro/y/truth", rr.Scalars([float(TRUE_GYRO_BIAS[1])]))
    rr.log("bias/gyro/z/truth", rr.Scalars([float(TRUE_GYRO_BIAS[2])]))

    rr.log("bias/accel/x/est", rr.Scalars([float(ba_est[0])]))
    rr.log("bias/accel/y/est", rr.Scalars([float(ba_est[1])]))
    rr.log("bias/accel/z/est", rr.Scalars([float(ba_est[2])]))
    rr.log("bias/accel/x/truth", rr.Scalars([float(TRUE_ACCEL_BIAS[0])]))
    rr.log("bias/accel/y/truth", rr.Scalars([float(TRUE_ACCEL_BIAS[1])]))
    rr.log("bias/accel/z/truth", rr.Scalars([float(TRUE_ACCEL_BIAS[2])]))

    # === ERROR METRICS ===
    pos_error = np.linalg.norm(pos_est - pos_truth)

    # Attitude error (simplified: sum of absolute angle differences)
    att_error = np.sqrt(
        (roll_est - roll_truth)**2 +
        (pitch_est - pitch_truth)**2 +
        (yaw_est - yaw_truth)**2
    )

    rr.log("error/position_error", rr.Scalars([float(pos_error)]))
    rr.log("error/attitude_error", rr.Scalars([float(np.degrees(att_error))]))

    # === RAW SENSORS ===
    imu = sensors['imu']
    rr.log("debug/raw_sensors/accel_x", rr.Scalars([float(imu['accel'][0])]))
    rr.log("debug/raw_sensors/accel_y", rr.Scalars([float(imu['accel'][1])]))
    rr.log("debug/raw_sensors/accel_z", rr.Scalars([float(imu['accel'][2])]))
    rr.log("debug/raw_sensors/gyro_x", rr.Scalars([float(imu['gyro'][0])]))
    rr.log("debug/raw_sensors/gyro_y", rr.Scalars([float(imu['gyro'][1])]))
    rr.log("debug/raw_sensors/gyro_z", rr.Scalars([float(imu['gyro'][2])]))
    rr.log("debug/raw_sensors/mag_x", rr.Scalars([float(imu['mag'][0])]))
    rr.log("debug/raw_sensors/mag_y", rr.Scalars([float(imu['mag'][1])]))
    rr.log("debug/raw_sensors/mag_z", rr.Scalars([float(imu['mag'][2])]))

    # === PERFORMANCE ===
    rr.log("performance/dt", rr.Scalars([float(dt_real * 1000)]))
    rr.log("performance/predict_time", rr.Scalars([float(t_predict)]))
    rr.log("performance/update_time", rr.Scalars([float(t_update)]))

    # === B_NED ===
    rr.log("debug/B_NED/x", rr.Scalars([float(B_NED_est[0])]))
    rr.log("debug/B_NED/y", rr.Scalars([float(B_NED_est[1])]))
    rr.log("debug/B_NED/z", rr.Scalars([float(B_NED_est[2])]))


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    print("=" * 60)
    print("EKF Glide Phase Validation Test")
    print("=" * 60)
    print(f"Calibration duration: {CALIBRATION_DURATION}s")
    print(f"Flight duration: {FLIGHT_DURATION}s")
    print(f"Sample rate: {1/DT:.0f} Hz")
    print(f"True gyro bias: {TRUE_GYRO_BIAS}")
    print(f"True accel bias: {TRUE_ACCEL_BIAS}")
    print("=" * 60)

    # Initialize Rerun
    setup_rerun()

    # Initialize EKF
    ekf = EKF(initialization_duration=CALIBRATION_DURATION, sample_rate=int(1/DT))

    step = 0
    sim_time = 0.0

    # =========================================================================
    # PHASE 1: CALIBRATION (stationary)
    # =========================================================================
    print("\n[PHASE 1] Calibration (planeur immobile)...")

    n_calib_samples = int(CALIBRATION_DURATION / DT)

    for i in range(n_calib_samples):
        sensors = generate_calibration_measurements()
        progress = ekf.compute_initial_state(sensors['imu'], sensors['gps'])

        if progress is not None and i % 100 == 0:
            print(f"   Calibration progress: {progress*100:.1f}%")

        step += 1

    print(f"   Calibration complete. EKF initialized: {ekf.isInitialized}")

    # =========================================================================
    # PHASE 2: GLIDE FLIGHT
    # =========================================================================
    print(f"\n[PHASE 2] Simulating {FLIGHT_DURATION}s glide flight...")

    n_flight_samples = int(FLIGHT_DURATION / DT)

    # Statistics
    pos_errors = []
    att_errors = []

    for i in range(n_flight_samples):
        t_loop_start = time.perf_counter()

        sim_time = i * DT

        # Generate ground truth
        truth = generate_ground_truth(sim_time)

        # Generate noisy sensor measurements
        sensors = generate_sensor_measurements(truth)

        # EKF Predict
        t_predict_start = time.perf_counter()
        ekf.predict(sensors['imu'], DT)
        t_predict = (time.perf_counter() - t_predict_start) * 1000  # ms

        # EKF Update (every sample for IMU, every 10 samples for GPS ~ 10Hz)
        t_update_start = time.perf_counter()

        if i % 10 == 0:  # GPS at 10 Hz
            ekf.update(sensors['imu'], sensors['gps'], phase="glide")
        else:
            ekf.update(sensors['imu'], gps_data=None, phase="glide")

        t_update = (time.perf_counter() - t_update_start) * 1000  # ms

        dt_real = time.perf_counter() - t_loop_start

        # Log to Rerun (decimated at 25 Hz)
        if i % 4 == 0:
            log_to_rerun(step, ekf, truth, sensors, dt_real, t_predict, t_update)

        # Collect statistics
        pos_est = ekf.x[4:7].flatten()
        pos_error = np.linalg.norm(pos_est - truth['position'])
        pos_errors.append(pos_error)

        q_est = ekf.x[0:4].flatten()
        roll_est, pitch_est, yaw_est = Utils.quaternion_to_euler(q_est)
        att_error = np.sqrt(
            (roll_est - truth['roll'])**2 +
            (pitch_est - truth['pitch'])**2 +
            (yaw_est - truth['yaw'])**2
        )
        att_errors.append(np.degrees(att_error))

        step += 1

        # Progress update
        if i % 1000 == 0 and i > 0:
            print(f"   t={sim_time:.1f}s | Pos err: {np.mean(pos_errors[-100:]):.2f}m | Att err: {np.mean(att_errors[-100:]):.2f} deg")

    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE - FINAL STATISTICS")
    print("=" * 60)

    print(f"\nPosition Error:")
    print(f"   Mean: {np.mean(pos_errors):.3f} m")
    print(f"   Std:  {np.std(pos_errors):.3f} m")
    print(f"   Max:  {np.max(pos_errors):.3f} m")

    print(f"\nAttitude Error:")
    print(f"   Mean: {np.mean(att_errors):.3f} deg")
    print(f"   Std:  {np.std(att_errors):.3f} deg")
    print(f"   Max:  {np.max(att_errors):.3f} deg")

    # Bias estimation comparison
    bg_est = ekf.x[10:13].flatten()
    ba_est = ekf.x[13:16].flatten()

    print(f"\nGyro Bias Estimation:")
    print(f"   True:      {TRUE_GYRO_BIAS}")
    print(f"   Estimated: {bg_est}")
    print(f"   Error:     {bg_est - TRUE_GYRO_BIAS}")

    print(f"\nAccel Bias Estimation:")
    print(f"   True:      {TRUE_ACCEL_BIAS}")
    print(f"   Estimated: {ba_est}")
    print(f"   Error:     {ba_est - TRUE_ACCEL_BIAS}")

    print("\n" + "=" * 60)
    print("Rerun visualization is running. Close viewer to exit.")
    print("=" * 60)

    # Keep Rerun open
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
