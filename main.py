import time
import numpy as np
import rerun as rr
from ekf import EKF
from imu_api import ImuReader

def main():
    PC_IP = "192.168.1.144"
    
    rr.init("Glider_INS_Remote", spawn=False)
    
    print(f"📡 Tentative de connexion à {PC_IP}:9876...")
    rr.connect_grpc(url=f"rerun+http://{PC_IP}:9876/proxy")
    print("✅ Connecté!")
    
    rr.log("world", rr.ViewCoordinates.RUB, static=True)
    # === LOGS TÉLÉMÉTRIE ===
    rr.log("telemetry/velocity_norm", rr.SeriesLines(colors=[[255, 0, 0]], names=["Velocity (m/s)"]), static=True)
    rr.log("telemetry/altitude", rr.SeriesLines(colors=[[0, 255, 0]], names=["Altitude (m)"]), static=True)
    rr.log("telemetry/velocity_x", rr.SeriesLines(colors=[[255, 100, 100]], names=["Vx (m/s)"]), static=True)
    rr.log("telemetry/velocity_y", rr.SeriesLines(colors=[[100, 255, 100]], names=["Vy (m/s)"]), static=True)
    rr.log("telemetry/velocity_z", rr.SeriesLines(colors=[[100, 100, 255]], names=["Vz (m/s)"]), static=True)

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

    # === LOGS INCERTITUDES (covariance) ===
    rr.log("debug/uncertainty/position", rr.SeriesLines(colors=[[255, 0, 255]], names=["Pos σ (m)"]), static=True)
    rr.log("debug/uncertainty/velocity", rr.SeriesLines(colors=[[0, 255, 255]], names=["Vel σ (m/s)"]), static=True)
    rr.log("debug/uncertainty/attitude", rr.SeriesLines(colors=[[255, 255, 0]], names=["Att σ (rad)"]), static=True)

    # === LOGS DONNÉES BRUTES ===
    rr.log("debug/accel_raw_norm", rr.SeriesLines(colors=[[0, 200, 255]], names=["Accel Norm"]), static=True)
    rr.log("debug/accel_body_x", rr.SeriesLines(colors=[[255, 50, 50]], names=["Accel X"]), static=True)
    rr.log("debug/accel_body_y", rr.SeriesLines(colors=[[50, 255, 50]], names=["Accel Y"]), static=True)
    rr.log("debug/accel_body_z", rr.SeriesLines(colors=[[50, 50, 255]], names=["Accel Z"]), static=True)
    
    
    imu = ImuReader(port="/dev/ttyS0", baudrate=115200)
    ekf = EKF(initialization_duration=5.0, sample_rate=100)
    
    last_time = time.time()
    step = 0  # ✅ Compteur pour timeline
    
    print("🚀 Démarrage du système...")
    
    with imu:
        while True:
            data = imu.read(timeout=0.1)
            if data is None:
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # ✅ CORRECTION : Créer des arrays 1D (shape (3,))
            accel = np.array([data['ax'], -data['ay'], -data['az']])  # (3,)
            gyro = np.array([data['gx'], data['gy'], data['gz']])      # (3,)
            mag = np.array([data['mx'], data['my'], data['mz']])       # (3,)
            
            imu_data = {'accel': accel, 'gyro': gyro, 'mag': mag}
            
            if not ekf.isInitialized:
                ekf.compute_initial_state(imu_data)
                step += 1
                continue
            
            # ✅ Définir timeline pour la navigation
            rr.set_time("step", sequence=step)
            step += 1
            ekf.predict(imu_data, dt)
            ekf.update(imu_data, gps_data=None, phase="glide")
            
            if step%3 == 0:
                log_to_rerun(ekf, data)


def log_to_rerun(ekf, raw_data):
    """ Centralise l'envoi des données à Rerun avec métriques EKF complètes """
    
    q = ekf.x[0:4].flatten()
    pos = ekf.x[4:7].flatten()
    vel = ekf.x[7:10].flatten()
    bg = ekf.x[10:13].flatten()
    ba = ekf.x[13:16].flatten()
    
    # ✅ Extraire Roll/Pitch/Yaw depuis quaternion
    roll, pitch, yaw = quaternion_to_euler(q)
    
    # === 1. VISUALISATION 3D ===
    rr_quat = rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])
    rr.log("world/glider", rr.Transform3D(translation=pos, rotation=rr_quat))
    rr.log("world/glider/body", rr.Boxes3D(half_sizes=[0.5, 0.2, 0.05], colors=[0, 255, 0]))
    
    # === 2. TÉLÉMÉTRIE ===
    rr.log("telemetry/velocity_norm", rr.Scalars([float(np.linalg.norm(vel))]))
    rr.log("telemetry/altitude", rr.Scalars([float(pos[2])]))
    rr.log("telemetry/velocity_x", rr.Scalars([float(vel[0])]))
    rr.log("telemetry/velocity_y", rr.Scalars([float(vel[1])]))
    rr.log("telemetry/velocity_z", rr.Scalars([float(vel[2])]))
    
    # === 3. ATTITUDE (en degrés) ===
    rr.log("attitude/roll", rr.Scalars([float(np.degrees(roll))]))
    rr.log("attitude/pitch", rr.Scalars([float(np.degrees(pitch))]))
    rr.log("attitude/yaw", rr.Scalars([float(np.degrees(yaw))]))
    
    # === 4. BIAIS ===
    rr.log("debug/bias/gyro_x", rr.Scalars([float(bg[0])]))
    rr.log("debug/bias/gyro_y", rr.Scalars([float(bg[1])]))
    rr.log("debug/bias/gyro_z", rr.Scalars([float(bg[2])]))
    rr.log("debug/bias/accel_x", rr.Scalars([float(ba[0])]))
    rr.log("debug/bias/accel_y", rr.Scalars([float(ba[1])]))
    rr.log("debug/bias/accel_z", rr.Scalars([float(ba[2])]))
    
    # === 5. INCERTITUDES (écart-types depuis covariance P) ===
    # Position uncertainty (sqrt de la trace des 3 premiers éléments)
    pos_var = np.diag(ekf.P[4:7, 4:7])
    pos_std = np.sqrt(np.mean(pos_var))
    rr.log("debug/uncertainty/position", rr.Scalars([float(pos_std)]))
    
    # Velocity uncertainty
    vel_var = np.diag(ekf.P[7:10, 7:10])
    vel_std = np.sqrt(np.mean(vel_var))
    rr.log("debug/uncertainty/velocity", rr.Scalars([float(vel_std)]))
    
    # Attitude uncertainty (quaternion variance)
    att_var = np.diag(ekf.P[0:4, 0:4])
    att_std = np.sqrt(np.mean(att_var))
    rr.log("debug/uncertainty/attitude", rr.Scalars([float(att_std)]))
    
    # === 6. DONNÉES BRUTES ===
    accel_norm = np.sqrt(raw_data['ax']**2 + raw_data['ay']**2 + raw_data['az']**2)
    rr.log("debug/accel_raw_norm", rr.Scalars([float(accel_norm)]))
    rr.log("debug/accel_body_x", rr.Scalars([float(raw_data['ax'])]))
    rr.log("debug/accel_body_y", rr.Scalars([float(raw_data['ay'])]))
    rr.log("debug/accel_body_z", rr.Scalars([float(raw_data['az'])]))

def quaternion_to_euler(q):
    """
    Convertit un quaternion [q0, q1, q2, q3] en angles d'Euler (roll, pitch, yaw)
    Convention: NED frame
    
    Returns:
        roll, pitch, yaw en radians
    """
    q0, q1, q2, q3 = q
    
    # Roll (rotation autour de X)
    roll = np.arctan2(2*(q0*q1 + q2*q3), (q0*q0+q3*q3-q1*q1-q2*q2))
    
    # Pitch (rotation autour de Y)
    sin_pitch = 2*(q0*q2 - q3*q1)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Éviter erreurs numériques
    pitch = np.arcsin(sin_pitch)
    
    # Yaw (rotation autour de Z)
    yaw = np.arctan2(2*(q0*q3 + q1*q2), (q0*q0+q1*q1-q2*q2-q3*q3))
    
    return roll, pitch, yaw

if __name__ == "__main__":
    main()