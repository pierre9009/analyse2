import time
import numpy as np
import rerun as rr
from ekf.ekf import EKF
from ekf.imu_api import ImuReader
from ekf.utils import Utils
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    #windows(cmd): ipconfig ,linux: ifconfig
    PC_IP = config['network']['IP']
    
    rr.init("Glider_INS_Remote", spawn=False)
    
    print(f"📡 Tentative de connexion à {PC_IP}:9876...")
    rr.connect_grpc(url=f"rerun+http://{PC_IP}:9876/proxy")
    print("✅ Connecté!")
    
    rr.log(
        "world",
        rr.ViewCoordinates(xyz=rr.components.ViewCoordinates.FRD),
        static=True
    )
    rr.log("world/glider/mesh", rr.Asset3D(path="./mesh/planeur.glb"), static=True)
    
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

    rr.log("debug/bias/B_NED_x", rr.SeriesLines(colors=[[100, 255, 0]], names=["mag X"]), static=True)
    rr.log("debug/bias/B_NED_y", rr.SeriesLines(colors=[[150, 255, 0]], names=["mag Y"]), static=True)
    rr.log("debug/bias/B_NED_z", rr.SeriesLines(colors=[[200, 255, 0]], names=["mag Z"]), static=True)
    rr.log("debug/bias/B_NED_norm", rr.SeriesLines(colors=[[200, 255, 0]], names=["Bias mag norm"]), static=True)


    # === LOGS INCERTITUDES ===
    rr.log("debug/uncertainty/position", rr.SeriesLines(colors=[[255, 0, 255]], names=["Pos σ (m)"]), static=True)
    rr.log("debug/uncertainty/velocity", rr.SeriesLines(colors=[[0, 255, 255]], names=["Vel σ (m/s)"]), static=True)
    rr.log("debug/uncertainty/attitude", rr.SeriesLines(colors=[[255, 255, 0]], names=["Att σ (rad)"]), static=True)

    # === LOGS DONNÉES BRUTES ===
    rr.log("debug/accel_raw_norm", rr.SeriesLines(colors=[[0, 200, 255]], names=["Accel Norm"]), static=True)
    rr.log("debug/accel_body_x", rr.SeriesLines(colors=[[255, 50, 50]], names=["Accel X"]), static=True)
    rr.log("debug/accel_body_y", rr.SeriesLines(colors=[[50, 255, 50]], names=["Accel Y"]), static=True)
    rr.log("debug/accel_body_z", rr.SeriesLines(colors=[[50, 50, 255]], names=["Accel Z"]), static=True)
    
    # === LOGS PERFORMANCE ===
    rr.log("debug/performance/dt", rr.SeriesLines(colors=[[255, 128, 0]], names=["dt (ms)"]), static=True)
    rr.log("debug/performance/ekf_predict", rr.SeriesLines(colors=[[255, 0, 128]], names=["Predict (ms)"]), static=True)
    rr.log("debug/performance/ekf_update", rr.SeriesLines(colors=[[128, 0, 255]], names=["Update (ms)"]), static=True)
    rr.log("debug/performance/logging", rr.SeriesLines(colors=[[0, 128, 255]], names=["Logging (ms)"]), static=True)
    rr.log("debug/performance/total", rr.SeriesLines(colors=[[255, 255, 0]], names=["Total (ms)"]), static=True)
    
    imu = ImuReader(port="/dev/ttyS0", baudrate=115200)
    ekf = EKF(initialization_duration=5.0, sample_rate=100)
    
    last_time = None
    step = 0
    
    # Statistiques
    timing_stats = {
        'dt': [],
        'predict': [],
        'update': [],
        'logging': [],
        'total': []
    }
    
    print("🚀 Démarrage du système...")
    
    with imu:
        while True:
            t_loop_start = time.perf_counter()
            
            data = imu.read(timeout=0.1)
            if data is None:
                continue
            
            current_time = time.time()
            
            # ✅ Gérer premier passage
            if last_time is None:
                last_time = current_time
                continue
            
            dt = current_time - last_time
            
            # ✅ Safety check dt
            if dt > 0.05 or dt < 0.001:  # Hors plage 20-1000 Hz
                print(f"⚠️ dt anormal: {dt*1000:.1f}ms")
                last_time = current_time
                continue
            
            last_time = current_time
            
            # ═══════════════════════════════════════════════════
            # MAPPING CAPTEURS
            # ═══════════════════════════════════════════════════
            accel = np.array([data['ax'], -data['ay'], -data['az']])
            gyro = np.array([data['gx'], -data['gy'], -data['gz']])
            mag = np.array([data['mx'], data['my'], data['mz']])
            
            imu_data = {'accel': accel, 'gyro': gyro, 'mag': mag}
            
            # ═══════════════════════════════════════════════════
            # CALIBRATION
            # ═══════════════════════════════════════════════════
            if not ekf.isInitialized:
                ekf.compute_initial_state(imu_data)
                step += 1
                continue
            
            rr.set_time("step", sequence=step)
            step += 1
            
            # ═══════════════════════════════════════════════════
            # EKF PREDICT
            # ═══════════════════════════════════════════════════
            t_predict_start = time.perf_counter()
            ekf.predict(imu_data, dt)
            t_predict = (time.perf_counter() - t_predict_start) * 1000  # ms
            
            # ═══════════════════════════════════════════════════
            # EKF UPDATE
            # ═══════════════════════════════════════════════════
            t_update_start = time.perf_counter()
            ekf.update(imu_data, gps_data=None, phase="glide")
            t_update = (time.perf_counter() - t_update_start) * 1000  # ms
            
            # ═══════════════════════════════════════════════════
            # LOGGING (décimé 25 Hz)
            # ═══════════════════════════════════════════════════
            t_logging = 0
            if step % 4 == 0:
                t_logging_start = time.perf_counter()
                log_to_rerun(ekf, data, dt, t_predict, t_update)
                t_logging = (time.perf_counter() - t_logging_start) * 1000  # ms
            
            # ═══════════════════════════════════════════════════
            # STATS PERFORMANCE
            # ═══════════════════════════════════════════════════
            t_total = (time.perf_counter() - t_loop_start) * 1000  # ms
            
            timing_stats['dt'].append(dt * 1000)
            timing_stats['predict'].append(t_predict)
            timing_stats['update'].append(t_update)
            timing_stats['logging'].append(t_logging)
            timing_stats['total'].append(t_total)
            
            # Affichage périodique
            if step % 1000 == 0:
                print(f"\n📊 Performance Stats (dernières 1000 itérations):")
                print(f"   dt:      {np.mean(timing_stats['dt']):.2f} ± {np.std(timing_stats['dt']):.2f} ms")
                print(f"   Predict: {np.mean(timing_stats['predict']):.2f} ± {np.std(timing_stats['predict']):.2f} ms")
                print(f"   Update:  {np.mean(timing_stats['update']):.2f} ± {np.std(timing_stats['update']):.2f} ms")
                print(f"   Logging: {np.mean(timing_stats['logging']):.2f} ± {np.std(timing_stats['logging']):.2f} ms")
                print(f"   Total:   {np.mean(timing_stats['total']):.2f} ± {np.std(timing_stats['total']):.2f} ms")
                print(f"   Capacité: {1000 / np.mean(timing_stats['total']):.1f} Hz")
                
                # Reset stats
                for key in timing_stats:
                    timing_stats[key] = []


def log_to_rerun(ekf, raw_data, dt, t_predict, t_update):
    """Centralise l'envoi des données à Rerun avec métriques EKF complètes"""
    
    q = ekf.x[0:4].flatten()
    pos = np.zeros(3)  # Position fixe pour l'instant
    vel = ekf.x[7:10].flatten()
    bg = ekf.x[10:13].flatten()
    ba = ekf.x[13:16].flatten()
    
    roll, pitch, yaw = Utils.quaternion_to_euler(q)
    
    # === 1. VISUALISATION 3D ===
    rr_quat = rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])
    rr.log("world/glider", rr.Transform3D(translation=pos, rotation=rr_quat))
    
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
    B_NED = ekf.x[16:19].flatten()

    rr.log("debug/mag/B_NED_x", rr.Scalars([float(B_NED[0])]))
    rr.log("debug/mag/B_NED_y", rr.Scalars([float(B_NED[1])]))
    rr.log("debug/mag/B_NED_z", rr.Scalars([float(B_NED[2])]))
    rr.log("debug/mag/B_NED_norm", rr.Scalars([float(np.linalg.norm(B_NED))]))
    
    # === 5. INCERTITUDES ===
    pos_var = np.diag(ekf.P[4:7, 4:7])
    pos_std = np.sqrt(np.mean(pos_var))
    rr.log("debug/uncertainty/position", rr.Scalars([float(pos_std)]))
    
    vel_var = np.diag(ekf.P[7:10, 7:10])
    vel_std = np.sqrt(np.mean(vel_var))
    rr.log("debug/uncertainty/velocity", rr.Scalars([float(vel_std)]))
    
    att_var = np.diag(ekf.P[0:4, 0:4])
    att_std = np.sqrt(np.mean(att_var))
    rr.log("debug/uncertainty/attitude", rr.Scalars([float(att_std)]))
    
    # === 6. DONNÉES BRUTES ===
    accel_norm = np.sqrt(raw_data['ax']**2 + raw_data['ay']**2 + raw_data['az']**2)
    rr.log("debug/accel_raw_norm", rr.Scalars([float(accel_norm)]))
    rr.log("debug/accel_body_x", rr.Scalars([float(raw_data['ax'])]))
    rr.log("debug/accel_body_y", rr.Scalars([float(raw_data['ay'])]))
    rr.log("debug/accel_body_z", rr.Scalars([float(raw_data['az'])]))
    
    # === 7. PERFORMANCE ===
    rr.log("debug/performance/dt", rr.Scalars([float(dt * 1000)]))  # ms
    rr.log("debug/performance/ekf_predict", rr.Scalars([float(t_predict)]))
    rr.log("debug/performance/ekf_update", rr.Scalars([float(t_update)]))


if __name__ == "__main__":
    main()