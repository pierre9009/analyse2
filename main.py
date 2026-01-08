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

    rr.log("telemetry/velocity_norm", rr.SeriesLines(names="Velocity (m/s)", color=[255, 0, 0]), static=True)
    rr.log("telemetry/altitude", rr.SeriesLines(names="Altitude (m)", color=[0, 255, 0]), static=True)

    rr.log("debug/bias/gyro_x", rr.SeriesLines(names="Bias Gyro X", color=[255, 100, 0]), static=True)
    rr.log("debug/bias/gyro_y", rr.SeriesLines(names="Bias Gyro Y", color=[255, 150, 0]), static=True)
    rr.log("debug/bias/gyro_z", rr.SeriesLines(names="Bias Gyro Z", color=[255, 200, 0]), static=True)

    rr.log("debug/bias/accel_x", rr.SeriesLines(names="Bias Gyro X", color=[255, 100, 0]), static=True)
    rr.log("debug/bias/accel_y", rr.SeriesLines(names="Bias Gyro Y", color=[255, 150, 0]), static=True)
    rr.log("debug/bias/accel_z", rr.SeriesLines(names="Bias Gyro Z", color=[255, 200, 0]), static=True)

    rr.log("debug/accel_raw_norm", rr.SeriesLines(names="Accel Norm", color=[0, 200, 255]), static=True)
    
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
    """ Centralise l'envoi des données à Rerun """
    
    q = ekf.x[0:4].flatten()    # (4,)
    #pos = ekf.x[4:7].flatten()  # (3,)
    pos = np.zeros((3,1)).flatten()
    vel = ekf.x[7:10].flatten() # (3,)
    bg = ekf.x[10:13].flatten() # (3,)
    ba = ekf.x[13:16].flatten() # (3,)
    
    # Quaternion : [x, y, z, w]
    rr_quat = rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])
    
    rr.log(
        "world/glider",
        rr.Transform3D(
            translation=pos,
            rotation=rr_quat
        )
    )
    
    rr.log(
        "world/glider/body", 
        rr.Boxes3D(half_sizes=[0.5, 0.2, 0.05], colors=[0, 255, 0])
    )
    
    # ✅ Scalars attend des listes de valeurs
    rr.log("telemetry/velocity_norm", rr.Scalars([float(np.linalg.norm(vel))]))
    rr.log("telemetry/altitude", rr.Scalars([float(pos[2])]))
    
    rr.log("debug/bias/gyro_x", rr.Scalars([float(bg[0])]))
    rr.log("debug/bias/gyro_y", rr.Scalars([float(bg[1])]))
    rr.log("debug/bias/gyro_z", rr.Scalars([float(bg[2])]))

    rr.log("debug/bias/accel_x", rr.Scalars([float(ba[0])]))
    rr.log("debug/bias/accel_y", rr.Scalars([float(ba[1])]))
    rr.log("debug/bias/accel_z", rr.Scalars([float(ba[2])]))
    
    accel_norm = np.sqrt(raw_data['ax']**2 + raw_data['ay']**2 + raw_data['az']**2)
    rr.log("debug/accel_raw_norm", rr.Scalars([float(accel_norm)]))

if __name__ == "__main__":
    main()