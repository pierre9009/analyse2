import time
import numpy as np
import rerun as rr
from ekf import EKF  # Ton fichier EKF
from imu_driver import ImuReader # Ton driver

def main():
    # 1. Initialisation de Rerun
    PC_IP = "192.168.1.144" 
    
    rr.init("Glider_INS_Remote", spawn=False)
    
    print(f"📡 Tentative de connexion à {PC_IP}:9876...")
    rr.connect(f"{PC_IP}:9876")
    
    # Configuration du repère 3D (Z up pour la visu, ou adapté selon ton NED)
    # On définit une entité "world" pour grouper les éléments
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # 2. Initialisation des composants
    # On suppose 100Hz pour l'IMU (sample_rate=100)
    imu = ImuReader(port="/dev/ttyS0", baudrate=115200)
    ekf = EKF(initialization_duration=5.0, sample_rate=100) # 5s pour tester vite
    
    last_time = time.time()
    
    print("🚀 Démarrage du système...")
    
    with imu:
        while True:
            # Lecture brute du capteur
            data = imu.read(timeout=0.1)
            if data is None:
                continue

            # Calcul du dt réel entre deux mesures
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Préparation des vecteurs pour l'EKF (format colonne 3x1)
            accel = np.array([data['ax'], data['ay'], data['az']]).reshape(3, 1)
            gyro = np.array([data['gx'], data['gy'], data['gz']]).reshape(3, 1)
            mag = np.array([data['mx'], data['my'], data['mz']]).reshape(3, 1)
            
            imu_data = {'accel': accel, 'gyro': gyro, 'mag': mag}

            # 3. Logique de l'EKF
            if not ekf.isInitialized:
                # Phase de Calibration
                progress = ekf.compute_initial_state(imu_data)
                if progress is not None:
                    # On affiche la progression dans Rerun via un Scalar
                    rr.log("debug/calib_progress", rr.Scalar(progress * 100))
                continue

            # Phase de Navigation
            # A. Prediction
            ekf.predict(imu_data, dt)
            
            # B. Update (En intérieur : pas de GPS, on utilise l'accel pour le Roll/Pitch)
            ekf.update(imu_data, gps_data=None, phase="glide")

            # 4. Logging Rerun
            log_to_rerun(ekf, data)

def log_to_rerun(ekf, raw_data):
    """ Centralise l'envoi des données à Rerun pour la clarté """
    
    # --- État Estimé ---
    q = ekf.x[0:4].flatten()     # [q0, q1, q2, q3] -> q0 est scalaire
    pos = ekf.x[4:7].flatten()   # [px, py, pz]
    vel = ekf.x[7:10].flatten()  # [vx, vy, vz]
    bg = ekf.x[10:13].flatten()  # Biais gyro
    ba = ekf.x[13:16].flatten()  # Biais accel

    # Rerun utilise l'ordre [x, y, z, w] pour les quaternions
    rr_quat = [q[1], q[2], q[3], q[0]]

    # A. Visualisation 3D du planeur
    rr.log(
        "world/glider",
        rr.Transform3D(
            translation=pos,
            rotation=rr.Quaternion(xyzw=rr_quat)
        )
    )
    # Ajouter un cube pour représenter le corps du planeur
    rr.log("world/glider/body", rr.Boxes3D(half_sizes=[0.5, 0.2, 0.05], colors=[0, 255, 0]))

    # B. Télémétrie (Graphiques)
    rr.log("telemetry/velocity", rr.SeriesLine(name="Vitesse"), rr.Scalar(np.linalg.norm(vel)))
    rr.log("telemetry/position/z", rr.Scalar(pos[2])) # Altitude (négative si NED)

    # C. Debug Biais (Crucial pour valider l'EKF)
    rr.log("debug/bias/gyro/x", rr.Scalar(bg[0]))
    rr.log("debug/bias/gyro/y", rr.Scalar(bg[1]))
    rr.log("debug/bias/gyro/z", rr.Scalar(bg[2]))
    
    # D. Données Brutes vs Estimées (Optionnel)
    rr.log("debug/accel/raw_norm", rr.Scalar(np.linalg.norm([raw_data['ax'], raw_data['ay'], raw_data['az']])))

if __name__ == "__main__":
    main()