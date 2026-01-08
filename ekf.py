"""
Extended Kalman Filter (EKF) for quaternion-based orientation estimation.

"""

import numpy as np
from utils import Utils

GRAVITY = 9.81  # Gravitational constant (m/s^2)


class EKF:
    """
    State vector: [q0, q1, q2, q3, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
    (quaternion + position + vitesse + biais gyro et accel)
    """
    def __init__(self, initialization_duration=30.0, sample_rate=100):
        """
        Create a new EKF instance.
        
        Args:
            initialization_duration: Durée calibration en secondes
            sample_rate: Fréquence échantillonnage IMU en Hz
        """
        # État d'initialisation
        self.isInitialized = False
        self.n_samples_needed = int(initialization_duration * sample_rate)
        self._calib_gyro = []
        self._calib_accel = []
        self._calib_mag = []
        self._calib_gps = []
        
        # Initialize state vector [q0, q1, q2, q3, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
        self.x = np.zeros((16, 1))
        
        # 2. Covariance initiale (16, 16)
        self.P = np.diag([
            0.01, 0.01, 0.01, 0.01,     # quaternion
            25, 25, 100,                 # position
            0.01, 0.01, 0.01,            # vitesse
            1e-4, 1e-4, 1e-4,            # biais gyro
            2.5e-3, 2.5e-3, 2.5e-3       # biais accel
        ])
        
        # 3. Bruit de processus (16, 16)
        self.Q = np.diag([
            1e-4, 1e-4, 1e-4, 1e-4,      # quaternion
            1e-2, 1e-2, 1e-2,            # position
            5e-3, 5e-3, 5e-3,            # vitesse
            1e-8, 1e-8, 1e-8,            # biais gyro
            1e-6, 1e-6, 1e-6             # biais accel
        ])
        
        # 4. Bruits de mesure (multiples) (à ajuster selon datasheet)
        self.R_gps = np.diag([25, 25, 100, 0.25, 0.25, 0.64])
        self.R_accel = np.diag([0.04, 0.04, 0.04])
        self.R_heading = (5 * np.pi/180)**2
    
    def compute_initial_state(self, imu_data, gps_data=None):
        """
        Accumule échantillons pour calibration puis initialise l'état.
        
        Args:
            imu_data: dict avec clés 'gyro' [gx,gy,gz], 'accel' [ax,ay,az], 'mag' [mx,my,mz]
            gps_data: dict optionnel avec clé 'position' [px,py,pz]
        
        Returns:
            float: progression calibration (0.0 à 1.0), ou None si terminé
        """
            
        if self.isInitialized:
            return None
        
        # Afficher message au premier appel
        if len(self._calib_gyro) == 0:
            print("🔧 Début calibration (30s)...")
            print("   ⚠️  Ne pas bouger le planeur!")
        
        # Accumuler échantillons
        self._calib_gyro.append(imu_data['gyro'])
        self._calib_accel.append(imu_data['accel'])
        self._calib_mag.append(imu_data['mag'])
                               
        if gps_data is not None and 'position' in gps_data:
            self._calib_gps.append(gps_data['position'])
        
        
        n_samples = len(self._calib_gyro)
        
        # Vérifier si calibration complète (continue seulement si on a toute les valeurs necessaires)
        if n_samples < self.n_samples_needed:
            return n_samples / self.n_samples_needed
        
        # Calcul état initial
        gyro_data = np.array(self._calib_gyro)
        accel_data = np.array(self._calib_accel)
        mag_data = np.array(self._calib_mag)
        
        # Vérification immobilité
        gyro_std = np.std(gyro_data, axis=0)
        accel_std = np.std(accel_data, axis=0)
        
        if np.max(gyro_std) > 0.02:
            print(f"   ⚠️  Gyro a bougé pendant calibration (std={gyro_std})")
        if np.max(accel_std) > 0.15:
            print(f"   ⚠️  Accéléromètre a bougé pendant calibration (std={accel_std})")
        
        # 1. Biais gyro (moyenne)
        b_gyro = np.mean(gyro_data, axis=0)
        
        # 2. Biais accéléro (moyenne - gravité)
        accel_mean = np.mean(accel_data, axis=0)
        b_accel = accel_mean - np.array([[0], [0], [-GRAVITY]])
        
        # 3. Quaternion initial (magnéto pour yaw, roll/pitch ≈ 0)
        mag_mean = np.mean(mag_data, axis=0)
        yaw_0 = np.arctan2(mag_mean[1], mag_mean[0]) #on suppose le planeur horizontale
        
        q_0 = Utils.quaternion_from_euler(0, 0, yaw_0)
        
        # 4. Position initiale (moyenne GPS ou zéro)
        if len(self._calib_gps) > 0:
            p_0 = np.mean(self._calib_gps, axis=0)
        else:
            p_0 = np.zeros(3)
            print("   ⚠️  Pas de GPS pendant calibration, position = [0,0,0]")

        
        # 5. Construire vecteur d'état [16x1]

        self.x = np.array([
            q_0[0].item(), q_0[1].item(), q_0[2].item(), q_0[3].item(),  # quaternion (4)
            p_0[0].item(), p_0[1].item(), p_0[2].item(),           # position (3)
            0, 0, 0,                          # vitesse (3)
            b_gyro[0].item(), b_gyro[1].item(), b_gyro[2].item(),  # biais gyro (3)
            b_accel[0].item(), b_accel[1].item(), b_accel[2].item() # biais accel (3)
        ]).reshape((16, 1))
        
        self.isInitialized = True
        
        print(f"✅ Calibration terminée!")
        print(f"   Biais gyro:  [{b_gyro[0].item():.4f}, {b_gyro[1].item():.4f}, {b_gyro[2].item():.4f}] rad/s")
        print(f"   Biais accel: [{b_accel[0].item():.3f}, {b_accel[1].item():.3f}, {b_accel[2].item():.3f}] m/s²")
        print(f"   Yaw initial: {np.rad2deg(yaw_0).item():.1f}°")
        print(f"   Position:    [{p_0[0].item():.2f}, {p_0[1].item():.2f}, {p_0[2].item():.2f}]")
        
        return None


    def predict(self, imu_data, dt):
        """
        Propage l'état x de dt secondes en avant.
        
        Args:
            imu_data: dict avec 'gyro' et 'accel'
            dt: pas de temps (secondes)
        """

        q = self.x[0:4]
        p = self.x[4:7]
        v = self.x[7:10]
        b_gyro = self.x[10:13]
        b_accel = self.x[13:16]

        omega_meas = imu_data['gyro']
        accel_meas = imu_data['accel']

        omega = omega_meas - b_gyro
        accel_body = accel_meas - b_accel

        # === 2. CALCULER JACOBIENNE AVEC ÉTAT ACTUEL ===
        F = Utils.compute_jacobian_F(q, omega, accel_body, b_accel, b_gyro, dt)

        # === 3. PROPAGER COVARIANCE (Standard EKF discret) ===
        self.P = F @ self.P @ F.T + self.Q

        # === 4. PROPAGER ÉTAT ===

        # 1. Propager quaternion
        dq = 0.5 * (Utils.skew_4x4(omega) @ q)
        q_new = q + dq * dt
        q_new = q_new / np.linalg.norm(q_new)

        # 2. Propager vitesse
        R = Utils.quaternion_to_rotation_matrix(q) #body vers NED
        accel_ned = R @ accel_body
        gravity_ned = np.array([[0], [0], [GRAVITY]])
        v_new = v + (accel_ned + gravity_ned) * dt

        # 3. Propager position
        p_new = p + v * dt

        # 4. Biais constants (mais EKF les ajustera via covariance)
        b_gyro_new = b_gyro
        b_accel_new = b_accel
            
        self.x = np.vstack([q_new, p_new, v_new, b_gyro_new, b_accel_new])

        


    def update(self, imu_data, gps_data=None, phase="glide"):
        """
        Applique les updates appropriés selon les données disponibles et la phase de vol.
        
        Args:
            imu_data: dict avec 'accel' [ax, ay, az] et 'gyro' [gx, gy, gz]
            gps_data: dict optionnel avec 'position' [px,py,pz] et 'velocity' [vx,vy,vz]
            mag_data: dict optionnel avec 'mag' [mx,my,mz]
            phase: str, phase de vol ("ascension", "drop", "glide")
        """
        if not self.isInitialized:
            return
        
        # === PHASE LARGAGE : Gyro-only, PAS D'UPDATE ===
        if phase == "drop":
            return  # Ne faire aucune correction pendant le largage
        
        # === UPDATE GPS (Position + Vitesse) ===
        if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
            self.update_gps_position_velocity(gps_data)
        
        # === UPDATE ACCÉLÉROMÈTRE (Roll/Pitch via gravité) ===
        if imu_data is not None and 'accel' in imu_data:
            # Vérifier conditions : vol quasi-rectiligne
            accel_meas = imu_data['accel']
            accel_norm = np.linalg.norm(accel_meas)
            
            # GATING : Seulement si |a| ≈ g (forces dynamiques faibles)
            if abs(accel_norm - GRAVITY) < 0.5:  # Seuil 0.5 m/s²
                self.update_accel_gravity(imu_data)
        
        # === UPDATE HEADING ===
        # Priorité 1 : GPS Heading (si conditions OK)
        # Priorité 2 : Magnétomètre (sinon)
        
        heading_updated = False
        
        # Essayer GPS heading d'abord
        if gps_data is not None and 'velocity' in gps_data:
            v_gps = gps_data['velocity']
            v_horizontal = np.sqrt(v_gps[0]**2 + v_gps[1]**2)
            
            # Extraire pitch pour vérifier angle
            q = self.x[0:4]
            pitch = Utils._get_pitch_from_quaternion(q)
            
            # Conditions GPS heading :
            # 1. Vitesse horizontale suffisante (> 5 m/s)
            # 2. Pas de piqué/cabré extrême (|pitch| < 30°)
            # 3. Phase glide (pas ascension avec vitesse faible)
            if v_horizontal > 5.0 and abs(pitch) < np.radians(30) and phase == "glide":
                self.update_heading_gps(gps_data)
        
    def update_gps_position_velocity(self, gps_data):
        """
        Update EKF avec mesures GPS position + vitesse.
        
        Args:
            gps_data: dict avec 'position' [px,py,pz] et 'velocity' [vx,vy,vz] en NED
        """
        if not self.isInitialized:
            return
        
        # 1. Extraire mesure GPS (6×1)
        z = np.vstack([
            gps_data['position'],   # [px, py, pz]
            gps_data['velocity']    # [vx, vy, vz]
        ])
        
        # 2. Prédiction de la mesure h(x) = [p, v]
        h = np.vstack([
            self.x[4:7],   # position
            self.x[7:10]   # vitesse
        ])
        
        # 3. Innovation
        y = z - h
        
        # 4. Jacobienne H (6×16)
        H = np.zeros((6, 16))
        H[0:3, 4:7] = np.eye(3)
        H[3:6, 7:10] = np.eye(3)
        
        # 5. Innovation covariance
        S = H @ self.P @ H.T + self.R_gps
        
        # 6. Gain de Kalman
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 7. Update état
        self.x = self.x + K @ y
        
        # 8. Update covariance
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P
        
        # 9. Normaliser quaternion après update
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])


    def update_accel_gravity(self, imu_data):

        if not self.isInitialized:
            return
        
        q = self.x[0:4]
        q0, q1, q2, q3 = q.flatten()

        z = imu_data['accel'].reshape((3,1))

        h = Utils.quaternion_to_rotation_matrix(self.x[0:4]).T @ np.array([[0], [0], [GRAVITY]])

        y = z - h

        H_jaco = np.array([[-2*q2*GRAVITY, 2*q3*GRAVITY, -2*q0*GRAVITY, 2*q1*GRAVITY],
                      [2*q1*GRAVITY, 2*q0*GRAVITY, 2*q3*GRAVITY, 2*q2*GRAVITY],
                      [0, -4*q1*GRAVITY, -4*q2*GRAVITY, 0]])
        H = np.hstack((H_jaco, np.zeros((3,12))))

        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P
        
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])
    
    
    def update_heading_gps(self, gps_data):
        """
        Update EKF avec heading GPS (correction yaw uniquement).
        
        Args:
            gps_data: dict avec 'velocity' [vx, vy, vz] en NED
        """
        if not self.isInitialized:
            return
        
        v_gps = gps_data['velocity']
        
        
        # 3. Calculer heading GPS depuis vecteur vitesse
        z_heading = np.arctan2(v_gps[1], v_gps[0])
        z_heading = z_heading.reshape((1, 1))
        
        # 4. Calculer heading prédit depuis quaternion
        q = self.x[0:4]
        q0, q1, q2, q3 = q.flatten()
        
        h_heading = np.arctan2(
            2 * (q0*q3 + q1*q2),
            (q0**2 + q1**2 - q2**2 - q3**2) #phillips et al. 2001
        )
        h_heading = np.array([[h_heading]]) 
        
        # 5. Innovation avec wrap-around
        y = z_heading - h_heading
        
        if y > np.pi:
            y -= 2 * np.pi
        elif y < -np.pi:
            y += 2 * np.pi
        
        # 6. Jacobienne H (1×16)
        num = 2 * (q0*q3 + q1*q2)
        den = q0**2 + q1**2 - q2**2 - q3**2
        
        # Dérivée de arctan2(num, den) = (den*∂num - num*∂den) / (num² + den²)
        denom = num**2 + den**2
        
        # ∂num/∂q
        dnum_dq0 = 2*q3
        dnum_dq1 = 2*q2
        dnum_dq2 = 2*q1
        dnum_dq3 = 2*q0
        
        # ∂den/∂q
        dden_dq0 = 2*q0
        dden_dq1 = 2*q1
        dden_dq2 = -2*q2
        dden_dq3 = -2*q3
        
        # ∂yaw/∂q
        dyaw_dq0 = (den * dnum_dq0 - num * dden_dq0) / denom
        dyaw_dq1 = (den * dnum_dq1 - num * dden_dq1) / denom
        dyaw_dq2 = (den * dnum_dq2 - num * dden_dq2) / denom
        dyaw_dq3 = (den * dnum_dq3 - num * dden_dq3) / denom
        
        H = np.zeros((1, 16))
        H[0, 0] = dyaw_dq0
        H[0, 1] = dyaw_dq1
        H[0, 2] = dyaw_dq2
        H[0, 3] = dyaw_dq3
        
        # 8. Innovation covariance
        S = H @ self.P @ H.T + self.R_heading
        
        # 9. Gain de Kalman
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 10. Update état
        self.x = self.x + K @ y
        
        # 11. Update covariance
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P
        
        # 12. Normaliser quaternion
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])