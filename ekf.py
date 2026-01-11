"""
Extended Kalman Filter (EKF) for quaternion-based orientation estimation.

"""

import numpy as np
from utils import Utils

GRAVITY = 9.81  # Gravitational constant (m/s^2)


class EKF:
    """
    State vector: [q0, q1, q2, q3, mx, my, mz, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
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
        self._q_previous = None  # Pour continuité
        
        # 2. Covariance initiale (16, 16)
        self.P = np.diag([
            0.01, 0.01, 0.01, 0.01,     # quaternion
            25, 25, 100,                 # position
            0.01, 0.01, 0.01,            # vitesse
            1e-4, 1e-4, 1e-4,            # biais gyro
            2.5e-3, 2.5e-3, 2.5e-3       # biais accel
        ])
        assert self.P.shape == (16, 16), f"Erreur x: shape attendue (16, 16), obtenue {self.P.shape}"
        
        # 3. Bruit de processus (16, 16)
        # Bias process noise increased to allow faster adaptation
        self.Q = np.diag([
            1e-5, 1e-5, 1e-5, 1e-5,      # quaternion
            1e-2, 1e-2, 1e-2,            # position
            5e-3, 5e-3, 5e-3,            # vitesse
            1e-6, 1e-6, 1e-6,            # biais gyro (increased from 1e-8)
            1e-4, 1e-4, 1e-4             # biais accel (increased from 1e-6)
        ])
        assert self.Q.shape == (16, 16), f"Erreur Q: shape attendue (16, 16), obtenue {self.Q.shape}"
        
        # 4. Bruits de mesure (multiples) (à ajuster selon datasheet)
        self.R_gps = np.diag([25, 25, 100, 0.25, 0.25, 0.64])
        self.R_accel = np.diag([0.5, 0.5, 0.5])
        self.R_heading_gps = (20 * np.pi/180)**2   # GPS heading noise (~5 deg)
        self.R_heading_mag = (30 * np.pi/180)**2  # Magnetometer heading noise (~10 deg)

        declination_deg = 2.85
        inclination_deg = 61.16
        D = np.radians(declination_deg)
        I = np.radians(inclination_deg)

        self.mag_ref = np.array([np.cos(I)*np.cos(D), np.cos(I)*np.sin(D), np.sin(I)]).reshape((3,1))
    
    def compute_initial_state(self, imu_data, gps_data=None):
        """
        Accumule échantillons pour calibration puis initialise l'état.
        Estime roll/pitch/yaw initial ET les biais même si le planeur est incliné.
        
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
        
        # Vérifier si calibration complète
        if n_samples < self.n_samples_needed:
            return n_samples / self.n_samples_needed
        
        # ═══════════════════════════════════════════════════════════
        # CALCUL ÉTAT INITIAL
        # ═══════════════════════════════════════════════════════════
        
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
        
        # ───────────────────────────────────────────────────────────
        # 1. BIAIS GYROSCOPE (simple moyenne)
        # ───────────────────────────────────────────────────────────
        b_gyro = np.mean(gyro_data, axis=0)
        
        # ───────────────────────────────────────────────────────────
        # 2. ORIENTATION INITIALE depuis accéléromètre + magnétomètre
        # ───────────────────────────────────────────────────────────
        accel_mean = np.mean(accel_data, axis=0)
        mag_mean = np.mean(mag_data, axis=0)
        
        # Roll et Pitch depuis accéléromètre (suppose immobile)
        # Accéléromètre mesure -g en body frame
        gx, gy, gz = -accel_mean
        ax, ay, az = gx, gy, gz
        
        roll_0 = np.arctan2(ay, az)  # Rotation autour de X
        pitch_0 = np.arctan2(-ax, np.sqrt(ay**2 + az**2))  # Rotation autour de Y
        
        # Yaw depuis magnétomètre avec compensation tilt
        # Formule tilt-compensée standard
        mx, my, mz = mag_mean
        
        mag_x_comp = mx * np.cos(pitch_0) + mz * np.sin(pitch_0)
        mag_y_comp = (mx * np.sin(roll_0) * np.sin(pitch_0) +
                    my * np.cos(roll_0) -
                    mz * np.sin(roll_0) * np.cos(pitch_0))
        
        yaw_0 = np.arctan2(-mag_y_comp, mag_x_comp)  # Convention NED
        
        # Quaternion initial
        q_0 = Utils.quaternion_from_euler(roll_0, pitch_0, yaw_0)
        self._enforce_quaternion_continuity()
        
        # ───────────────────────────────────────────────────────────
        # 3. BIAIS ACCÉLÉROMÈTRE
        # ───────────────────────────────────────────────────────────
        # Gravité attendue en body frame avec l'orientation calculée
        R_0 = Utils.quaternion_to_rotation_matrix(q_0.reshape(4, 1))
        g_ned = np.array([0, 0, GRAVITY])
        g_body_expected = R_0.T @ g_ned  # NED → body
        
        # Accéléromètre mesure -g_body (force spécifique)
        accel_expected = -g_body_expected
        
        # Biais = mesure - attendu
        b_accel = accel_mean - accel_expected
        
        # ───────────────────────────────────────────────────────────
        # 4. POSITION INITIALE
        # ───────────────────────────────────────────────────────────
        if len(self._calib_gps) > 0:
            p_0 = np.mean(self._calib_gps, axis=0)
        else:
            p_0 = np.zeros(3)
            print("   ⚠️  Pas de GPS pendant calibration, position = [0,0,0]")
        
        # ───────────────────────────────────────────────────────────
        # 5. CONSTRUIRE VECTEUR D'ÉTAT
        # ───────────────────────────────────────────────────────────
        self.x = np.array([
            q_0[0], q_0[1], q_0[2], q_0[3],  # quaternion (4)
            p_0[0], p_0[1], p_0[2],           # position (3)
            0, 0, 0,                           # vitesse (3)
            b_gyro[0], b_gyro[1], b_gyro[2],  # biais gyro (3)
            b_accel[0], b_accel[1], b_accel[2] # biais accel (3)
        ]).reshape((16, 1))
        
        self.isInitialized = True
        
        # ───────────────────────────────────────────────────────────
        # AFFICHAGE RÉSULTATS
        # ───────────────────────────────────────────────────────────
        print(f"✅ Calibration terminée!")
        print(f"   Orientation initiale:")
        print(f"      Roll:  {np.rad2deg(roll_0):+7.2f}°")
        print(f"      Pitch: {np.rad2deg(pitch_0):+7.2f}°")
        print(f"      Yaw:   {np.rad2deg(yaw_0):+7.2f}°")
        print(f"   Biais gyro:  [{b_gyro[0]:+.4f}, {b_gyro[1]:+.4f}, {b_gyro[2]:+.4f}] rad/s")
        print(f"   Biais accel: [{b_accel[0]:+.3f}, {b_accel[1]:+.3f}, {b_accel[2]:+.3f}] m/s²")
        print(f"   Position:    [{p_0[0]:.2f}, {p_0[1]:.2f}, {p_0[2]:.2f}] m")
        
        return None


    def predict(self, imu_data, dt):
        """
        Propage l'état x de dt secondes en avant.
        
        Args:
            imu_data: dict avec 'gyro' et 'accel'
            dt: pas de temps (secondes)
        """

        # === 1. PREPARER LA DATA D'ENTRE ===
        q = self.x[0:4]
        assert q.shape == (4, 1)

        p = self.x[4:7]
        assert p.shape == (3, 1)

        v = self.x[7:10]
        assert v.shape == (3, 1)

        b_gyro = self.x[10:13]
        assert b_gyro.shape == (3, 1)

        b_accel = self.x[13:16]
        assert b_accel.shape == (3, 1)

        accel_meas = np.array(imu_data['accel']).reshape((3,1))
        omega_meas = np.array(imu_data['gyro']).reshape((3,1))
        

        omega_body = omega_meas - b_gyro
        accel_body = accel_meas - b_accel

        # === 2. CALCULER JACOBIENNE AVEC ÉTAT ACTUEL ===
        F = Utils.compute_jacobian_F(q, omega_body, accel_body, dt)

        # === 3. PROPAGER COVARIANCE ===
        self.P = F @ self.P @ F.T + self.Q

        # === 4. PROPAGER ÉTAT ===

        # 1. Propager quaternion
        dq = 0.5 * (Utils.skew_4x4(omega_body) @ q)
        q_new = q + dq * dt
        q_new = q_new / np.linalg.norm(q_new)

        # 2. Propager vitesse
        R = Utils.quaternion_to_rotation_matrix(q) #body vers NED
        accel_ned = R @ accel_body
        gravity_ned = np.array([0, 0, GRAVITY]).reshape((3,1))
        v_new = v + (accel_ned + gravity_ned) * dt      #Immobile: on mesure la reaction à la gravité donc on ajoute la gravité pour faire 0 d'acceleration

        # 3. Propager position
        p_new = p + v * dt

        # 4. Biais constants (mais EKF les ajustera via covariance)
        b_gyro_new = b_gyro
        b_accel_new = b_accel

        assert q_new.shape == (4, 1)
        assert p_new.shape == (3, 1)
        assert v_new.shape == (3, 1)
        assert b_gyro_new.shape == (3, 1)
        assert b_accel_new.shape == (3, 1)
            
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
        
        # === PHASE LARGAGE ===
        if phase == "ascension":
            #GPS position et vitesse
            if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
                position = np.array(gps_data['position']).reshape((3,1))
                velocity = np.array(gps_data['velocity']).reshape((3,1))

                self.update_gps_position_velocity(position, velocity)

            #Accel: correction roll/pitch
            if imu_data is not None and 'accel' in imu_data:
                accel_meas = np.array(imu_data['accel']).reshape((3,1))
                accel_norm = np.linalg.norm(accel_meas)

                # Seulement si |a| ≈ g (forces dynamiques faibles, on se raproche de la gravité seul)
                if abs(accel_norm - GRAVITY) < 0.5:  # Seuil 0.5 m/s²
                    self.update_accel_gravity(accel_meas)

            #Magnetometre : correction du cap 
            if imu_data is not None and 'mag' in imu_data:
                mag = np.array(imu_data['mag']).reshape((3,1))
                self.update_heading_mag(mag)
            return
        
        # === PHASE LARGAGE ===
        if phase == "drop":
            # Priorité roll pitch

            # UPDATE ACCÉLÉROMÈTRE (Roll/Pitch via gravité)
            if imu_data is not None and 'accel' in imu_data:
                accel_meas = np.array(imu_data['accel']).reshape((3,1))
                accel_norm = np.linalg.norm(accel_meas)
                # Seulement si |a| ≈ g (forces dynamiques faibles, on se raproche de la gravité seul)
                if abs(accel_norm - GRAVITY) < 0.5:  # Seuil 0.5 m/s²
                    self.update_accel_gravity(accel_meas)

            # UPDATE GPS (Position + Vitesse)
            if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
                position = np.array(gps_data['position']).reshape((3,1))
                velocity = np.array(gps_data['velocity']).reshape((3,1))

                self.update_gps_position_velocity(position, velocity)

            #Magnetometre : correction du cap 
            if imu_data is not None and 'mag' in imu_data:
                mag = np.array(imu_data['mag']).reshape((3,1))
                self.update_heading_mag(mag)

            return 
        
        # === PHASE LARGAGE ===
        if phase == "glide":
            # UPDATE GPS (Position + Vitesse)
            if gps_data is not None and 'position' in gps_data and 'velocity' in gps_data:
                position = np.array(gps_data['position']).reshape((3,1))
                velocity = np.array(gps_data['velocity']).reshape((3,1))

                self.update_gps_position_velocity(position, velocity)


            # UPDATE ACCÉLÉROMÈTRE (Roll/Pitch via gravité)
            if imu_data is not None and 'accel' in imu_data:
                accel_meas = np.array(imu_data['accel']).reshape((3,1))
                accel_norm = np.linalg.norm(accel_meas)
                # Seulement si |a| ≈ g (forces dynamiques faibles, on se raproche de la gravité seul)
                if abs(accel_norm - GRAVITY) < 0.5:  # Seuil 0.5 m/s²
                    self.update_accel_gravity(accel_meas)


            # UPDATE HEADING 
            if gps_data is not None and 'velocity' in gps_data:
                v_gps = np.array(gps_data['velocity']).reshape((3,1))
                v_horizontal = np.sqrt(v_gps[0]**2 + v_gps[1]**2)

                # GPS heading conditions: Sufficient horizontal speed (> 5 m/s)
                if v_horizontal > 5.0:
                    self.update_heading_gps(v_gps)
                else:
                    if imu_data is not None and 'mag' in imu_data:
                        mag = np.array(imu_data['mag']).reshape((3,1))
                        self.update_heading_mag(mag)

            elif imu_data is not None and 'mag' in imu_data:
                mag = np.array(imu_data['mag']).reshape((3,1))
                self.update_heading_mag(mag)
            return

        
    def update_gps_position_velocity(self, position, velocity):
        """
        Update EKF avec mesures GPS position + vitesse.
        
        Args:
            gps_data: dict avec 'position' [px,py,pz] et 'velocity' [vx,vy,vz] en NED
        """
        if not self.isInitialized:
            return
        
        # 1. Extraire mesure GPS (6×1)
        z = np.vstack([
            position,   # [px, py, pz]
            velocity    # [vx, vy, vz]
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
        assert K.shape == (16,6)
        
        # 7. Update état
        self.x = self.x + K @ y
        
        # 8. Update covariance
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P

        # 9. Normaliser quaternion après update
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])
        self._enforce_quaternion_continuity()



    def update_accel_gravity(self, accel_meas):
        """
        Update EKF with accelerometer measurement for roll/pitch correction.

        Measurement model: z = R^T @ [0, 0, -g]^T + b_accel + noise
        where R is body→NED rotation matrix, so R^T is NED→body.
        """
        if not self.isInitialized:
            return

        q = self.x[0:4]
        assert q.shape == (4,1)

        q0, q1, q2, q3 = q.flatten()
        b_accel = self.x[13:16]  # Accelerometer bias
        assert b_accel.shape == (3,1)

        z = accel_meas

        # Measurement prediction: h(x) = R^T @ [0, 0, -g]^T + b_accel
        R_T = Utils.quaternion_to_rotation_matrix(q).T  # NED → body
        h = R_T @ np.array([0, 0, -GRAVITY]).reshape((3,1)) + b_accel       #immobile alors on doit mesurer la reaction à la gravité

        y = z - h

        # Jacobian ∂h/∂q (derived analytically from h = R^T @ [0,0,-g]^T):
        # h1 = -2g*(q1*q3 - q0*q2) = 2g*q0*q2 - 2g*q1*q3
        # h2 = -2g*(q2*q3 + q0*q1) = -2g*q0*q1 - 2g*q2*q3
        # h3 = -g + 2g*(q1² + q2²)
        H_q = np.array([
            [ 2*q2*GRAVITY, -2*q3*GRAVITY,  2*q0*GRAVITY, -2*q1*GRAVITY],
            [-2*q1*GRAVITY, -2*q0*GRAVITY, -2*q3*GRAVITY, -2*q2*GRAVITY],
            [            0,  4*q1*GRAVITY,  4*q2*GRAVITY,             0]
        ])

        # Full Jacobian H (3×16):
        # [H_q(3×4), zeros(3×3), zeros(3×3), zeros(3×3), I(3×3)]
        #  quat      pos         vel         bg          ba
        H = np.zeros((3, 16))
        H[:, 0:4] = H_q              # ∂h/∂q
        H[:, 13:16] = np.eye(3)      # ∂h/∂b_accel = I(3×3)

        S = H @ self.P @ H.T + self.R_accel

        K = self.P @ H.T @ np.linalg.inv(S)
        assert K.shape == (16,3)

        self.x = self.x + K @ y
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])
        self._enforce_quaternion_continuity()

        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P
    
    
    def update_heading_gps(self, v_gps):
        """
        Update EKF avec heading GPS (correction yaw uniquement).
        
        Args:
            gps_data: dict avec 'velocity' [vx, vy, vz] en NED
        """
        if not self.isInitialized:
            return        
        
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
        h_heading = np.array([h_heading]).reshape((1, 1))
        
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
        S = H @ self.P @ H.T + self.R_heading_gps
        
        # 9. Gain de Kalman
        K = self.P @ H.T @ np.linalg.inv(S)
        assert K.shape == (16,1)
        
        # 10. Update état
        self.x = self.x + K @ y
        
        # 11. Update covariance
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P

        # 12. Normaliser quaternion
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])
        self._enforce_quaternion_continuity()

    def update_heading_mag(self, mag_meas):
        if not self.isInitialized:
            return
        
        mag_norm = np.linalg.norm(mag_meas)
        if mag_norm < 1e-6:
            return
        
        mag_n = mag_meas / mag_norm
        mag_ref_n = self.mag_ref / np.linalg.norm(self.mag_ref)
        
        q = self.x[0:4]
        R = Utils.quaternion_to_rotation_matrix(q)
        h = R.T @ mag_ref_n
        h = h / np.linalg.norm(h)
        
        z = mag_n
        y = z - h
        
        # ✅ JACOBIENNE PAR DIFFÉRENCES FINIES (correcte par construction)
        q_flat = q.flatten()
        epsilon = 1e-7
        H_q = np.zeros((3, 4))
        
        for i in range(4):
            q_plus = q_flat.copy()
            q_plus[i] += epsilon
            q_plus = q_plus / np.linalg.norm(q_plus)
            
            R_plus = Utils.quaternion_to_rotation_matrix(q_plus.reshape(4,1))
            h_plus = R_plus.T @ mag_ref_n
            h_plus = h_plus.flatten() / np.linalg.norm(h_plus)
            
            H_q[:, i] = (h_plus - h.flatten()) / epsilon
        
        H = np.zeros((3, 16))
        H[:, 0:4] = H_q
        
        # ✅ GATING STRICT sur innovation
        innovation_norm = np.linalg.norm(y)
        if innovation_norm > 0.3:  # Réduit de 0.5 à 0.3 (≈17° max)
            print(f"⚠️ Mag innovation: {np.rad2deg(innovation_norm):.1f}° > 17° → skip")
            return
        
        S = H @ self.P @ H.T + np.diag([self.R_heading_mag]*3)
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        # ✅ SATURER LE GAIN (éviter corrections brutales)
        K_max = 0.1  # Limiter l'agressivité
        K[0:4, :] = np.clip(K[0:4, :], -K_max, K_max)
        
        self.x = self.x + K @ y
        self.x[0:4] = self.x[0:4] / np.linalg.norm(self.x[0:4])
        self._enforce_quaternion_continuity()
        
        # Joseph form
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ np.diag([self.R_heading_mag]*3) @ K.T


    def _enforce_quaternion_continuity(self):
        """Force quaternion à rester dans le même hémisphère (q ≡ -q)."""
        q_current = self.x[0:4].flatten()
        
        if self._q_previous is None:
            self._q_previous = q_current.copy()
            return
        
        # Si dot < 0, on est dans l'hémisphère opposé
        if np.dot(q_current, self._q_previous) < 0:
            self.x[0:4] = -self.x[0:4]
            q_current = -q_current
        
        self._q_previous = q_current.copy()