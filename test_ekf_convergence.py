"""
Test de convergence EKF avec données synthétiques bruitées.
Génère une trajectoire réaliste de planeur et compare estimations vs vérité terrain.
"""

import numpy as np
import matplotlib.pyplot as plt
from ekf import EKF
from utils import Utils

# =============================================================================
# PARAMÈTRES DE SIMULATION
# =============================================================================
GRAVITY = 9.81
DT = 0.01  # 100 Hz IMU
DURATION = 120.0  # 2 minutes de vol
GPS_RATE = 5  # GPS à 5 Hz (toutes les 20 itérations IMU)

# Bruits capteurs (écarts-types)
GYRO_NOISE_STD = 0.01  # rad/s
ACCEL_NOISE_STD = 0.1  # m/s²
GPS_POS_NOISE_STD = np.array([2.5, 2.5, 5.0])  # m (N, E, D)
GPS_VEL_NOISE_STD = np.array([0.3, 0.3, 0.5])  # m/s
MAG_NOISE_STD = 0.05  # µT normalisé

# Biais constants (vérité terrain)
TRUE_GYRO_BIAS = np.array([0.005, -0.003, 0.002])  # rad/s
TRUE_ACCEL_BIAS = np.array([0.05, -0.08, 0.03])  # m/s²


# =============================================================================
# GÉNÉRATION DE TRAJECTOIRE RÉALISTE
# =============================================================================
def generate_trajectory(duration, dt):
    """
    Génère une trajectoire de planeur réaliste.

    Phase 1 (0-30s): Calibration - immobile
    Phase 2 (30-60s): Vol rectiligne avec légère descente
    Phase 3 (60-90s): Virage à gauche (30°)
    Phase 4 (90-120s): Vol rectiligne stabilisé

    Returns:
        times: vecteur temps
        true_states: dict avec position, velocity, quaternion, euler pour chaque instant
    """
    n_samples = int(duration / dt)
    times = np.linspace(0, duration, n_samples)

    # Initialisation
    positions = np.zeros((n_samples, 3))  # NED
    velocities = np.zeros((n_samples, 3))
    eulers = np.zeros((n_samples, 3))  # roll, pitch, yaw
    quaternions = np.zeros((n_samples, 4))
    angular_rates = np.zeros((n_samples, 3))  # omega body frame
    accelerations_body = np.zeros((n_samples, 3))

    # Conditions initiales
    pos = np.array([0.0, 0.0, -500.0])  # 500m altitude (NED: z négatif = altitude)
    vel = np.array([0.0, 0.0, 0.0])
    roll, pitch, yaw = 0.0, 0.0, np.radians(45)  # Cap initial 45° (NE)

    for i, t in enumerate(times):
        # === PHASE 1: Calibration (immobile) ===
        if t < 30.0:
            # Stationnaire
            omega = np.array([0.0, 0.0, 0.0])
            accel_body = np.array([0.0, 0.0, -GRAVITY])  # Gravité en body frame

        # === PHASE 2: Vol rectiligne avec descente ===
        elif t < 60.0:
            if t < 30.5:
                # Transition douce vers le vol
                vel = np.array([15.0, 15.0, 2.0])  # ~21 m/s, léger taux de descente
                pitch = np.radians(-5)  # Légèrement en descente

            omega = np.array([0.0, 0.0, 0.0])

            # Accélération body = R^T @ (-g_ned) quand stationnaire en body
            R = Utils.quaternion_to_rotation_matrix(Utils.quaternion_from_euler(roll, pitch, yaw).reshape(4,1))
            accel_body = R.T @ np.array([0.0, 0.0, GRAVITY])  # gravité vue du body

        # === PHASE 3: Virage à gauche ===
        elif t < 90.0:
            # Virage coordonné
            turn_rate = np.radians(15) / 30.0  # 15° en 30s
            roll_target = np.radians(15)  # Inclinaison 15°

            # Transition roll
            if t < 65.0:
                roll = roll_target * (t - 60.0) / 5.0
            elif t > 85.0:
                roll = roll_target * (90.0 - t) / 5.0
            else:
                roll = roll_target

            yaw += turn_rate * dt

            omega = np.array([0.0, 0.0, turn_rate])  # rotation autour de Z body

            R = Utils.quaternion_to_rotation_matrix(Utils.quaternion_from_euler(roll, pitch, yaw).reshape(4,1))
            # En virage: accélération centripète + gravité
            v_horizontal = np.sqrt(vel[0]**2 + vel[1]**2)
            centripetal = v_horizontal * turn_rate
            accel_body = R.T @ np.array([0.0, 0.0, GRAVITY])
            accel_body[1] += centripetal  # Force latérale en body

        # === PHASE 4: Vol stabilisé ===
        else:
            roll = 0.0
            omega = np.array([0.0, 0.0, 0.0])

            R = Utils.quaternion_to_rotation_matrix(Utils.quaternion_from_euler(roll, pitch, yaw).reshape(4,1))
            accel_body = R.T @ np.array([0.0, 0.0, GRAVITY])

        # Intégration position et vitesse
        if t >= 30.0:
            # Direction de vol selon yaw
            speed = 21.0  # m/s
            vel = np.array([
                speed * np.cos(yaw) * np.cos(pitch),
                speed * np.sin(yaw) * np.cos(pitch),
                -speed * np.sin(pitch) + 2.0  # Légère descente
            ])
            pos = pos + vel * dt

        # Quaternion depuis Euler
        q = Utils.quaternion_from_euler(roll, pitch, yaw)

        # Stockage
        positions[i] = pos
        velocities[i] = vel
        eulers[i] = [roll, pitch, yaw]
        quaternions[i] = q
        angular_rates[i] = omega
        accelerations_body[i] = accel_body

    return {
        'times': times,
        'positions': positions,
        'velocities': velocities,
        'eulers': eulers,
        'quaternions': quaternions,
        'angular_rates': angular_rates,
        'accelerations_body': accelerations_body
    }


# =============================================================================
# GÉNÉRATION DES MESURES BRUITÉES
# =============================================================================
def generate_noisy_measurements(true_states):
    """
    Ajoute du bruit réaliste aux mesures capteurs.
    """
    n = len(true_states['times'])

    # Gyroscope: omega_meas = omega_true + bias + noise
    gyro_meas = (true_states['angular_rates']
                 + TRUE_GYRO_BIAS
                 + np.random.normal(0, GYRO_NOISE_STD, (n, 3)))

    # Accéléromètre: accel_meas = accel_true + bias + noise
    accel_meas = (true_states['accelerations_body']
                  + TRUE_ACCEL_BIAS
                  + np.random.normal(0, ACCEL_NOISE_STD, (n, 3)))

    # Magnétomètre: direction du nord magnétique en body frame
    # Supposons déclinaison = 0, inclinaison = 60° (typique France)
    mag_ned = np.array([0.5, 0.0, 0.866])  # Normalisé, inclinaison 60°
    mag_meas = np.zeros((n, 3))
    for i in range(n):
        q = true_states['quaternions'][i]
        R = Utils.quaternion_to_rotation_matrix(q.reshape(4,1))
        mag_body = R.T @ mag_ned
        mag_meas[i] = mag_body + np.random.normal(0, MAG_NOISE_STD, 3)

    # GPS: position et vitesse (à 5 Hz)
    gps_times = []
    gps_positions = []
    gps_velocities = []

    gps_interval = int(1.0 / (GPS_RATE * DT))
    for i in range(0, n, gps_interval):
        gps_times.append(true_states['times'][i])
        gps_positions.append(
            true_states['positions'][i] + np.random.normal(0, GPS_POS_NOISE_STD)
        )
        gps_velocities.append(
            true_states['velocities'][i] + np.random.normal(0, GPS_VEL_NOISE_STD)
        )

    return {
        'gyro': gyro_meas,
        'accel': accel_meas,
        'mag': mag_meas,
        'gps_times': np.array(gps_times),
        'gps_positions': np.array(gps_positions),
        'gps_velocities': np.array(gps_velocities)
    }


# =============================================================================
# EXÉCUTION DU FILTRE EKF
# =============================================================================
def run_ekf(true_states, measurements):
    """
    Exécute l'EKF sur les mesures bruitées.
    """
    n = len(true_states['times'])

    # Historique des estimations
    estimated_positions = np.zeros((n, 3))
    estimated_velocities = np.zeros((n, 3))
    estimated_eulers = np.zeros((n, 3))
    estimated_gyro_bias = np.zeros((n, 3))
    estimated_accel_bias = np.zeros((n, 3))
    covariances = np.zeros((n, 16))

    # Initialisation EKF
    ekf = EKF(initialization_duration=30.0, sample_rate=int(1/DT))

    gps_idx = 0

    for i in range(n):
        t = true_states['times'][i]

        # Préparer données IMU
        imu_data = {
            'gyro': measurements['gyro'][i],
            'accel': measurements['accel'][i],
            'mag': measurements['mag'][i]
        }

        # Données GPS (si disponible à cet instant)
        gps_data = None
        if gps_idx < len(measurements['gps_times']):
            if abs(t - measurements['gps_times'][gps_idx]) < DT/2:
                gps_data = {
                    'position': measurements['gps_positions'][gps_idx].reshape(3, 1),
                    'velocity': measurements['gps_velocities'][gps_idx].reshape(3, 1)
                }
                gps_idx += 1

        # Phase de calibration
        if not ekf.isInitialized:
            progress = ekf.compute_initial_state(imu_data, gps_data)
            if ekf.isInitialized:
                print(f"EKF initialisé à t={t:.1f}s")
        else:
            # Prédiction
            ekf.predict(imu_data, DT)

            # Déterminer phase de vol
            if t < 35.0:
                phase = "ascension"  # Juste après calibration
            else:
                phase = "glide"

            # Update
            ekf.update(imu_data, gps_data, phase)

        # Stocker estimations
        if ekf.isInitialized:
            estimated_positions[i] = ekf.x[4:7].flatten()
            estimated_velocities[i] = ekf.x[7:10].flatten()

            q = ekf.x[0:4].flatten()
            roll, pitch, yaw = Utils.quaternion_to_euler(q)
            estimated_eulers[i] = [roll, pitch, yaw]

            estimated_gyro_bias[i] = ekf.x[10:13].flatten()
            estimated_accel_bias[i] = ekf.x[13:16].flatten()
            covariances[i] = np.diag(ekf.P)
        else:
            # Avant initialisation, copier les vraies valeurs (pour éviter discontinuités plot)
            estimated_positions[i] = true_states['positions'][i]
            estimated_velocities[i] = true_states['velocities'][i]
            estimated_eulers[i] = true_states['eulers'][i]

    return {
        'positions': estimated_positions,
        'velocities': estimated_velocities,
        'eulers': estimated_eulers,
        'gyro_bias': estimated_gyro_bias,
        'accel_bias': estimated_accel_bias,
        'covariances': covariances
    }


# =============================================================================
# VISUALISATION
# =============================================================================
def plot_results(true_states, measurements, estimates):
    """
    Génère les plots comparatifs.
    """
    times = true_states['times']

    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    fig.suptitle('Test de Convergence EKF - Comparaison Vérité Terrain vs Estimations', fontsize=14)

    # Ligne verticale pour marquer fin de calibration
    calib_end = 30.0

    # === LIGNE 1: POSITION ===
    labels_pos = ['Position N (m)', 'Position E (m)', 'Position D (m)']
    for j in range(3):
        ax = axes[0, j]
        ax.plot(times, true_states['positions'][:, j], 'b-', label='Vérité', linewidth=2)
        ax.plot(times, estimates['positions'][:, j], 'r--', label='Estimé', linewidth=1.5)
        ax.scatter(measurements['gps_times'], measurements['gps_positions'][:, j],
                   c='g', s=10, alpha=0.5, label='GPS bruité')
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel(labels_pos[j])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title('Position (NED)')

    # === LIGNE 2: VITESSE ===
    labels_vel = ['Vitesse N (m/s)', 'Vitesse E (m/s)', 'Vitesse D (m/s)']
    for j in range(3):
        ax = axes[1, j]
        ax.plot(times, true_states['velocities'][:, j], 'b-', label='Vérité', linewidth=2)
        ax.plot(times, estimates['velocities'][:, j], 'r--', label='Estimé', linewidth=1.5)
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel(labels_vel[j])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_title('Vitesse (NED)')

    # === LIGNE 3: ORIENTATION (Euler) ===
    labels_euler = ['Roll (°)', 'Pitch (°)', 'Yaw (°)']
    for j in range(3):
        ax = axes[2, j]
        ax.plot(times, np.rad2deg(true_states['eulers'][:, j]), 'b-', label='Vérité', linewidth=2)
        ax.plot(times, np.rad2deg(estimates['eulers'][:, j]), 'r--', label='Estimé', linewidth=1.5)
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel(labels_euler[j])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[2, 0].set_title('Orientation (Euler)')

    # === LIGNE 4: MESURES CAPTEURS (Gyro et Accel) ===
    labels_gyro = ['Gyro X (rad/s)', 'Gyro Y (rad/s)', 'Gyro Z (rad/s)']
    for j in range(3):
        ax = axes[3, j]
        ax.plot(times, true_states['angular_rates'][:, j], 'b-', label='Vrai', linewidth=2)
        ax.plot(times, measurements['gyro'][:, j], 'orange', alpha=0.5, label='Mesuré (bruité)', linewidth=0.5)
        ax.axhline(y=TRUE_GYRO_BIAS[j], color='green', linestyle='--', label=f'Vrai biais: {TRUE_GYRO_BIAS[j]:.4f}')
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel(labels_gyro[j])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[3, 0].set_title('Gyroscope')

    # === LIGNE 5: BIAIS ESTIMÉS ===
    labels_bias = ['Biais Gyro X', 'Biais Gyro Y', 'Biais Gyro Z']
    for j in range(3):
        ax = axes[4, j]
        ax.axhline(y=TRUE_GYRO_BIAS[j], color='b', linestyle='-', label='Vrai biais gyro', linewidth=2)
        ax.plot(times, estimates['gyro_bias'][:, j], 'r--', label='Estimé gyro', linewidth=1.5)
        ax.axhline(y=TRUE_ACCEL_BIAS[j], color='c', linestyle='-', label='Vrai biais accel', linewidth=2)
        ax.plot(times, estimates['accel_bias'][:, j], 'm--', label='Estimé accel', linewidth=1.5)
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.set_ylabel(labels_bias[j])
        ax.set_xlabel('Temps (s)')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[4, 0].set_title('Estimation des biais')

    plt.tight_layout()
    plt.savefig('ekf_convergence_test.png', dpi=150)
    plt.show()

    # === FIGURE 2: Erreurs ===
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    fig2.suptitle('Erreurs d\'estimation', fontsize=14)

    # Erreur position
    for j in range(3):
        ax = axes2[0, j]
        error = estimates['positions'][:, j] - true_states['positions'][:, j]
        ax.plot(times, error, 'r-', linewidth=1)
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel(f'Erreur {labels_pos[j]}')
        ax.grid(True, alpha=0.3)

        # Stats après calibration
        mask = times > calib_end
        rmse = np.sqrt(np.mean(error[mask]**2))
        ax.set_title(f'RMSE: {rmse:.2f}')

    # Erreur orientation
    for j in range(3):
        ax = axes2[1, j]
        error = np.rad2deg(estimates['eulers'][:, j] - true_states['eulers'][:, j])
        # Wrap yaw error
        if j == 2:
            error = np.where(error > 180, error - 360, error)
            error = np.where(error < -180, error + 360, error)
        ax.plot(times, error, 'r-', linewidth=1)
        ax.axvline(x=calib_end, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel(f'Erreur {labels_euler[j]}')
        ax.set_xlabel('Temps (s)')
        ax.grid(True, alpha=0.3)

        mask = times > calib_end
        rmse = np.sqrt(np.mean(error[mask]**2))
        ax.set_title(f'RMSE: {rmse:.2f}°')

    plt.tight_layout()
    plt.savefig('ekf_errors.png', dpi=150)
    plt.show()

    # === FIGURE 3: Trajectoire 3D ===
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')

    mask = times > calib_end
    ax3.plot(true_states['positions'][mask, 1],
             true_states['positions'][mask, 0],
             -true_states['positions'][mask, 2],
             'b-', label='Vérité', linewidth=2)
    ax3.plot(estimates['positions'][mask, 1],
             estimates['positions'][mask, 0],
             -estimates['positions'][mask, 2],
             'r--', label='Estimé', linewidth=1.5)

    ax3.set_xlabel('Est (m)')
    ax3.set_ylabel('Nord (m)')
    ax3.set_zlabel('Altitude (m)')
    ax3.set_title('Trajectoire 3D')
    ax3.legend()

    plt.savefig('ekf_trajectory_3d.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TEST DE CONVERGENCE EKF")
    print("=" * 60)

    # Fixer seed pour reproductibilité
    np.random.seed(42)

    print("\n1. Génération de la trajectoire de référence...")
    true_states = generate_trajectory(DURATION, DT)
    print(f"   - Durée: {DURATION}s, {len(true_states['times'])} échantillons")
    print(f"   - Position finale vraie: N={true_states['positions'][-1, 0]:.1f}m, "
          f"E={true_states['positions'][-1, 1]:.1f}m, "
          f"Alt={-true_states['positions'][-1, 2]:.1f}m")

    print("\n2. Génération des mesures bruitées...")
    measurements = generate_noisy_measurements(true_states)
    print(f"   - Biais gyro vrais:  {TRUE_GYRO_BIAS}")
    print(f"   - Biais accel vrais: {TRUE_ACCEL_BIAS}")
    print(f"   - {len(measurements['gps_times'])} mesures GPS générées")

    print("\n3. Exécution de l'EKF...")
    estimates = run_ekf(true_states, measurements)

    print("\n4. Calcul des métriques de performance...")
    mask = true_states['times'] > 30.0  # Après calibration

    pos_error = estimates['positions'][mask] - true_states['positions'][mask]
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    print(f"   - RMSE Position: N={pos_rmse[0]:.2f}m, E={pos_rmse[1]:.2f}m, D={pos_rmse[2]:.2f}m")

    euler_error = np.rad2deg(estimates['eulers'][mask] - true_states['eulers'][mask])
    euler_error[:, 2] = np.where(euler_error[:, 2] > 180, euler_error[:, 2] - 360, euler_error[:, 2])
    euler_error[:, 2] = np.where(euler_error[:, 2] < -180, euler_error[:, 2] + 360, euler_error[:, 2])
    euler_rmse = np.sqrt(np.mean(euler_error**2, axis=0))
    print(f"   - RMSE Orientation: Roll={euler_rmse[0]:.2f}°, Pitch={euler_rmse[1]:.2f}°, Yaw={euler_rmse[2]:.2f}°")

    # Erreur biais à la fin
    final_gyro_bias_error = estimates['gyro_bias'][-1] - TRUE_GYRO_BIAS
    final_accel_bias_error = estimates['accel_bias'][-1] - TRUE_ACCEL_BIAS
    print(f"   - Erreur biais gyro final:  {final_gyro_bias_error}")
    print(f"   - Erreur biais accel final: {final_accel_bias_error}")

    print("\n5. Génération des graphiques...")
    plot_results(true_states, measurements, estimates)

    print("\n" + "=" * 60)
    print("TEST TERMINÉ - Fichiers sauvegardés:")
    print("   - ekf_convergence_test.png")
    print("   - ekf_errors.png")
    print("   - ekf_trajectory_3d.png")
    print("=" * 60)
