"""
Vérification de la cohérence des repères Body et NED dans la génération de trajectoire.
Plot les vecteurs de base des repères pour détecter les erreurs de convention.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import Utils

GRAVITY = 9.81

def verify_frame_at_instant(t, roll, pitch, yaw, omega_body, accel_body, velocity_ned):
    """
    Vérifie la cohérence physique à un instant donné.
    
    Args:
        t: temps (s)
        roll, pitch, yaw: angles Euler (rad)
        omega_body: vitesse angulaire en body frame (rad/s)
        accel_body: accélération mesurée par IMU en body frame (m/s²)
        velocity_ned: vitesse en NED (m/s)
    """
    print(f"\n{'='*70}")
    print(f"INSTANT t = {t:.1f}s")
    print(f"{'='*70}")
    
    # Quaternion et matrice de rotation
    q = Utils.quaternion_from_euler(roll, pitch, yaw)
    R_body_to_ned = Utils.quaternion_to_rotation_matrix(q.reshape(4, 1))
    
    print(f"\n1. ANGLES D'EULER:")
    print(f"   Roll:  {np.rad2deg(roll):+7.2f}°")
    print(f"   Pitch: {np.rad2deg(pitch):+7.2f}°")
    print(f"   Yaw:   {np.rad2deg(yaw):+7.2f}°")
    
    print(f"\n2. QUATERNION [w, x, y, z]:")
    print(f"   q = [{q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f}]")
    print(f"   Norme = {np.linalg.norm(q):.6f}")
    
    print(f"\n3. MATRICE DE ROTATION Body → NED:")
    print(f"{R_body_to_ned}")
    
    # Vérification : colonnes de R sont les axes body exprimés en NED
    X_body_in_ned = R_body_to_ned[:, 0]  # Axe X du planeur (nez) en NED
    Y_body_in_ned = R_body_to_ned[:, 1]  # Axe Y du planeur (aile droite) en NED
    Z_body_in_ned = R_body_to_ned[:, 2]  # Axe Z du planeur (bas) en NED
    
    print(f"\n4. AXES BODY EXPRIMÉS EN NED:")
    print(f"   X_body (nez vers l'avant) en NED:  [{X_body_in_ned[0]:+.3f}, {X_body_in_ned[1]:+.3f}, {X_body_in_ned[2]:+.3f}]")
    print(f"   Y_body (aile droite) en NED:       [{Y_body_in_ned[0]:+.3f}, {Y_body_in_ned[1]:+.3f}, {Y_body_in_ned[2]:+.3f}]")
    print(f"   Z_body (vers le bas) en NED:       [{Z_body_in_ned[0]:+.3f}, {Z_body_in_ned[1]:+.3f}, {Z_body_in_ned[2]:+.3f}]")
    
    # Vérifications attendues
    print(f"\n5. VÉRIFICATIONS:")
    
    # 5a. Si roll=pitch=0, X_body doit pointer selon yaw
    if abs(roll) < 0.01 and abs(pitch) < 0.01:
        expected_x = np.array([np.cos(yaw), np.sin(yaw), 0])
        error = np.linalg.norm(X_body_in_ned - expected_x)
        print(f"   ✓ Roll=Pitch=0 → X_body devrait pointer vers yaw={np.rad2deg(yaw):.1f}°")
        print(f"     Attendu: [{expected_x[0]:+.3f}, {expected_x[1]:+.3f}, {expected_x[2]:+.3f}]")
        print(f"     Erreur:  {error:.6f} {'✅' if error < 0.01 else '❌'}")
    
    # 5b. Gravité en NED → gravité en body
    g_ned = np.array([0, 0, GRAVITY])
    g_body_expected = R_body_to_ned.T @ g_ned
    
    print(f"\n6. GRAVITÉ:")
    print(f"   g_NED = [0, 0, +{GRAVITY}] (vers le bas)")
    print(f"   g_body (R^T @ g_NED) = [{g_body_expected[0]:+.3f}, {g_body_expected[1]:+.3f}, {g_body_expected[2]:+.3f}]")
    
    # 5c. Accéléromètre mesure la force spécifique = -g_body si immobile
    print(f"\n7. ACCÉLÉROMÈTRE:")
    print(f"   Mesure IMU (accel_body) = [{accel_body[0]:+.3f}, {accel_body[1]:+.3f}, {accel_body[2]:+.3f}]")
    print(f"   Force spécifique attendue si immobile = -g_body")
    print(f"                                          = [{-g_body_expected[0]:+.3f}, {-g_body_expected[1]:+.3f}, {-g_body_expected[2]:+.3f}]")
    
    accel_error = np.linalg.norm(accel_body - (-g_body_expected))
    print(f"   Erreur: {accel_error:.3f} m/s² {'✅ OK' if accel_error < 0.5 else '❌ PROBLÈME'}")
    
    # 5d. Si le planeur vole, la vitesse NED doit être cohérente avec l'orientation
    v_ned_norm = np.linalg.norm(velocity_ned)
    if v_ned_norm > 1.0:  # Si en mouvement
        v_direction = velocity_ned / v_ned_norm
        # La vitesse devrait être alignée avec X_body (nez du planeur)
        alignment = np.dot(v_direction, X_body_in_ned)
        print(f"\n8. COHÉRENCE VITESSE:")
        print(f"   Vitesse NED = [{velocity_ned[0]:+.2f}, {velocity_ned[1]:+.2f}, {velocity_ned[2]:+.2f}] m/s")
        print(f"   Direction vitesse: [{v_direction[0]:+.3f}, {v_direction[1]:+.3f}, {v_direction[2]:+.3f}]")
        print(f"   Alignement avec X_body (nez): {alignment:.3f}")
        print(f"   {'✅ Cohérent' if alignment > 0.9 else '⚠️  Dérive latérale ou incohérence'}")
    
    # 5e. Gyroscope
    print(f"\n9. GYROSCOPE (body frame):")
    print(f"   ω_body = [{omega_body[0]:+.4f}, {omega_body[1]:+.4f}, {omega_body[2]:+.4f}] rad/s")
    print(f"          = [{np.rad2deg(omega_body[0]):+.2f}, {np.rad2deg(omega_body[1]):+.2f}, {np.rad2deg(omega_body[2]):+.2f}] °/s")


def plot_frames_3d(instants_data):
    """
    Plot 3D des repères Body et NED à différents instants.
    """
    fig = plt.figure(figsize=(16, 12))
    
    n_instants = len(instants_data)
    
    for idx, data in enumerate(instants_data):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        t = data['t']
        roll, pitch, yaw = data['roll'], data['pitch'], data['yaw']
        
        # Quaternion et rotation
        q = Utils.quaternion_from_euler(roll, pitch, yaw)
        R = Utils.quaternion_to_rotation_matrix(q.reshape(4, 1))
        
        # Repère NED (origine)
        origin = np.array([0, 0, 0])
        
        # Axes NED (rouge, vert, bleu)
        scale_ned = 2.0
        ax.quiver(origin[0], origin[1], origin[2], 
                  scale_ned, 0, 0, color='r', arrow_length_ratio=0.15, linewidth=2, label='NED X (Nord)')
        ax.quiver(origin[0], origin[1], origin[2], 
                  0, scale_ned, 0, color='g', arrow_length_ratio=0.15, linewidth=2, label='NED Y (Est)')
        ax.quiver(origin[0], origin[1], origin[2], 
                  0, 0, scale_ned, color='b', arrow_length_ratio=0.15, linewidth=2, label='NED Z (Bas)')
        
        # Axes Body (cyan, magenta, jaune)
        X_body = R[:, 0] * 1.5  # Nez du planeur
        Y_body = R[:, 1] * 1.5  # Aile droite
        Z_body = R[:, 2] * 1.5  # Bas du planeur
        
        ax.quiver(origin[0], origin[1], origin[2], 
                  X_body[0], X_body[1], X_body[2], color='c', arrow_length_ratio=0.2, linewidth=3, label='Body X (nez)')
        ax.quiver(origin[0], origin[1], origin[2], 
                  Y_body[0], Y_body[1], Y_body[2], color='m', arrow_length_ratio=0.2, linewidth=3, label='Body Y (aile)')
        ax.quiver(origin[0], origin[1], origin[2], 
                  Z_body[0], Z_body[1], Z_body[2], color='y', arrow_length_ratio=0.2, linewidth=3, label='Body Z (bas)')
        
        # Gravité
        g_scale = 1.0
        ax.quiver(origin[0], origin[1], origin[2], 
                  0, 0, g_scale, color='orange', arrow_length_ratio=0.2, linewidth=2, linestyle='--', label='Gravité')
        
        # Configuration
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('Nord (m)', fontsize=10)
        ax.set_ylabel('Est (m)', fontsize=10)
        ax.set_zlabel('Bas (m)', fontsize=10)
        ax.set_title(f't={t:.1f}s | R={np.rad2deg(roll):.1f}° P={np.rad2deg(pitch):.1f}° Y={np.rad2deg(yaw):.1f}°', fontsize=11)
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('frame_verification_3d.png', dpi=150)
    print("\n📊 Graphique sauvegardé: frame_verification_3d.png")
    plt.show()


def test_specific_cases():
    """
    Test des cas particuliers pour vérifier les conventions.
    """
    test_cases = [
        {
            't': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0,
            'omega': np.array([0, 0, 0]),
            'velocity_ned': np.array([0, 0, 0]),
            'description': 'Planeur à plat, nez vers le nord, immobile'
        },
        {
            't': 30,
            'roll': 0, 'pitch': 0, 'yaw': np.radians(45),
            'omega': np.array([0, 0, 0]),
            'velocity_ned': np.array([15, 15, 0]),
            'description': 'Planeur horizontal, cap 45° (NE), vitesse 21 m/s'
        },
        {
            't': 60,
            'roll': np.radians(15), 'pitch': np.radians(-5), 'yaw': np.radians(60),
            'omega': np.array([0, 0, 0.01]),
            'velocity_ned': np.array([10, 17, 2]),
            'description': 'Virage à gauche, 15° roll, -5° pitch (descente)'
        },
        {
            't': 90,
            'roll': 0, 'pitch': np.radians(-10), 'yaw': np.radians(90),
            'omega': np.array([0, 0, 0]),
            'velocity_ned': np.array([0, 21, 3.7]),
            'description': 'Vol rectiligne vers l\'Est, descente 10°'
        },
        {
            't': 100,
            'roll': 0, 'pitch': 0, 'yaw': np.radians(180),
            'omega': np.array([0, 0, 0]),
            'velocity_ned': np.array([-21, 0, 0]),
            'description': 'Retour vers le sud'
        },
        {
            't': 110,
            'roll': np.radians(-20), 'pitch': 0, 'yaw': np.radians(270),
            'omega': np.array([0, 0, -0.02]),
            'velocity_ned': np.array([0, -21, 0]),
            'description': 'Virage à droite serré, cap ouest'
        }
    ]
    
    instants_for_plot = []
    
    for case in test_cases:
        print(f"\n\n{'#'*70}")
        print(f"CAS: {case['description']}")
        print(f"{'#'*70}")
        
        # Calculer accel_body selon la physique
        q = Utils.quaternion_from_euler(case['roll'], case['pitch'], case['yaw'])
        R = Utils.quaternion_to_rotation_matrix(q.reshape(4, 1))
        g_ned = np.array([0, 0, GRAVITY])
        g_body = R.T @ g_ned
        accel_body = -g_body  # Force spécifique si immobile en body
        
        # Si en mouvement, ajouter forces dynamiques (simplifié)
        v_norm = np.linalg.norm(case['velocity_ned'])
        if v_norm > 1.0 and abs(case['omega'][2]) > 0.001:
            # Force centripète en virage
            centripetal_accel = v_norm * case['omega'][2]
            accel_body[1] += centripetal_accel  # Force latérale
        
        verify_frame_at_instant(
            case['t'],
            case['roll'], case['pitch'], case['yaw'],
            case['omega'],
            accel_body,
            case['velocity_ned']
        )
        
        instants_for_plot.append({
            't': case['t'],
            'roll': case['roll'],
            'pitch': case['pitch'],
            'yaw': case['yaw']
        })
    
    # Plot 3D
    plot_frames_3d(instants_for_plot)


if __name__ == '__main__':
    print("="*70)
    print("VÉRIFICATION DES REPÈRES BODY ET NED")
    print("="*70)
    
    test_specific_cases()
    
    print("\n" + "="*70)
    print("VÉRIFICATION TERMINÉE")
    print("="*70)
    print("\n💡 Points à vérifier:")
    print("   1. accel_body ≈ [0, 0, -9.81] quand planeur horizontal immobile")
    print("   2. X_body pointe dans la direction du yaw quand roll=pitch=0")
    print("   3. Vitesse NED alignée avec X_body (nez) quand en vol")
    print("   4. Gravité toujours [0, 0, +9.81] en NED")
    print("   5. Quaternion normalisé (norme = 1)")