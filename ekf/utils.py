import numpy as np

GRAVITY = 9.81

class Utils:
    
    def quaternion_from_euler(roll, pitch, yaw):
        """
        Convertit angles d'Euler (rad) en quaternion [w, x, y, z].
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
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
    
    
    def quaternion_to_rotation_matrix(q):
        """
        Matrice de rotation body → NED depuis quaternion.
        Convention : q = [q0, q1, q2, q3] = [w, x, y, z]
        """
        #q = q / np.linalg.norm(q)
        q0, q1, q2, q3 = q.flatten()
        
        # Formule standard (body → NED)
        R = np.array([
            [1 - 2*(q2**2 + q3**2),  2*(q1*q2 - q0*q3),      2*(q1*q3 + q0*q2)     ],
            [2*(q1*q2 + q0*q3),      1 - 2*(q1**2 + q3**2),  2*(q2*q3 - q0*q1)     ],
            [2*(q1*q3 - q0*q2),      2*(q2*q3 + q0*q1),      1 - 2*(q1**2 + q2**2) ]
        ])
        return R

    def skew_4x4(x):
        x0 = x[0,0]
        x1 = x[1,0]
        x2 = x[2,0]
        result = np.array([[0, -x0, -x1, -x2],
                           [x0, 0, x2, -x1],
                           [x1, -x2, 0, x0],
                           [x2, x1, -x0, 0]])
        return result

    
    def compute_jacobian_F(q, omega, accel_body, dt):
        """
        Calcule la jacobienne 16x16 du modèle de prédiction.
        """
        F = np.eye(16)

        q0 = q[0,0]
        q1 = q[1,0]
        q2 = q[2,0]
        q3 = q[3,0]

        wcx = omega[0,0]
        wcy = omega[1,0]
        wcz = omega[2,0]

        acx = accel_body[0,0]
        acy = accel_body[1,0]
        acz = accel_body[2,0]



        A1 = np.array([[0, -0.5*wcx, -0.5*wcy, -0.5*wcz],
                      [0.5*wcx, 0, 0.5*wcz, -0.5*wcy],
                      [0.5*wcy, -0.5*wcz, 0, 0.5*wcx],
                      [0.5*wcz, 0.5*wcy, -0.5*wcx, 0]])
        
        A2 = np.array([[0.5*q1, 0.5*q2, 0.5*q3],
                      [-0.5*q0, 0.5*q3, -0.5*q2],
                      [-0.5*q3, -0.5*q0, 0.5*q1],
                      [0.5*q2, -0.5*q1, -0.5*q0]])
        
        A3 = np.array([[-2*q3*acy+2*q2*acz, 2*q2*acy+2*q3*acz, -4*q2*acx+2*q1*acy+2*q0*acz, -4*q3*acx-2*q0*acy+2*q1*acz],
                      [2*q3*acx-2*q1*acz, 2*q2*acx-4*q1*acy-2*q0*acz, 2*q1*acx+2*q3*acz, 2*q0*acx-4*q3*acy+2*q2*acz],
                      [-2*q2*acx+2*q1*acy, 2*q3*acx+2*q0*acy-4*q1*acz, -2*q0*acx+2*q3*acy-4*q2*acz, 2*q1*acx+2*q2*acy]])
        
        R = Utils.quaternion_to_rotation_matrix(q)
        A4 = -R
        
        
        
        F1 = np.hstack((A1, np.zeros((4,6)), A2, np.zeros((4,3))))
        F2 = np.hstack((np.zeros((3,4)), np.zeros((3,3)), np.eye(3), np.zeros((3,6))))
        F3 = np.hstack((A3, np.zeros((3,6)), np.zeros((3,3)), A4))
        F4 = np.zeros((6,16))

        TEMPO = np.vstack((F1, F2, F3, F4))
        assert TEMPO.shape == (16,16)

        F = np.eye(16) + TEMPO*dt
        assert F.shape == (16,16)

        return F
    

    def compute_jacobian_F_extended(q, omega, accel_body, dt):
        """
        Jacobienne 19×19 avec estimation B_NED.
        Structure: [q(4), p(3), v(3), b_gyro(3), b_accel(3), B_NED(3)]
        """
        F = np.eye(19)
        
        # Récupérer ancienne Jacobienne 16×16
        F_16 = Utils.compute_jacobian_F(q, omega, accel_body, dt)
        
        # Copier dans coin supérieur gauche
        F[0:16, 0:16] = F_16
        
        # B_NED est constant : F[16:19, 16:19] = I (déjà fait par np.eye)
        
        return F
