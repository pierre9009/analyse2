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
        x0 = x[0].item()
        x1 = x[1].item()
        x2 = x[2].item()
        result = np.array([[0, -x0, -x1, -x2],
                           [x0, 0, x2, -x1],
                           [x1, -x2, 0, x0],
                           [x2, x1, -x0, 0]])
        return result
    
    def _get_pitch_from_quaternion(q):
        """
        Extrait le pitch (tangage) depuis un quaternion.
        
        Returns:
            pitch en radians
        """
        q0, q1, q2, q3 = q.flatten()
        
        # Formule standard pour pitch
        sin_pitch = 2 * (q0*q2 - q3*q1)
        
        # Clamp pour éviter erreurs numériques dans arcsin
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        
        pitch = np.arcsin(sin_pitch)
        
        return pitch
    
    def compute_jacobian_F(q, omega, accel_body, biais_acc, biais_gyro, dt):
        """
        Calcule la jacobienne 16x16 du modèle de prédiction.
        """
        F = np.eye(16)

        q0 = q[0].item()
        q1 = q[1].item()
        q2 = q[2].item()
        q3 = q[3].item()

        bgx = biais_gyro[0]
        bgy = biais_gyro[1]
        bgz = biais_gyro[2]

        bax = biais_acc[0]
        bay = biais_acc[1]
        baz = biais_acc[2]

        wcx = (omega[0] - bgx).item()
        wcy = (omega[1] - bgy).item()
        wcz = (omega[2] - bgz).item()

        acx = (accel_body[0] - bax).item()
        acy = (accel_body[1] - bay).item()
        acz = (accel_body[2] - baz).item()



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

        F = np.eye(16) + TEMPO*dt

        return F
