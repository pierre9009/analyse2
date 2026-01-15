import numpy as np
np.set_printoptions(suppress=True, precision=8)
# coef venant de https://github.com/mikoff/imu-calib
SA = np.array([ 0.0018, -0.0004, 0.0050])
MA = np.array([-0.0030, 0.0007, -0.0037])

SG=np.array([-0.0012,  0.0022, 0.0021])
MG=np.array([-0.0021, 0.0011, 0.0018])
EG =np.array([0.0024,  0.0018,  -0.0031])

# 1) Inversion exacte d'une 3x3 triangulaire supérieure
# M = [[a,b,c],
#      [0,d,e],
#      [0,0,f]]



def build_M(S, M_cross):
    Sx, Sy, Sz = S
    NOx, NOy, NOz = M_cross
    M = np.array([
        [1.0 + Sx, NOz,      NOy],
        [0.0,      1.0 + Sy, NOx],
        [0.0,      0.0,      1.0 + Sz],
    ], dtype=float)
    return M


def compute_Minv_from_S_M(S, M_cross):
    M = build_M(S, M_cross)
    Minv = np.linalg.inv(M)
    return Minv

def R_e_inv(E_gyro):
    E0, E1, E2 = E_gyro
    R = np.array([
        [1.0, -E2,      E1],
        [E2,      1.0, -E0],
        [-E1,      E0,      1.0],
    ], dtype=float)
    Rinv = np.linalg.inv(R)
    return Rinv
print("="*40)
print("Minv pour l'accelero")
print(compute_Minv_from_S_M(SA,MA))

print("="*40)
print("matrice pour le gyro")
print(compute_Minv_from_S_M(SG,MG)@R_e_inv(EG))
print("="*40)