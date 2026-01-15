#!/usr/bin/env python3
"""Enregistre les données IMU dans un fichier .log"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ekf.imu_api import ImuReader

with open("imu_data.log", "w") as f, ImuReader(port="/dev/ttyS0", baudrate=115200) as imu:
    print("Enregistrement dans imu_data.log (Ctrl+C pour arrêter)...")
    while True:
        m = imu.read()
        if m:
            f.write(f"{m['ax']} {m['ay']} {m['az']} {m['gx']} {m['gy']} {m['gz']} {m['mx']} {m['my']} {m['mz']} {m['seq']} {m['tempC']}\n")
            f.flush()
            print(f"{m['ax']} {m['ay']} {m['az']} {m['gx']} {m['gy']} {m['gz']}\n")