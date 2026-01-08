#!/usr/bin/env python3

import time
from imu_api import ImuReader

def main():
    print("Lecture IMU...")
    
    with ImuReader(port="/dev/ttyS0", baudrate=115200) as imu:
        while True:
            data = imu.read(timeout=1.0)
            
            if data:
                print(f"Seq: {data['seq']} | "
                      f"Acc: [{data['ax']:.2f}, {data['ay']:.2f}, {data['az']:.2f}] | "
                      f"Gyr: [{data['gx']:.2f}, {data['gy']:.2f}, {data['gz']:.2f}] | "
                      f"Mag: [{data['mx']:.2f}, {data['my']:.2f}, {data['mz']:.2f}]")
            else:
                print("Timeout - pas de données")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt")