import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ekf.gps_api import GPSApi

def main():
    # Initialisation de l'API avec les pins configurés pour pigpio
    # rx_pin: connecté au TX du module GPS
    # tx_pin: connecté au RX du module GPS
    # en_pin: pin d'activation (GPIO 0 / ID_SD sur RPi)
    gps = GPSApi(rx_pin=24, tx_pin=23, en_pin=17)

    try:
        # Initialisation du module (Configuration GGA, RMC et 10Hz)
        if not gps.initialize():
            print("Erreur : Impossible d'initialiser le module GPS.")
            return

        print("--- Démarrage de la lecture GPS (10 Hz) ---")
        print("En attente d'un fix valide des satellites...")

        while True:
            # Mise à jour des données à partir du flux série simulé
            gps.update()

            if gps.is_fixed:
                # Affichage des coordonnées et de la vitesse
                # La vitesse est extraite de la trame RMC 
                # Les coordonnées sont converties en degrés décimaux 
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"Lat: {gps.latitude:.6f} | "
                      f"Lon: {gps.longitude:.6f} | "
                      f"Vitesse: {gps.speed_ms:.2f} m/s")
            else:
                # Si le module est alimenté (EN=High) mais n'a pas de fix 
                print(f"[{time.strftime('%H:%M:%S')}] Recherche de satellites...", end='\r')

            # Pause de 0.1s pour correspondre à la fréquence de rafraîchissement de 10Hz 
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur.")
    finally:
        # Libération des ressources et extinction du module 
        gps.cleanup()

if __name__ == "__main__":
    main()