"""
Fichier test avec une ogique de base pour le déplacement des drones, à remplacer par la 
logique de flocking
"""

import random
import config


def update(drones, surface_rect, dt):
    for drone in drones:
        # initialise une vitesse aléatoire une seule fois, stockée dans .data
        if "vx" not in drone.data:
            drone.data["vx"] = random.uniform(-1, 1) * 80
            drone.data["vy"] = random.uniform(-1, 1) * 80

        drone.x += drone.data["vx"] * dt
        drone.y += drone.data["vy"] * dt

        # rebond sur les bords de la surface à balayer
        if drone.x < surface_rect.left or drone.x > surface_rect.right:
            drone.data["vx"] *= -1
            drone.x = max(surface_rect.left, min(surface_rect.right, drone.x))
        if drone.y < surface_rect.top or drone.y > surface_rect.bottom:
            drone.data["vy"] *= -1
            drone.y = max(surface_rect.top, min(surface_rect.bottom, drone.y))

        # orientation du triangle du drone dans le sens du déplacement
        import math
        drone.heading = math.atan2(drone.data["vy"], drone.data["vx"])