"""
Paramètres du simulateur (fenêtre, rendu). Ne contient aucun paramètre de 
comportement/logique de drone 

"""

# ---------- Fenêtre ----------
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
BACKGROUND_COLOR = (18, 20, 28)

# ---------- Zone à balayer ----------
SURFACE_MARGIN = 20               
SURFACE_BORDER_COLOR = (90, 95, 110)

# ---------- Drones  ----------
NUM_DRONES_DEMO = 4               # nombre de drones pour le mode démo
DRONE_RADIUS = 5
DRONE_COLOR = (80, 220, 255)
DRONE_ID_LABEL = False            # afficher l'id à côté de chaque drone

# ---------- Trainée (trace du déplacement) ----------
SHOW_TRAIL = False
TRAIL_LENGTH = 60                 # nombre de positions mémorisées par drone
TRAIL_COLOR = (80, 220, 255)

# ---------- Visualisation de la couverture de zone  ----------

SHOW_COVERAGE = True
SENSOR_RADIUS_VISUAL = 45.0     # Surface couverte par un drone lors de son passage 
GRID_CELL_SIZE = 12             # A modifié selon la logique de drone considéré (altitude,
COVERED_COLOR = (60, 200, 100, 90)  # FOV, correspondance réel d'un pixel en m) ici 45px

# ---------- Divers ----------
RANDOM_SEED = None