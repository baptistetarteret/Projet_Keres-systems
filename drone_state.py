"""
Représentation minimal d'un drone pour l'affichage.

Pour la logique de flocking, modifier les .x, .y, .heading et 
.data pour stocker les  infos : état interne,
vitesse, batterie, etc.) à chaque appel de leur fonction de mise à jour.
"""


class DroneState:
    def __init__(self, drone_id, x, y, heading=0.0):
        self.id = drone_id
        self.x = float(x)
        self.y = float(y)
        self.heading = heading

        
        self.data = {}  # Permet de stocker n'importe quelle donnée propre à la
                        # logique de drone si necessaire(vitesse, état de 
                        #flocking, batterie...)

        
        self.trail = [] # Historique de positions rempli automatiquement par le
                        # simulateur pour dessiner la traînée

    def position(self):
        return (self.x, self.y)

    def __repr__(self):
        return f"DroneState(id={self.id}, x={self.x:.1f}, y={self.y:.1f})"