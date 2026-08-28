"""""
Le simulateur met a jour la position du drone a partir de la fonction update du fichier
demo_logic. Cette fonction est à modifier en fonction de la logique de flocking considéré 
dans cette étude.
"""""

import config
from simulator import Simulator
from demo_logic import update as demo_update
 
 
def main():
    sim = Simulator(
        num_drones=config.NUM_DRONES_DEMO,
        update_fn=demo_update,   #  à remplacer par la fonction de controle par flocking
    )
    sim.run()
 
 
if __name__ == "__main__":
    main()
 