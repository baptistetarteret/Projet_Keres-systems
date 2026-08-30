"""
Essaim de drones : recherche de cibles en zone inconnue, avec obstacles,
capteur conique, carte de confiance probabiliste et pheromones virtuelles.

Choix de modelisation :
  1. Entite logique  : drone a voilure tournante (position, cap, vitesse), sature
                       en vitesse, en acceleration longitudinale et en taux de virage.
  2. Voisinage       : calcule au temps courant. Disque de rayon R (metrique) ou
                       k plus proches voisins (topologique).
  3. Influence       : uniforme, ou ponderation a decroissance quadratique.
  4. Commande        : raisonnement en vitesse (modele du 1er ordre), pas en force.
  5. Arbitrage       : allocation prioritaire d'un budget de commande (delta-v).
                       Somme ponderee fournie comme point de comparaison.
  6. Obstacles       : disques tires aleatoirement. Bloquent le deplacement ET
                       occultent le capteur. La radio, elle, passe au travers.
  7. Mission         : trouver des cibles ponctuelles (grappes + isolees). La
                       mission s'acheve quand toutes les cibles sont trouvees.
  8. Capteur         : cone oriente selon le cap (portee + demi-angle), et non un
                       disque. Le cap devient donc determinant.
  9. Connaissance    : carte de confiance dans [0,1] par cellule, et non un booleen.
                       c <- 1 - (1-c) exp(-lambda dt), lambda decroissant avec la
                       distance et l'ecart angulaire. Sature a 1, croit avec le
                       temps de survol, se compose correctement entre drones.
 10. Communication   : chaque drone porte ses propres couches, et les drones a moins
                       de comm_radius forment un graphe dont on prend les
                       composantes connexes (transitivite). Fusion par maximum,
                       et somme des contributions au sein d'une composante.
 11. Pheromones      : deux champs virtuels sur la meme grille.
                       -> DOPAMINE deposee sur une cible detectee : attire les drones
                       et accelere la detection alentour.
                       -> CORTISOL depose sur une zone bien observee et sterile : son
                       role propre est d'extrapoler spatialement, donc de
                       decourager des cellules jamais vues mais entourees de vide.
                       Dynamique depot -> diffusion -> evaporation.

Hypotheses assumees : localisation parfaite, capteur idealise par un cone net.
La transitivite supprime le delai de propagation dans une composante : comm_radius
est donc le curseur entre decentralisation pure et oracle global. Une verite
terrain globale, invisible aux drones, sert uniquement a arreter le chronometre.

Commandes clavier :
    HAUT/BAS         selectionner un parametre
    GAUCHE/DROITE    ajuster le parametre selectionne
    N   voisinage    : metrique <-> k plus proches voisins
    I   influence    : aucune -> quadratique -> quadratique a support compact
    A   arbitrage    : allocation prioritaire <-> somme ponderee
    B   bords        : steer / wrap / bounce
    C   couche affichee : confiance -> dopamine -> cortisol -> aucune
    M   vue          : connaissance collective <-> celle du drone 0
    L   liens de communication      V  voisinage du drone 0
    S   cone capteur                P  rayons de perception
    ESPACE pause     R reinitialiser      ECHAP quitter
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pygame
from pygame import Vector2

#  Parametrage :

NEIGHBORHOODS = ("metric", "knn")
INFLUENCES = ("none", "quad", "quad_compact")
ARBITRATIONS = ("priority", "weighted")
BOUNDARIES = ("steer", "wrap", "bounce")
LAYERS = ("confiance", "dopamine", "cortisol", "aucune")

# Ordre de priorite de l'arbitrage (du plus vital au plus accessoire) :
PRIORITY_ORDER = (
    "separation",
    "obstacle",
    "boundary",
    "coverage",
    "alignment",
    "cohesion",
)


@dataclass
class Params:
    """Tous les parametres du modele, regroupes en un objet injectable."""

    # --- 1. entite logique : le drone ------------------------------------- #
    n_drones: int = 24
    v_max: float = 120.0  # px/s
    v_min: float = 40.0  # px/s (0 = vol stationnaire permis)
    a_max: float = 260.0  # px/s2
    omega_max: float = 160.0  # deg/s (160 -> multirotor, 45 -> voilure fixe)
    safety_radius: float = 14.0

    # --- 2. capteur conique ------------------------------------------------ #
    sensor_range: float = 90.0  # px, portee du cone
    sensor_half_angle: float = 35.0  # deg, demi-angle d'ouverture
    sensor_gain: float = 2.4  # 1/s, taux de detection sur l'axe, a d=0

    # --- 3. voisinage (au temps courant) ----------------------------------- #
    neighborhood: str = "metric"
    perception_radius: float = 90.0
    k_neighbors: int = 6

    # --- 4. fonction d'influence -------------------------------------------- #
    influence: str = "quad"
    influence_scale: float = 0.5

    # --- 5. communication ---------------------------------------------------- #
    comm_radius: float = 160.0  # px, portee radio (independante de la perception)

    # --- 6. arbitrage et gains ----------------------------------------------- #
    arbitration: str = "priority"
    authority: float = 220.0
    separation_radius: float = 30.0
    g_separation: float = 1.6
    g_obstacle: float = 2.0
    g_boundary: float = 1.2
    g_coverage: float = 1.0
    g_alignment: float = 0.5
    g_cohesion: float = 0.2

    # --- 7. bords de la zone --------------------------------------------------- #
    boundary_mode: str = "steer"
    boundary_margin: float = 80.0

    # --- 8. obstacles ------------------------------------------------------------ #
    n_obstacles: int = 4
    obstacle_r_min: float = 45.0
    obstacle_r_max: float = 95.0
    obstacle_margin: float = 35.0  # epaisseur de la zone de rappel autour du disque

    # --- 9. cibles ---------------------------------------------------------------- #
    n_clusters: int = 3
    per_cluster: int = 5
    cluster_spread: float = 70.0  # px, ecart-type du tirage dans une grappe
    n_isolated: int = 5

    # --- 10. carte de confiance ----------------------------------------------------- #
    cell_size: int = 16
    detect_threshold: float = 0.70  # confiance au-dela de laquelle une cible est vue
    sterile_threshold: float = 0.88  # ... et au-dela de laquelle une cellule vide
    #                                      est declaree sterile (> detect_threshold,
    #                                      sinon on declarerait sterile une cellule
    #                                      dont la cible n'a pas encore ete detectee)
    explore_threshold: float = 0.60  # en deca : la cellule merite encore un survol

    # --- 11. pheromones ----------------------------------------------------------- #
    dopa_deposit: float = 1.0
    dopa_diffuse: float = 0.16  # coefficient du laplacien (rester < 0.25)
    dopa_evap: float = 0.992  # facteur multiplicatif par pas
    dopa_attract: float = 2.5  # poids dans le score de ciblage
    dopa_gain: float = 1.8  # multiplicateur du taux de detection lambda

    cort_deposit: float = 0.25
    cort_diffuse: float = 0.20
    cort_evap: float = 0.996
    cort_repel: float = 0.8

    # --- 12. ciblage ---------------------------------------------------------------- #
    lookahead_cells: int = 12
    coverage_every: int = 6
    capture_factor: float = 1.2  # cible atteinte si dist < capture * cell_size

    def sanitize(self) -> None:
        """Maintient les invariants apres une edition clavier."""
        self.v_min = min(self.v_min, self.v_max)
        self.n_drones = max(1, int(self.n_drones))
        self.k_neighbors = max(1, min(int(self.k_neighbors), max(1, self.n_drones - 1)))
        self.cell_size = max(4, int(self.cell_size))
        self.sterile_threshold = max(
            self.sterile_threshold, self.detect_threshold + 0.05
        )
        self.dopa_diffuse = min(self.dopa_diffuse, 0.24)
        self.cort_diffuse = min(self.cort_diffuse, 0.24)


# (libelle, attribut, min, max, pas)
TUNABLES = [
    ("Separation", "g_separation", 0.0, 4.0, 0.1),
    ("Alignement", "g_alignment", 0.0, 4.0, 0.1),
    ("Cohesion", "g_cohesion", 0.0, 4.0, 0.05),
    ("Couverture", "g_coverage", 0.0, 4.0, 0.1),
    ("Obstacle", "g_obstacle", 0.0, 6.0, 0.2),
    ("Autorite", "authority", 20.0, 500.0, 10.0),
    ("R percept.", "perception_radius", 20.0, 240.0, 5.0),
    ("k voisins", "k_neighbors", 1, 30, 1),
    ("R comm.", "comm_radius", 20.0, 900.0, 20.0),
    ("R capteur", "sensor_range", 20.0, 220.0, 5.0),
    ("Demi-angle", "sensor_half_angle", 5.0, 180.0, 5.0),
    ("Gain capteur", "sensor_gain", 0.2, 8.0, 0.2),
    ("Attir. dopa", "dopa_attract", 0.0, 10.0, 0.25),
    ("Gain dopa", "dopa_gain", 0.0, 6.0, 0.2),
    ("Rep. cortis.", "cort_repel", 0.0, 5.0, 0.1),
    ("Vitesse max", "v_max", 40.0, 300.0, 10.0),
    ("Virage max", "omega_max", 20.0, 720.0, 10.0),
    ("Effectif", "n_drones", 1, 120, 1),
]

C_BG = (16, 16, 19)
C_CONF = (60, 190, 150)
C_DOPA = (239, 159, 39)
C_CORT = (90, 130, 210)
C_DRONE = (127, 119, 221)
C_HL = (216, 90, 48)
C_OBST = (52, 52, 60)
C_TARGET = (200, 200, 205)
C_FOUND = (120, 220, 150)


#  Fonctions utilitaires :


def limit(v: Vector2, max_len: float) -> Vector2:
    l2 = v.length_squared()
    if l2 > max_len * max_len and l2 > 1e-12:
        return v * (max_len / math.sqrt(l2))
    return v


def set_mag(v: Vector2, mag: float) -> Vector2:
    l2 = v.length_squared()
    if l2 < 1e-12:
        return Vector2(0, 0)
    return v * (mag / math.sqrt(l2))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def union_find_groups(n: int, rows, cols) -> tuple[np.ndarray, list[list[int]]]:
    """Composantes connexes d'un graphe donne par ses aretes. Renvoie
    (composante de chaque sommet, liste des membres par composante)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(rows, cols):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj

    label: dict[int, int] = {}
    comp_of = np.empty(n, dtype=np.int32)
    groups: list[list[int]] = []
    for x in range(n):
        r = find(x)
        if r not in label:
            label[r] = len(groups)
            groups.append([])
        comp_of[x] = label[r]
        groups[label[r]].append(x)
    return comp_of, groups


#  Voisinage d'interaction (metrique ou topologique) et fonction d'influence


class Neighborhood:
    """
    Determine, une fois par frame, avec qui chaque drone interagit.

    Le voisinage est calcule au temps courant : chaque agent decide en regardant
    ou sont ses voisins maintenant, sans extrapolation.

      - "metric" : tous les drones dans un disque de rayon R. Le nombre de voisins
        depend de la densite locale, donc l'essaim se fragmente quand il se dilue.
      - "knn"    : les k plus proches, sans contrainte de distance. Le nombre de
        voisins est constant par construction, ce qui rend la cohesion beaucoup
        plus robuste a la dilatation de l'essaim.
    """

    def __init__(self) -> None:
        self.mean_degree = 0.0
        self.mean_nn_dist = 0.0
        self.violations = 0

    @staticmethod
    def _influence(d: np.ndarray, d0: np.ndarray, mode: str) -> np.ndarray:
        """
        "none"         : w = 1, tous les voisins pesent pareil.
        "quad"         : w = 1/(1+(d/d0)^2), decroissance quadratique, bornee en
                         d=0 et jamais nulle (pas de discontinuite au bord).
        "quad_compact" : w = (1-d/R)^2, meme ordre mais a support compact.
        """
        if mode == "none":
            return np.ones_like(d)
        if mode == "quad_compact":
            return np.clip(1.0 - d / np.maximum(d0, 1e-6), 0.0, 1.0) ** 2
        return 1.0 / (1.0 + (d / np.maximum(d0, 1e-6)) ** 2)

    def build(self, drones: list["Drone"], p: Params, dist: np.ndarray) -> list[list]:
        n = len(drones)
        if n < 2:
            self.mean_degree = 0.0
            self.mean_nn_dist = 0.0
            self.violations = 0
            return [[] for _ in range(n)]

        if p.neighborhood == "knn":
            k = max(1, min(int(p.k_neighbors), n - 1))
            sel = np.argpartition(dist, k - 1, axis=1)[:, :k]
            dsel = np.take_along_axis(dist, sel, axis=1)
            ref = dsel.max(axis=1, keepdims=True)
            d0 = ref * (p.influence_scale if p.influence == "quad" else 1.0)
            wsel = self._influence(dsel, np.broadcast_to(d0, dsel.shape), p.influence)
            rows = np.repeat(np.arange(n), sel.shape[1])
            cols = sel.ravel()
            dd = dsel.ravel()
            ww = wsel.ravel()
        else:
            R = p.perception_radius
            rows, cols = np.nonzero(dist <= R)
            dd = dist[rows, cols]
            d0 = R * (p.influence_scale if p.influence == "quad" else 1.0)
            ww = self._influence(dd, np.full_like(dd, d0), p.influence)

        out: list[list] = [[] for _ in range(n)]
        for i, j, d, w in zip(rows.tolist(), cols.tolist(), dd.tolist(), ww.tolist()):
            if math.isfinite(d):
                out[i].append((drones[j], d, w))

        self.mean_degree = len(dd) / float(n)
        self.mean_nn_dist = float(dist.min(axis=1).mean())
        self.violations = int((dist < p.safety_radius).sum() // 2)
        return out


#  Le monde : obstacles, cibles, verite terrain


class World:
    """
    Geometrie de la mission. Contient ce qui EXISTE reellement : les obstacles,
    les cibles, et la trace de celles qui ont ete trouvees. Les drones n'ont
    jamais acces a `target_mask` autrement qu'a travers leur propre observation.
    """

    def __init__(self, bounds: pygame.Rect, p: Params, rng: random.Random):
        self.cell = p.cell_size
        self.cols = bounds.width // self.cell + 1
        self.rows = bounds.height // self.cell + 1
        self.bounds = bounds
        self.obstacles: list[tuple[float, float, float]] = []
        self._make_obstacles(p, rng)

        # cellules recouvertes par un obstacle : ni observables, ni ciblables
        self.blocked = np.zeros((self.cols, self.rows), dtype=bool)
        xs = (np.arange(self.cols) + 0.5) * self.cell
        ys = (np.arange(self.rows) + 0.5) * self.cell
        gx, gy = xs[:, None], ys[None, :]
        for ox, oy, orad in self.obstacles:
            self.blocked |= ((gx - ox) ** 2 + (gy - oy) ** 2) <= orad * orad

        self.target_mask = np.zeros((self.cols, self.rows), dtype=bool)
        self.found = np.zeros((self.cols, self.rows), dtype=bool)
        self._make_targets(p, rng)
        self.n_targets = int(self.target_mask.sum())

    def _make_obstacles(self, p: Params, rng: random.Random) -> None:
        b = self.bounds
        tries = 0
        while len(self.obstacles) < int(p.n_obstacles) and tries < 400:
            tries += 1
            r = rng.uniform(p.obstacle_r_min, p.obstacle_r_max)
            x = rng.uniform(b.left + r + 40, b.right - r - 40)
            y = rng.uniform(b.top + r + 40, b.bottom - r - 40)
            ok = True
            for ox, oy, orad in self.obstacles:
                if math.hypot(x - ox, y - oy) < r + orad + 70:
                    ok = False
                    break
            if ok:
                self.obstacles.append((x, y, r))

    def _free_cell(self, rng: random.Random) -> tuple[int, int]:
        for _ in range(500):
            i = rng.randrange(self.cols)
            j = rng.randrange(self.rows)
            if not self.blocked[i, j]:
                return i, j
        return 0, 0

    def _make_targets(self, p: Params, rng: random.Random) -> None:
        # Grappes : donnent son sens a la dopamine. Trouver une cible informe
        # alors reellement sur la presence probable d'autres cibles alentour.
        for _ in range(int(p.n_clusters)):
            ci, cj = self._free_cell(rng)
            cx, cy = (ci + 0.5) * self.cell, (cj + 0.5) * self.cell
            placed = 0
            tries = 0
            while placed < int(p.per_cluster) and tries < 200:
                tries += 1
                x = rng.gauss(cx, p.cluster_spread)
                y = rng.gauss(cy, p.cluster_spread)
                i = int(x // self.cell)
                j = int(y // self.cell)
                if 0 <= i < self.cols and 0 <= j < self.rows and not self.blocked[i, j]:
                    if not self.target_mask[i, j]:
                        self.target_mask[i, j] = True
                        placed += 1
        # Isolees : empechent l'essaim de se contenter d'exploiter les grappes.
        for _ in range(int(p.n_isolated)):
            i, j = self._free_cell(rng)
            self.target_mask[i, j] = True

    def blocks_segment(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Un obstacle coupe-t-il le segment ? (utilise pour les traces de debug)"""
        for ox, oy, orad in self.obstacles:
            dx, dy = x1 - x0, y1 - y0
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                continue
            t = max(0.0, min(1.0, ((ox - x0) * dx + (oy - y0) * dy) / l2))
            qx = x0 + t * dx - ox
            qy = y0 + t * dy - oy
            if qx * qx + qy * qy < orad * orad:
                return True
        return False

    def inside_obstacle(self, pos: Vector2) -> tuple[float, float, float] | None:
        for ox, oy, orad in self.obstacles:
            if (pos.x - ox) ** 2 + (pos.y - oy) ** 2 < orad * orad:
                return (ox, oy, orad)
        return None


#  Le drone


class Drone:
    """
    Agent autonome. Etat interne : position, cap, module de vitesse.

    On raisonne en vitesse et non en acceleration : chaque comportement exprime
    une correction de vitesse dv (px/s), l'arbitre en retient une combinaison, et
    la cinematique saturee du drone dit ce qui est reellement realisable dans le
    pas de temps. C'est ce qu'expose l'autopilote d'un multirotor : une consigne
    de vitesse, pas une consigne de poussee.
    """

    __slots__ = (
        "pos",
        "heading",
        "speed",
        "vel",
        "params",
        "target",
        "next_eval",
        "index",
        "alloc",
        "color",
    )

    def __init__(
        self, pos: Vector2, heading: float, speed: float, params: Params, index: int = 0
    ):
        self.pos = Vector2(pos)
        self.heading = heading
        self.speed = speed
        self.vel = Vector2(math.cos(heading), math.sin(heading)) * speed
        self.params = params
        self.target: Vector2 | None = None
        self.next_eval = index % max(1, int(params.coverage_every))
        self.index = index
        self.alloc: dict[str, float] = {}
        self.color = C_DRONE

    # -- comportements : chacun renvoie une correction de vitesse -------------- #

    def _w_separation(self, nb) -> Vector2:
        p = self.params
        push = Vector2(0, 0)
        urgency = 0.0
        for other, d, w in nb:
            if d >= p.separation_radius or d < 1e-6:
                continue
            push += (self.pos - other.pos) * (w / (d * d))
            urgency = max(urgency, 1.0 - d / p.separation_radius)
        if urgency <= 0.0:
            return Vector2(0, 0)
        # l'urgence reste purement geometrique : l'influence pondere la direction
        # de fuite, jamais le declenchement de la securite.
        return set_mag(push, p.v_max * urgency) * p.g_separation

    def _w_obstacle(self, world: World) -> Vector2:
        p = self.params
        push = Vector2(0, 0)
        urgency = 0.0
        for ox, oy, orad in world.obstacles:
            off = self.pos - Vector2(ox, oy)
            d = off.length()
            outer = orad + p.obstacle_margin
            if d >= outer:
                continue
            u = 1.0 if d <= orad else 1.0 - (d - orad) / max(1.0, p.obstacle_margin)
            if d < 1e-6:
                off = Vector2(1, 0)
                d = 1.0
            push += off / d * u
            urgency = max(urgency, u)
        if urgency <= 0.0:
            return Vector2(0, 0)
        return set_mag(push, p.v_max * urgency) * p.g_obstacle

    def _w_alignment(self, nb) -> Vector2:
        p = self.params
        acc = Vector2(0, 0)
        tot = 0.0
        for other, _d, w in nb:
            acc += other.vel * w
            tot += w
        if tot <= 1e-9:
            return Vector2(0, 0)
        return (acc / tot - self.vel) * p.g_alignment

    def _w_cohesion(self, nb) -> Vector2:
        p = self.params
        acc = Vector2(0, 0)
        tot = 0.0
        for other, _d, w in nb:
            acc += other.pos * w
            tot += w
        if tot <= 1e-9:
            return Vector2(0, 0)
        offset = acc / tot - self.pos
        dist = offset.length()
        if dist < 1e-6:
            return Vector2(0, 0)
        slow = min(1.0, dist / max(1.0, p.separation_radius * 2.0))
        return (set_mag(offset, p.v_max * slow) - self.vel) * p.g_cohesion

    def _w_boundary(self, bounds: pygame.Rect) -> Vector2:
        p = self.params
        if p.boundary_mode != "steer":
            return Vector2(0, 0)
        m = p.boundary_margin
        push = Vector2(0, 0)
        if self.pos.x < bounds.left + m:
            push.x += 1.0 - (self.pos.x - bounds.left) / m
        elif self.pos.x > bounds.right - m:
            push.x -= 1.0 - (bounds.right - self.pos.x) / m
        if self.pos.y < bounds.top + m:
            push.y += 1.0 - (self.pos.y - bounds.top) / m
        elif self.pos.y > bounds.bottom - m:
            push.y -= 1.0 - (bounds.bottom - self.pos.y) / m
        depth = push.length()
        if depth < 1e-6:
            return Vector2(0, 0)
        return (set_mag(push, p.v_max) - self.vel) * (p.g_boundary * min(1.0, depth))

    def _w_coverage(self) -> Vector2:
        """La cible est choisie par le simulateur a partir de la carte du drone."""
        p = self.params
        if p.g_coverage <= 0.0 or self.target is None:
            return Vector2(0, 0)
        offset = self.target - self.pos
        if offset.length() < p.capture_factor * p.cell_size:
            self.target = None
            return Vector2(0, 0)
        return (set_mag(offset, p.v_max) - self.vel) * p.g_coverage

    # -- arbitrage -------------------------------------------------------------- #

    def decide(self, nb, world: World, bounds: pygame.Rect) -> Vector2:
        p = self.params
        want = {
            "separation": self._w_separation(nb),
            "obstacle": self._w_obstacle(world),
            "boundary": self._w_boundary(bounds),
            "coverage": self._w_coverage(),
            "alignment": self._w_alignment(nb),
            "cohesion": self._w_cohesion(nb),
        }

        if p.arbitration == "weighted":
            # Reference : somme ponderee. Tout s'exprime, la saturation finale
            # ecrete tout de la meme facon, donc une exigence de securite peut
            # etre diluee par plusieurs exigences de confort.
            total = Vector2(0, 0)
            for name in PRIORITY_ORDER:
                total += want[name]
            self.alloc = {k: want[k].length() for k in PRIORITY_ORDER}
            return limit(self.vel + limit(total, p.authority), p.v_max)

        # Allocation prioritaire : l'autorite de commande est une ressource finie.
        # Chaque comportement consomme dans la limite du reliquat et ne laisse aux
        # suivants que ce qu'il n'a pas utilise.
        total = Vector2(0, 0)
        remaining = p.authority
        alloc: dict[str, float] = {}
        for name in PRIORITY_ORDER:
            dv = want[name]
            m = dv.length()
            if remaining <= 1e-6 or m <= 1e-9:
                alloc[name] = 0.0
                continue
            if m > remaining:
                dv = dv * (remaining / m)
                m = remaining
            total += dv
            remaining -= m
            alloc[name] = m
        self.alloc = alloc
        return limit(self.vel + total, p.v_max)

    # -- cinematique saturee ------------------------------------------------------ #

    def apply(
        self, v_des: Vector2, dt: float, bounds: pygame.Rect, world: World
    ) -> None:
        p = self.params
        if v_des.length_squared() > 1e-9:
            d_th = wrap_pi(math.atan2(v_des.y, v_des.x) - self.heading)
            lim = math.radians(p.omega_max) * dt
            self.heading = wrap_pi(self.heading + max(-lim, min(lim, d_th)))

        sp_des = min(p.v_max, max(p.v_min, v_des.length()))
        lim_a = p.a_max * dt
        d_sp = max(-lim_a, min(lim_a, sp_des - self.speed))
        self.speed = min(p.v_max, max(p.v_min, self.speed + d_sp))

        self.vel = Vector2(math.cos(self.heading), math.sin(self.heading)) * self.speed
        self.pos += self.vel * dt
        self._boundary(bounds)
        self._push_out(world)

    def _push_out(self, world: World) -> None:
        """Contrainte dure : on ne penetre jamais un obstacle."""
        hit = world.inside_obstacle(self.pos)
        if hit is None:
            return
        ox, oy, orad = hit
        off = self.pos - Vector2(ox, oy)
        if off.length_squared() < 1e-9:
            off = Vector2(1, 0)
        self.pos = Vector2(ox, oy) + set_mag(off, orad + 1.0)

    def _boundary(self, b: pygame.Rect) -> None:
        mode = self.params.boundary_mode
        if mode == "wrap":
            if self.pos.x < b.left:
                self.pos.x += b.width
            elif self.pos.x > b.right:
                self.pos.x -= b.width
            if self.pos.y < b.top:
                self.pos.y += b.height
            elif self.pos.y > b.bottom:
                self.pos.y -= b.height
        elif mode == "bounce":
            if self.pos.x < b.left or self.pos.x > b.right:
                self.pos.x = max(b.left, min(b.right, self.pos.x))
                self.heading = wrap_pi(math.pi - self.heading)
            if self.pos.y < b.top or self.pos.y > b.bottom:
                self.pos.y = max(b.top, min(b.bottom, self.pos.y))
                self.heading = -self.heading
            self.vel = (
                Vector2(math.cos(self.heading), math.sin(self.heading)) * self.speed
            )
        else:
            self.pos.x = max(b.left, min(b.right, self.pos.x))
            self.pos.y = max(b.top, min(b.bottom, self.pos.y))

    # -- rendu ---------------------------------------------------------------------- #

    def draw(self, screen: pygame.Surface, show_cone: bool) -> None:
        p = self.params
        if show_cone:
            half = math.radians(p.sensor_half_angle)
            pts = [(self.pos.x, self.pos.y)]
            steps = 10
            for s in range(steps + 1):
                a = self.heading - half + 2 * half * s / steps
                pts.append(
                    (
                        self.pos.x + p.sensor_range * math.cos(a),
                        self.pos.y + p.sensor_range * math.sin(a),
                    )
                )
            pygame.draw.polygon(screen, (40, 110, 95), pts, 1)
        ca, sa = math.cos(self.heading), math.sin(self.heading)
        tri = [
            (self.pos.x + lx * ca - ly * sa, self.pos.y + lx * sa + ly * ca)
            for lx, ly in ((9, 0), (-5, 4), (-2.5, 0), (-5, -4))
        ]
        pygame.draw.polygon(screen, self.color, tri)


#  Le simulateur


class Swarm:
    """
    Porte les cartes de connaissance. Chaque drone a SES couches ; elles sont
    stockees en un tableau (n_drones, cols, rows) par couche pour pouvoir etre
    traitees d'un bloc.

    Ordre d'un pas :
      1. voisinage d'interaction
      2. graphe de communication -> composantes connexes (transitivite)
      3. fusion par composante (maximum, idempotent)
      4. choix de cible, arbitrage, cinematique
      5. observation dans le cone, sommee par composante
      6. depots de pheromones
      7. diffusion + evaporation
      8. verite terrain et test de fin

    La fusion precede l'observation : un drone qui rejoint un groupe doit
    profiter de sa connaissance avant de choisir ou regarder.
    """

    def __init__(self, params: Params, bounds: pygame.Rect, seed: int | None = None):
        self.params = params
        self.bounds = bounds
        self.seed = seed
        self.rng = random.Random(seed)
        self.neigh = Neighborhood()
        self.drones: list[Drone] = []
        self.world = World(bounds, params, self.rng)
        self.frame = 0
        self.time = 0.0
        self.done = False
        self.completion_time: float | None = None
        self.alloc_mean = {k: 0.0 for k in PRIORITY_ORDER}
        self.last_nbs: list[list] = []
        self.comm_pairs: list[tuple[int, int]] = []
        self.n_components = 1
        self.conf = np.zeros((0, 0, 0), dtype=np.float32)
        self.dopa = np.zeros((0, 0, 0), dtype=np.float32)
        self.cort = np.zeros((0, 0, 0), dtype=np.float32)
        self.known_t = np.zeros((0, 0, 0), dtype=bool)
        self.known_s = np.zeros((0, 0, 0), dtype=bool)
        self.reset()

    # -- initialisation --------------------------------------------------------- #

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self.drones.clear()
        self.time = 0.0
        self.frame = 0
        self.done = False
        self.completion_time = None
        self.world = World(self.bounds, self.params, self.rng)
        self._alloc_layers(int(self.params.n_drones))
        for i in range(int(self.params.n_drones)):
            self._spawn(i)

    def _alloc_layers(self, n: int) -> None:
        w = self.world
        shape = (n, w.cols, w.rows)
        self.conf = np.zeros(shape, dtype=np.float32)
        self.dopa = np.zeros(shape, dtype=np.float32)
        self.cort = np.zeros(shape, dtype=np.float32)
        self.known_t = np.zeros(shape, dtype=bool)
        self.known_s = np.zeros(shape, dtype=bool)

    def _spawn(self, index: int) -> None:
        b = self.bounds
        for _ in range(200):
            pos = Vector2(
                self.rng.uniform(b.left + 20, b.right - 20),
                self.rng.uniform(b.top + 20, b.bottom - 20),
            )
            if self.world.inside_obstacle(pos) is None:
                break
        self.drones.append(
            Drone(
                pos,
                self.rng.uniform(-math.pi, math.pi),
                self.params.v_max * 0.5,
                self.params,
                index,
            )
        )

    def _sync_population(self) -> None:
        n = int(self.params.n_drones)
        cur = len(self.drones)
        if n == cur:
            return
        w = self.world
        if n > cur:
            pad = ((0, n - cur), (0, 0), (0, 0))
            self.conf = np.pad(self.conf, pad)
            self.dopa = np.pad(self.dopa, pad)
            self.cort = np.pad(self.cort, pad)
            self.known_t = np.pad(self.known_t, pad)
            self.known_s = np.pad(self.known_s, pad)
            while len(self.drones) < n:
                self._spawn(len(self.drones))
        else:
            self.conf = self.conf[:n]
            self.dopa = self.dopa[:n]
            self.cort = self.cort[:n]
            self.known_t = self.known_t[:n]
            self.known_s = self.known_s[:n]
            del self.drones[n:]
        _ = w

    # -- communication ------------------------------------------------------------ #

    def _communication(self, dist: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
        n = len(self.drones)
        if n < 2:
            self.comm_pairs = []
            self.n_components = n
            return np.zeros(n, dtype=np.int32), [[i] for i in range(n)]
        rows, cols = np.nonzero(dist <= self.params.comm_radius)
        self.comm_pairs = [
            (int(i), int(j)) for i, j in zip(rows.tolist(), cols.tolist()) if i < j
        ]
        comp_of, groups = union_find_groups(n, rows, cols)
        self.n_components = len(groups)
        return comp_of, groups

    def _fuse(self, groups: list[list[int]]) -> None:
        """
        Fusion par maximum element par element. Le maximum est idempotent : deux
        drones qui restent cote a cote fusionnent a chaque frame sans que
        l'information soit recomptee. Il est aussi commutatif et associatif, donc
        l'ordre de traitement des membres n'a aucune importance.
        """
        for members in groups:
            if len(members) < 2:
                continue
            m = np.asarray(members)
            self.conf[m] = self.conf[m].max(axis=0)
            self.dopa[m] = self.dopa[m].max(axis=0)
            self.cort[m] = self.cort[m].max(axis=0)
            self.known_t[m] = self.known_t[m].any(axis=0)
            self.known_s[m] = self.known_s[m].any(axis=0)

    # -- capteur conique ------------------------------------------------------------ #

    def _cone_lambda(self, d: Drone, dopa: np.ndarray):
        """
        Taux de detection instantane lambda sur la fenetre locale du drone.
        Renvoie (i0, i1, j0, j1, lam) ou None si la fenetre est vide.

        lambda decroit avec la distance et avec l'ecart angulaire a l'axe, s'annule
        au bord du cone, et est multiplie par (1 + gain * dopamine) : c'est par la
        que la dopamine accelere la detection dans une zone jugee prometteuse.
        """
        p = self.params
        w = self.world
        cell = w.cell
        R = p.sensor_range
        r = int(R // cell) + 1
        cx, cy = int(d.pos.x // cell), int(d.pos.y // cell)
        i0, i1 = max(0, cx - r), min(w.cols, cx + r + 1)
        j0, j1 = max(0, cy - r), min(w.rows, cy + r + 1)
        if i0 >= i1 or j0 >= j1:
            return None

        dx = (np.arange(i0, i1) + 0.5) * cell - d.pos.x
        dy = (np.arange(j0, j1) + 0.5) * cell - d.pos.y
        DX, DY = dx[:, None], dy[None, :]
        dist = np.sqrt(DX * DX + DY * DY)
        safe = np.maximum(dist, 1e-6)

        hx, hy = math.cos(d.heading), math.sin(d.heading)
        cosang = (DX * hx + DY * hy) / safe
        cos_half = math.cos(math.radians(p.sensor_half_angle))

        # la cellule sous le drone est toujours observee (visee nadir)
        mask = (dist <= R) & ((cosang >= cos_half) | (dist <= cell))
        mask &= ~w.blocked[i0:i1, j0:j1]

        # occultation : un obstacle sur le segment drone -> cellule coupe la vue
        for ox, oy, orad in w.obstacles:
            wx, wy = ox - d.pos.x, oy - d.pos.y
            l2 = DX * DX + DY * DY + 1e-9
            t = np.clip((wx * DX + wy * DY) / l2, 0.0, 1.0)
            qx = t * DX - wx
            qy = t * DY - wy
            mask &= (qx * qx + qy * qy) >= orad * orad

        denom = max(1e-6, 1.0 - cos_half)
        angular = np.clip((cosang - cos_half) / denom, 0.0, 1.0)
        angular = np.where(dist <= cell, 1.0, angular)
        radial = np.clip(1.0 - dist / R, 0.0, 1.0)
        lam = p.sensor_gain * radial * angular
        lam = lam * (1.0 + p.dopa_gain * dopa[i0:i1, j0:j1])
        return i0, i1, j0, j1, np.where(mask, lam, 0.0).astype(np.float32)

    # -- ciblage ---------------------------------------------------------------------- #

    def _choose_target(self, d: Drone, conf, dopa, cort) -> Vector2 | None:
        """
        Cellule maximisant  besoin x (1 + attirance dopamine - repulsion cortisol)
        x biais de cap / sqrt(distance), dans la fenetre locale du drone.

        Le besoin vaut 1 pour une cellule jamais observee et tombe a 0 quand la
        confiance atteint le seuil d'exploration. Le sqrt evite que le drone ne
        choisisse toujours la cellule juste derriere son capteur ; le biais de cap
        penalise les demi-tours, ce qui compte des lors que le virage est borne.
        """
        p = self.params
        w = self.world
        cell = w.cell
        reach = int(p.lookahead_cells)
        cx, cy = int(d.pos.x // cell), int(d.pos.y // cell)
        i0, i1 = max(0, cx - reach), min(w.cols, cx + reach + 1)
        j0, j1 = max(0, cy - reach), min(w.rows, cy + reach + 1)

        if i0 < i1 and j0 < j1:
            cw = conf[i0:i1, j0:j1]
            need = np.clip(
                (p.explore_threshold - cw) / max(1e-6, p.explore_threshold), 0.0, 1.0
            )
            attract = (
                1.0
                + p.dopa_attract * dopa[i0:i1, j0:j1]
                - p.cort_repel * cort[i0:i1, j0:j1]
            )
            attract = np.clip(attract, 0.0, None)

            dx = (np.arange(i0, i1) + 0.5) * cell - d.pos.x
            dy = (np.arange(j0, j1) + 0.5) * cell - d.pos.y
            DX, DY = dx[:, None], dy[None, :]
            dist = np.sqrt(DX * DX + DY * DY) + 1.0
            hx, hy = math.cos(d.heading), math.sin(d.heading)
            bias = 0.55 + 0.45 * (DX * hx + DY * hy) / dist

            score = need * attract * bias / np.sqrt(dist)
            score = np.where(w.blocked[i0:i1, j0:j1], 0.0, score)
            k = int(np.argmax(score))
            if score.flat[k] > 1e-9:
                i, j = np.unravel_index(k, score.shape)
                return Vector2((i0 + i + 0.5) * cell, (j0 + j + 0.5) * cell)

        # Repli : cellule mal connue la plus proche, dans SA propre carte. Aucun
        # oracle ici : un drone peut viser une cellule qu'un autre a deja couverte
        # si l'information ne lui est pas parvenue.
        idx = np.argwhere((conf < p.explore_threshold) & ~w.blocked)
        if idx.size == 0:
            return None
        cxy = (idx + 0.5) * cell
        d2 = (cxy[:, 0] - d.pos.x) ** 2 + (cxy[:, 1] - d.pos.y) ** 2
        k = int(np.argmin(d2))
        return Vector2(float(cxy[k, 0]), float(cxy[k, 1]))

    # -- pheromones -------------------------------------------------------------------- #

    @staticmethod
    def _diffuse_evaporate(A: np.ndarray, rate: float, evap: float) -> None:
        """Laplacien a 5 points sur toutes les couches d'un coup, puis evaporation."""
        if A.size == 0:
            return
        P = np.pad(A, ((0, 0), (1, 1), (1, 1)), mode="edge")
        lap = (
            P[:, :-2, 1:-1]
            + P[:, 2:, 1:-1]
            + P[:, 1:-1, :-2]
            + P[:, 1:-1, 2:]
            - 4.0 * A
        )
        A += rate * lap
        A *= evap

    # -- un pas ------------------------------------------------------------------------ #

    def step(self, dt: float) -> None:
        p = self.params
        p.sanitize()
        self.frame += 1
        self.time += dt
        self._sync_population()

        n = len(self.drones)
        if n == 0:
            return
        w = self.world

        pos = np.array([(d.pos.x, d.pos.y) for d in self.drones], dtype=np.float64)
        if n > 1:
            diff = pos[None, :, :] - pos[:, None, :]
            dist = np.sqrt((diff**2).sum(-1))
            np.fill_diagonal(dist, np.inf)
        else:
            dist = np.full((1, 1), np.inf)

        # 1. voisinage d'interaction
        nbs = self.neigh.build(self.drones, p, dist)
        self.last_nbs = nbs

        # 2-3. communication puis fusion (avant l'observation)
        comp_of, groups = self._communication(dist)
        self._fuse(groups)

        # 4. ciblage, arbitrage, cinematique
        for i, d in enumerate(self.drones):
            if self.frame >= d.next_eval or d.target is None:
                d.target = self._choose_target(
                    d, self.conf[i], self.dopa[i], self.cort[i]
                )
                d.next_eval = self.frame + max(1, int(p.coverage_every))
        desired = [d.decide(nbs[i], w, self.bounds) for i, d in enumerate(self.drones)]
        for d, v in zip(self.drones, desired):
            d.apply(v, dt, self.bounds, w)

        # 5. observation : les contributions de tous les membres d'une composante
        #    sont sommees avant d'etre appliquees a leur carte commune. La
        #    summabilite est donc exacte pour des drones qui volent ensemble.
        lam = np.zeros((len(groups), w.cols, w.rows), dtype=np.float32)
        for i, d in enumerate(self.drones):
            res = self._cone_lambda(d, self.dopa[i])
            if res is None:
                continue
            i0, i1, j0, j1, lw = res
            lam[comp_of[i], i0:i1, j0:j1] += lw

        for ci, members in enumerate(groups):
            rep = members[0]
            c = self.conf[rep]
            c += (1.0 - c) * (1.0 - np.exp(-lam[ci] * dt))
            np.clip(c, 0.0, 1.0, out=c)

            # 6. depots. On ne depose qu'a la transition, sinon une grappe deja
            #    exploitee resterait eternellement attractive.
            detected = (c >= p.detect_threshold) & w.target_mask & ~self.known_t[rep]
            if detected.any():
                self.dopa[rep][detected] += p.dopa_deposit
                self.known_t[rep] |= detected
            sterile = (
                (c >= p.sterile_threshold)
                & ~self.known_t[rep]
                & ~self.known_s[rep]
                & ~w.blocked
            )
            if sterile.any():
                self.cort[rep][sterile] += p.cort_deposit
                self.known_s[rep] |= sterile

            if len(members) > 1:
                m = np.asarray(members)
                self.conf[m] = c
                self.dopa[m] = self.dopa[rep]
                self.cort[m] = self.cort[rep]
                self.known_t[m] = self.known_t[rep]
                self.known_s[m] = self.known_s[rep]

        # 7. diffusion puis evaporation
        self._diffuse_evaporate(self.dopa, p.dopa_diffuse, p.dopa_evap)
        self._diffuse_evaporate(self.cort, p.cort_diffuse, p.cort_evap)

        # 8. verite terrain (invisible aux drones) et fin de mission
        best = self.conf.max(axis=0)
        w.found |= w.target_mask & (best >= p.detect_threshold)
        if not self.done and w.n_targets > 0 and int(w.found.sum()) >= w.n_targets:
            self.done = True
            self.completion_time = self.time

        if self.frame % 10 == 0:
            self.alloc_mean = {
                k: sum(d.alloc.get(k, 0.0) for d in self.drones) / n
                for k in PRIORITY_ORDER
            }

    # -- indicateurs -------------------------------------------------------------------- #

    def polarization(self) -> float:
        if not self.drones:
            return 0.0
        sx = sum(math.cos(d.heading) for d in self.drones)
        sy = sum(math.sin(d.heading) for d in self.drones)
        return math.hypot(sx, sy) / len(self.drones)

    def metrics(self) -> dict:
        w = self.world
        if self.conf.size:
            collective = float(self.conf.max(axis=0).mean())
            solo = float(self.conf[0].mean())
        else:
            collective = solo = 0.0
        return {
            "found": int(w.found.sum()),
            "targets": w.n_targets,
            "conf_col": collective,
            "conf_solo": solo,
            "polar": self.polarization(),
            "degree": self.neigh.mean_degree,
            "nn_dist": self.neigh.mean_nn_dist,
            "violations": self.neigh.violations,
            "comps": self.n_components,
        }

    def layer(self, name: str, collective: bool) -> np.ndarray | None:
        if not self.conf.size:
            return None
        if name == "confiance":
            src = self.conf
        elif name == "dopamine":
            src = self.dopa
        elif name == "cortisol":
            src = self.cort
        else:
            return None
        return src.max(axis=0) if collective else src[0]


#  Interface


def render_layer(
    screen: pygame.Surface,
    field: np.ndarray | None,
    color: tuple[int, int, int],
    cell: int,
    scale: float = 1.0,
) -> None:
    if field is None:
        return
    a = np.clip(field * scale, 0.0, 1.0)[..., None] * 0.55
    bg = np.array(C_BG, dtype=np.float64)
    fg = np.array(color, dtype=np.float64)
    rgb = bg * (1.0 - a) + fg * a
    surf = pygame.surfarray.make_surface(rgb.astype(np.uint8))
    cols, rows = field.shape
    screen.blit(pygame.transform.scale(surf, (cols * cell, rows * cell)), (0, 0))


def draw_hud(
    screen,
    font,
    params: Params,
    swarm: Swarm,
    selected: int,
    fps: float,
    layer: str,
    collective: bool,
) -> None:
    m = swarm.metrics()
    if swarm.done and swarm.completion_time is not None:
        t_txt = f"t {swarm.completion_time:6.1f}s FINI"
    else:
        t_txt = f"t {swarm.time:6.1f}s"
    header = [
        (f"FPS {fps:5.1f}  drones {len(swarm.drones):3d}  {t_txt}", (200, 200, 205)),
        (f"cibles {m['found']:2d} / {m['targets']:2d}", (120, 220, 190)),
        (
            f"voisinage {params.neighborhood}"
            + (
                f" k={params.k_neighbors}"
                if params.neighborhood == "knn"
                else f" R={params.perception_radius:.0f}"
            ),
            (150, 200, 240),
        ),
        (
            f"influence {params.influence}  arb. {params.arbitration[:4]}",
            (150, 200, 240),
        ),
        (f"radio R={params.comm_radius:.0f}  groupes {m['comps']:2d}", (150, 200, 240)),
        (
            f"couche : {layer} ({'collectif' if collective else 'drone 0'})",
            (240, 180, 120),
        ),
        ("", None),
        (
            f"confiance moy. {m['conf_col'] * 100:5.1f} %  (d0 {m['conf_solo'] * 100:4.1f} %)",
            (120, 220, 190),
        ),
        (
            f"polarisation {m['polar']:5.2f}  voisins {m['degree']:4.1f}",
            (200, 200, 205),
        ),
        (
            f"d_min {m['nn_dist']:5.1f} px  conflits {m['violations']:3d}",
            (220, 140, 140),
        ),
        ("", None),
    ]
    lh = 17
    h = 20 + lh * (len(header) + len(TUNABLES) + len(PRIORITY_ORDER) + 2)
    panel = pygame.Surface((272, h), pygame.SRCALPHA)
    panel.fill((18, 18, 20, 215))
    screen.blit(panel, (10, 10))

    y = 18
    for txt, col in header:
        if txt:
            screen.blit(font.render("  " + txt, True, col), (16, y))
        y += lh

    for i, (label, attr, _lo, _hi, _st) in enumerate(TUNABLES):
        val = getattr(params, attr)
        txt = (
            f"{label:<12}{val:7.2f}"
            if isinstance(val, float)
            else f"{label:<12}{val:7d}"
        )
        sel = i == selected
        screen.blit(
            font.render(
                ("> " if sel else "  ") + txt,
                True,
                (235, 235, 235) if sel else (150, 150, 158),
            ),
            (16, y),
        )
        y += lh

    y += 6
    screen.blit(font.render("  budget alloue (px/s)", True, (170, 170, 178)), (16, y))
    y += lh
    total = max(1e-6, params.authority)
    for name in PRIORITY_ORDER:
        v = swarm.alloc_mean.get(name, 0.0)
        bw = int(80 * min(1.0, v / total))
        screen.blit(font.render(f"  {name[:10]:<11}", True, (150, 150, 158)), (16, y))
        pygame.draw.rect(screen, (55, 55, 62), pygame.Rect(126, y + 4, 80, 8), 1)
        if bw:
            pygame.draw.rect(screen, (120, 200, 170), pygame.Rect(126, y + 4, bw, 8))
        screen.blit(font.render(f"{v:5.0f}", True, (150, 150, 158)), (212, y))
        y += lh


def draw_done_banner(screen, font_big, swarm: Swarm, W: int) -> None:
    if not swarm.done or swarm.completion_time is None:
        return
    msg = (
        f"MISSION TERMINEE — {swarm.world.n_targets} cibles en "
        f"{swarm.completion_time:.1f} s   (R : relancer)"
    )
    surf = font_big.render(msg, True, C_FOUND)
    rect = surf.get_rect(center=(W // 2, 34))
    bgr = pygame.Surface((rect.width + 24, rect.height + 14), pygame.SRCALPHA)
    bgr.fill((18, 18, 20, 230))
    screen.blit(bgr, (rect.left - 12, rect.top - 7))
    screen.blit(surf, rect)


def cycle(seq, cur):
    return seq[(seq.index(cur) + 1) % len(seq)]


def main(headless: bool = False, max_frames: int = 0, seed: int | None = None) -> None:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    W, H = 1180, 740
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Essaim de drones — cibles, obstacles, pheromones")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 12)
    font_big = pygame.font.SysFont("monospace", 22, bold=True)

    params = Params()
    bounds = pygame.Rect(0, 0, W, H)
    swarm = Swarm(params, bounds, seed)

    selected = 0
    paused = False
    layer = "confiance"
    collective = True
    show_links = False
    show_vision = False
    show_cone = False
    show_radii = False
    frames = 0
    running = True

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    swarm.reset()
                elif ev.key == pygame.K_c:
                    layer = cycle(LAYERS, layer)
                elif ev.key == pygame.K_m:
                    collective = not collective
                elif ev.key == pygame.K_l:
                    show_links = not show_links
                elif ev.key == pygame.K_v:
                    show_vision = not show_vision
                    for d in swarm.drones:
                        d.color = C_DRONE
                elif ev.key == pygame.K_s:
                    show_cone = not show_cone
                elif ev.key == pygame.K_p:
                    show_radii = not show_radii
                elif ev.key == pygame.K_n:
                    params.neighborhood = cycle(NEIGHBORHOODS, params.neighborhood)
                elif ev.key == pygame.K_i:
                    params.influence = cycle(INFLUENCES, params.influence)
                elif ev.key == pygame.K_a:
                    params.arbitration = cycle(ARBITRATIONS, params.arbitration)
                elif ev.key == pygame.K_b:
                    params.boundary_mode = cycle(BOUNDARIES, params.boundary_mode)
                elif ev.key == pygame.K_UP:
                    selected = (selected - 1) % len(TUNABLES)
                elif ev.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(TUNABLES)
                elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    _lab, attr, lo, hi, stp = TUNABLES[selected]
                    sign = 1 if ev.key == pygame.K_RIGHT else -1
                    cur = getattr(params, attr)
                    new = min(hi, max(lo, cur + sign * stp))
                    setattr(
                        params,
                        attr,
                        int(new) if isinstance(cur, int) else round(new, 2),
                    )

        if not paused and not swarm.done:
            swarm.step(dt)

        screen.fill(C_BG)
        w = swarm.world

        if layer == "confiance":
            render_layer(screen, swarm.layer(layer, collective), C_CONF, w.cell)
        elif layer == "dopamine":
            render_layer(screen, swarm.layer(layer, collective), C_DOPA, w.cell, 0.7)
        elif layer == "cortisol":
            render_layer(screen, swarm.layer(layer, collective), C_CORT, w.cell, 1.5)

        for ox, oy, orad in w.obstacles:
            pygame.draw.circle(screen, C_OBST, (int(ox), int(oy)), int(orad))
            pygame.draw.circle(screen, (80, 80, 92), (int(ox), int(oy)), int(orad), 1)

        for i, j in np.argwhere(w.target_mask):
            x = int((i + 0.5) * w.cell)
            y = int((j + 0.5) * w.cell)
            if w.found[i, j]:
                pygame.draw.circle(screen, C_FOUND, (x, y), 5)
            else:
                pygame.draw.circle(screen, C_TARGET, (x, y), 4, 1)

        if params.boundary_mode == "steer":
            mg = int(params.boundary_margin)
            pygame.draw.rect(
                screen, (55, 55, 62), pygame.Rect(mg, mg, W - 2 * mg, H - 2 * mg), 1
            )

        if show_links:
            for i, j in swarm.comm_pairs:
                pygame.draw.line(
                    screen, (50, 70, 90), swarm.drones[i].pos, swarm.drones[j].pos, 1
                )

        if show_radii:
            for i, d in enumerate(swarm.drones):
                if params.neighborhood == "metric":
                    r = int(params.perception_radius)
                else:
                    nb = swarm.last_nbs[i] if i < len(swarm.last_nbs) else []
                    r = int(max((dd for _o, dd, _w in nb), default=0))
                if r > 0:
                    pygame.draw.circle(
                        screen, (45, 70, 90), (int(d.pos.x), int(d.pos.y)), r, 1
                    )

        if show_vision and swarm.drones:
            d0 = swarm.drones[0]
            for d in swarm.drones:
                d.color = C_DRONE
            d0.color = C_HL
            if params.neighborhood == "metric":
                pygame.draw.circle(
                    screen,
                    C_HL,
                    (int(d0.pos.x), int(d0.pos.y)),
                    int(params.perception_radius),
                    1,
                )
            for other, _d, _wt in (swarm.last_nbs[0] if swarm.last_nbs else []):
                other.color = (200, 160, 90)
                pygame.draw.line(screen, (90, 70, 50), d0.pos, other.pos, 1)
            if d0.target is not None:
                pygame.draw.line(screen, C_CONF, d0.pos, d0.target, 1)

        for d in swarm.drones:
            d.draw(screen, show_cone)

        draw_hud(
            screen, font, params, swarm, selected, clock.get_fps(), layer, collective
        )
        draw_done_banner(screen, font_big, swarm, W)
        pygame.display.flip()

        frames += 1
        if max_frames and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=0, help="arret apres N frames")
    ap.add_argument("--headless", action="store_true", help="sans fenetre (test)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    main(headless=args.headless, max_frames=args.frames, seed=args.seed)
