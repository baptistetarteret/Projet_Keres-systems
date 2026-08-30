"""
Essaim de drones : flocking et decouverte d'une zone fixe en 2D.

Choix de modelisation :
  1. Entite logique  : un drone a voilure tournante, etat (position, cap, vitesse),
                       avec contrainte de saturation (en vitesse, en acceleration
                       longitudinale et en taux de virage).
  2. Voisinage       : calcule au temps courant (non predictif). Disque de rayon R
                       (metrique) ou k plus proches voisins (topologique).
  3. Fonc. d'Influence : uniforme, ou ponderation a decroissance quadratique.
  4. Commande        : raisonnement en vitesse (modele du 1er ordre), pas en force.
  5. Arbitrage       : par allocation prioritaire d'un budget de commande (delta-v),
                       les comportements servis dans l'ordre, le reliquat seul est
                       redistribue.
                       Rq : un arbitrage par somme ponderee est fourni comme point
                       de comparaison.
  6. Mission         : balayage d'une zone fixe, une seule fois. Chaque cellule vue
                       reste acquise et la mission s'arrete quand 100 % de la zone a
                       ete decouvert (timer qui mesure le temps de couverture).

Commandes clavier :
    HAUT/BAS         selectionner un parametre
    GAUCHE/DROITE    ajuster le parametre selectionne
    N   voisinage    : metrique <-> k plus proches voisins
    I   influence    : aucune -> quadratique -> quadratique a support compact
    A   arbitrage    : allocation prioritaire <-> somme ponderee
    B   bords        : steer / wrap / bounce
    C   carte de couverture      V  voisinage du drone 0      S  empreintes capteur
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

# Ordre de priorite de l'arbitrage (du plus vital au plus accessoire) :

PRIORITY_ORDER = ("separation", "boundary", "coverage", "alignment", "cohesion")


@dataclass
class Params:
    """Tous les parametres du modele, regroupes en un objet injectable."""

    # --- 1. entite logique : le drone ------------------------------------- #
    n_drones: int = 30
    v_max: float = 90.0  # px/s   vitesse air maximale
    v_min: float = (
        45.0  # px/s   vitesse de croisiere minimale (0 = vol stationnaire permis)
    )
    a_max: float = 260.0  # px/s2  acceleration longitudinale
    omega_max: float = (
        200.0  # deg/s  taux de virage (200 -> multirotor, 45 -> voilure fixe)
    )
    sensor_radius: float = 34.0  # px     empreinte capteur au sol
    safety_radius: float = (
        14.0  # px     distance en deca de laquelle on compte une quasi-collision
    )

    # --- 2. voisinage (au temps courant) ---------------------------------- #
    neighborhood: str = "metric"
    perception_radius: float = 78.0  # utilise seulement en mode "metric"
    k_neighbors: int = 7  # utilise seulement en mode "knn"

    # --- 3. fonction d'influence ------------------------------------------- #
    influence: str = "quad"
    influence_scale: float = 0.5  # d0 = influence_scale * rayon de reference

    # --- 4. arbitrage et gains --------------------------------------------- #
    arbitration: str = "priority"
    authority: float = 230.0  # px/s : budget total de correction de vitesse
    separation_radius: float = 30.0
    g_separation: float = 1.6
    g_boundary: float = 1.2
    g_coverage: float = 0.9
    g_alignment: float = 0.7
    g_cohesion: float = 0.35

    # --- 5. bords de la zone ------------------------------------------------ #
    boundary_mode: str = "steer"
    boundary_margin: float = 90.0

    # --- 6. mission de couverture ------------------------------------------- #
    cell_size: int = 18  # px, resolution de la carte
    lookahead_cells: int = 10  # rayon de recherche de cible, en cellules
    coverage_every: int = 6  # recalcul de la cible 1 frame sur N (echelonne)
    capture_factor: float = (
        0.6  # cible atteinte si dist < capture_factor * sensor_radius
    )

    def sanitize(self) -> None:
        """Maintient les invariants apres une edition clavier."""
        self.v_min = min(self.v_min, self.v_max)
        self.k_neighbors = max(
            1, min(int(self.k_neighbors), max(1, int(self.n_drones) - 1))
        )
        self.n_drones = int(self.n_drones)
        self.cell_size = int(self.cell_size)


# (libelle, attribut, min, max, pas)
TUNABLES = [
    ("Separation", "g_separation", 0.0, 4.0, 0.1),
    ("Alignement", "g_alignment", 0.0, 4.0, 0.1),
    ("Cohesion", "g_cohesion", 0.0, 4.0, 0.05),
    ("Couverture", "g_coverage", 0.0, 4.0, 0.1),
    ("Bords", "g_boundary", 0.0, 4.0, 0.1),
    ("Autorite", "authority", 20.0, 500.0, 10.0),
    ("R percept.", "perception_radius", 20.0, 240.0, 5.0),
    ("k voisins", "k_neighbors", 1, 30, 1),
    ("R separat.", "separation_radius", 5.0, 100.0, 2.0),
    ("R capteur", "sensor_radius", 10.0, 120.0, 2.0),
    ("Vitesse max", "v_max", 40.0, 400.0, 10.0),
    ("Virage max", "omega_max", 20.0, 720.0, 10.0),
    ("Effectif", "n_drones", 5, 400, 5),
]

C_BG = (16, 16, 19)
C_COV = (60, 190, 150)
C_DRONE = (127, 119, 221)
C_HL = (216, 90, 48)


#  Fonctions Utilitaires :


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


#  2. Voisinage considere (metrique ou topologique)
#  3. Fonction d'influence


class Neighborhood:
    """
    Determine, une fois par frame, avec qui chaque drone interagit.

    Le voisinage est calcule au temps courant : chaque agent decide en regardant
    ou sont ses voisins maintenant (positions courantes p_j), sans extrapolation.

    Deux definitions du voisinage, strictement interchangeables :
      - "metric" : tous les drones dans un disque de rayon R (modele de portee
        de capteur ou de liaison radio -> le nombre de voisins depend de la
        densite locale, donc l'essaim se fragmente quand il se dilue).
      - "knn"    : les k plus proches, sans aucune contrainte de distance
        (modele topologique, cf. les mesures sur les vols d'etourneaux -> le
        nombre de voisins est constant par construction, ce qui rend la
        cohesion beaucoup plus robuste a la dilatation de l'essaim).

    Le calcul est vectorise : matrice de distances complete en O(n^2), ce qui
    reste plus rapide qu'un hachage spatial en Python pur en dessous d'un
    millier d'agents, et surtout exact. Indispensable pour que la comparaison
    entre les deux voisinages ne soit pas polluee par une approximation.
    """

    def __init__(self) -> None:
        self.mean_degree = 0.0
        self.mean_nn_dist = 0.0
        self.violations = 0
        self.n_components = 1
        self.largest_frac = 1.0

    @staticmethod
    def _components(n: int, rows, cols) -> tuple[int, float]:
        """
        Composantes connexes du graphe d'interaction (arcs consideres non orientes).

        C'est l'indicateur qui separe le mieux les deux voisinages : un voisinage
        metrique perd ses arcs des que l'essaim se dilue et se fragmente en
        sous-groupes qui ne se retrouvent plus, alors qu'un voisinage a k plus
        proches voisins conserve k arcs par agent quelle que soit la densite.
        """
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in zip(rows, cols):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        sizes: dict[int, int] = {}
        for x in range(n):
            r = find(x)
            sizes[r] = sizes.get(r, 0) + 1
        return len(sizes), max(sizes.values()) / float(n)

    @staticmethod
    def _influence(d: np.ndarray, d0: np.ndarray, mode: str) -> np.ndarray:
        """
        Poids accorde a un voisin situe a la distance d.

        "none"          : w = 1, tous les voisins pesent pareil.
        "quad"          : w = 1 / (1 + (d/d0)^2). Decroissance quadratique
                          (equivalent a d^-2 en champ lointain), bornee en d=0,
                          jamais nulle : pas de discontinuite quand un voisin
                          entre ou sort du voisinage.
        "quad_compact"  : w = (1 - d/R)^2. Meme ordre de decroissance mais a
                          support compact : le poids s'annule exactement au bord
                          du voisinage. Plus propre topologiquement, au prix
                          d'une derivee non nulle a l'annulation.
        """
        if mode == "none":
            return np.ones_like(d)
        if mode == "quad_compact":
            return np.clip(1.0 - d / np.maximum(d0, 1e-6), 0.0, 1.0) ** 2
        return 1.0 / (1.0 + (d / np.maximum(d0, 1e-6)) ** 2)

    def build(
        self, drones: list["Drone"], p: Params, with_metrics: bool = False
    ) -> list[list[tuple]]:
        n = len(drones)
        if n < 2:
            self.mean_degree = 0.0
            self.mean_nn_dist = 0.0
            self.violations = 0
            self.n_components, self.largest_frac = (n, 1.0 if n else 0.0)
            return [[] for _ in range(n)]
        pos = np.array([(d.pos.x, d.pos.y) for d in drones], dtype=np.float64)

        # Distances entre positions courantes (aucune extrapolation).
        diff = pos[None, :, :] - pos[:, None, :]
        dist = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(dist, np.inf)

        if p.neighborhood == "knn":
            k = max(1, min(int(p.k_neighbors), n - 1))
            sel = np.argpartition(dist, k - 1, axis=1)[:, :k]
            dsel = np.take_along_axis(dist, sel, axis=1)
            # rayon de reference propre a chaque drone : distance du k-ieme voisin
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

        out: list[list[tuple]] = [[] for _ in range(n)]
        ri, ci = rows.tolist(), cols.tolist()
        for i, j, d, w in zip(ri, ci, dd.tolist(), ww.tolist()):
            if math.isfinite(d):
                out[i].append((drones[j], d, w))
        self.mean_degree = len(dd) / float(n)

        if with_metrics:
            # Les distances de securite se lisent directement sur dist, puisque
            # le voisinage est deja calcule au temps courant.
            self.mean_nn_dist = float(dist.min(axis=1).mean()) if n > 1 else 0.0
            self.violations = int((dist < p.safety_radius).sum() // 2)
            self.n_components, self.largest_frac = self._components(n, ri, ci)
        return out


#  6. Carte de couverture : la mission (zone fixe, une seule passe)


class CoverageMap:
    """
    La surface est discretisee en cellules ou chaque cellule memorise seulement le
    fait d'avoir ete survolee au moins une fois. Une cellule vue reste acquise :
    la mission est un balayage unique de la zone, et non une surveillance
    persistante. Elle se termine lorsque toutes les cellules ont ete decouvertes.

    Indicateur principal :
      - explored : fraction de cellules vues au moins une fois (croissante,
        atteint 1.0 en fin de mission).
    """

    def __init__(self, width: int, height: int, cell_size: int):
        self.cell = max(2, int(cell_size))
        self.cols = width // self.cell + 1
        self.rows = height // self.cell + 1
        self.seen = np.zeros((self.cols, self.rows), dtype=bool)
        self._st_radius = -1.0
        self._stencil = None
        self._cache: pygame.Surface | None = None
        self._cache_frame = -999

    # ---- marquage ---------------------------------------------------------- #
    def _get_stencil(self, radius: float) -> np.ndarray:
        if radius != self._st_radius:
            r = int(radius // self.cell) + 1
            g = np.arange(-r, r + 1) * self.cell
            d2 = g[:, None] ** 2 + g[None, :] ** 2
            self._stencil = d2 <= radius * radius
            self._st_radius = radius
        return self._stencil

    def mark_all(self, drones: list["Drone"], radius: float) -> None:
        st = self._get_stencil(radius)
        r = st.shape[0] // 2
        for d in drones:
            cx, cy = int(d.pos.x // self.cell), int(d.pos.y // self.cell)
            i0, j0 = cx - r, cy - r
            si, sj = max(0, -i0), max(0, -j0)
            i0c, j0c = max(0, i0), max(0, j0)
            i1c = min(self.cols, i0 + st.shape[0])
            j1c = min(self.rows, j0 + st.shape[1])
            if i0c >= i1c or j0c >= j1c:
                continue
            sub = st[si : si + (i1c - i0c), sj : sj + (j1c - j0c)]
            self.seen[i0c:i1c, j0c:j1c][sub] = True

    # ---- choix de cible ---------------------------------------------------- #
    def _nearest_unseen(self, pos: Vector2) -> Vector2 | None:
        """Cellule non vue la plus proche dans toute la zone (repli global)."""
        idx = np.argwhere(~self.seen)
        if idx.size == 0:
            return None
        c = self.cell
        cxy = (idx + 0.5) * c
        d2 = (cxy[:, 0] - pos.x) ** 2 + (cxy[:, 1] - pos.y) ** 2
        k = int(np.argmin(d2))
        return Vector2(float(cxy[k, 0]), float(cxy[k, 1]))

    def best_target(self, pos: Vector2, heading: Vector2, reach: int) -> Vector2 | None:
        """
        Cellule NON VUE maximisant  (biais de cap) / sqrt(distance) dans le
        voisinage local. Si tout est deja vu autour du drone, on bascule sur la
        cellule non vue la plus proche a l'echelle de toute la zone, pour que les
        drones migrent vers les dernieres poches non couvertes et que la mission
        puisse atteindre 100 %.

        Le sqrt (et non la distance brute) evite que le drone ne choisisse
        systematiquement la cellule immediatement derriere son capteur. Le biais
        de cap penalise les demi-tours et lisse la trajectoire, ce qui compte des
        lors que le taux de virage est borne.
        """
        c = self.cell
        cx, cy = int(pos.x // c), int(pos.y // c)
        i0, i1 = max(0, cx - reach), min(self.cols, cx + reach + 1)
        j0, j1 = max(0, cy - reach), min(self.rows, cy + reach + 1)
        if i0 >= i1 or j0 >= j1:
            return self._nearest_unseen(pos)

        sub_seen = self.seen[i0:i1, j0:j1]
        if sub_seen.all():
            return self._nearest_unseen(pos)

        dx = (np.arange(i0, i1) + 0.5) * c - pos.x
        dy = (np.arange(j0, j1) + 0.5) * c - pos.y
        dxg, dyg = dx[:, None], dy[None, :]
        dist = np.sqrt(dxg * dxg + dyg * dyg) + 1.0
        bias = 0.55 + 0.45 * (dxg * heading.x + dyg * heading.y) / dist
        score = bias / np.sqrt(dist)
        score[sub_seen] = -np.inf  # on ne cible que les cellules non vues

        k = int(np.argmax(score))
        if not np.isfinite(score.flat[k]):
            return self._nearest_unseen(pos)
        i, j = np.unravel_index(k, score.shape)
        return Vector2((i0 + i + 0.5) * c, (j0 + j + 0.5) * c)

    def is_seen(self, pos: Vector2) -> bool:
        i = min(self.cols - 1, max(0, int(pos.x // self.cell)))
        j = min(self.rows - 1, max(0, int(pos.y // self.cell)))
        return bool(self.seen[i, j])

    def is_complete(self) -> bool:
        return bool(self.seen.all())

    # ---- indicateurs ------------------------------------------------------- #
    def stats(self) -> dict:
        return {
            "explored": float(self.seen.mean()),
            "remaining": int((~self.seen).sum()),
        }

    # ---- rendu ------------------------------------------------------------- #
    def render(self, screen: pygame.Surface, frame: int, every: int = 4) -> None:
        if self._cache is None or frame - self._cache_frame >= every:
            a = np.where(self.seen, 0.4, 0.0)[..., None]
            bg = np.array(C_BG, dtype=np.float64)
            fg = np.array(C_COV, dtype=np.float64)
            rgb = bg * (1.0 - a) + fg * a
            surf = pygame.surfarray.make_surface(rgb.astype(np.uint8))
            self._cache = pygame.transform.scale(
                surf, (self.cols * self.cell, self.rows * self.cell)
            )
            self._cache_frame = frame
        screen.blit(self._cache, (0, 0))


# --------------------------------------------------------------------------- #
#  1. L'entite logique : le drone                                              #
#  4. Commande en vitesse + arbitrage par allocation prioritaire               #
# --------------------------------------------------------------------------- #


class Drone:
    """
    Agent autonome. Etat interne : position, cap, module de vitesse.

    On raisonne en vitesse et non en acceleration : chaque comportement exprime
    une correction de vitesse souhaitee dv (px/s), l'arbitre en retient une
    combinaison, et la cinematique du drone -- taux de virage borne, acceleration
    longitudinale bornee, vitesse bornee -- se charge de dire ce qui est
    reellement realisable dans le pas de temps. La dynamique du 2nd ordre
    (masse, force) est donc remplacee par un asservissement de vitesse sature,
    ce qui correspond a ce qu'expose reellement l'autopilote d'un multirotor :
    on lui envoie une consigne de vitesse, pas une consigne de poussee.
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

    # ------------------------------------------------------------------ #
    #  Les comportements. Chacun renvoie une correction de vitesse.       #
    # ------------------------------------------------------------------ #

    def _w_separation(self, nb) -> Vector2:
        """Increment lateral pur, d'intensite proportionnelle a l'intrusion."""
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
        # l'urgence reste purement geometrique : la fonction d'influence pondere
        # la direction de fuite, jamais le declenchement de la securite.
        return set_mag(push, p.v_max * urgency) * p.g_separation

    def _w_alignment(self, nb) -> Vector2:
        """En vitesse, l'alignement s'ecrit directement : dv = g (v_moy - v)."""
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
        # approche amortie : on ralentit en arrivant sur le barycentre
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

    def _w_coverage(self, cov: CoverageMap, frame: int) -> Vector2:
        p = self.params
        if p.g_coverage <= 0.0:
            return Vector2(0, 0)
        # Si la cible a ete decouverte entre-temps (par ce drone ou un autre),
        # on l'abandonne pour en rechercher une nouvelle.
        if self.target is not None and cov.is_seen(self.target):
            self.target = None
        # Cible reevaluee au plus une fois toutes les coverage_every frames, avec
        # un decalage par drone pour lisser la charge.
        if frame >= self.next_eval or self.target is None:
            h = Vector2(math.cos(self.heading), math.sin(self.heading))
            self.target = cov.best_target(self.pos, h, p.lookahead_cells)
            self.next_eval = frame + max(1, int(p.coverage_every))
        if self.target is None:
            return Vector2(0, 0)
        offset = self.target - self.pos
        if offset.length() < p.capture_factor * p.sensor_radius:
            self.target = None  # atteinte
            return Vector2(0, 0)
        return (set_mag(offset, p.v_max) - self.vel) * p.g_coverage

    #  Arbitrage (separation, alignement, cohesion etc.) :

    def decide(self, nb, cov: CoverageMap, bounds: pygame.Rect, frame: int) -> Vector2:
        p = self.params
        want = {
            "separation": self._w_separation(nb),
            "boundary": self._w_boundary(bounds),
            "coverage": self._w_coverage(cov, frame),
            "alignment": self._w_alignment(nb),
            "cohesion": self._w_cohesion(nb),
        }

        if p.arbitration == "weighted":
            # Cas de Reference par somme ponderee.
            total = Vector2(0, 0)
            for name in PRIORITY_ORDER:
                total += want[name]
            self.alloc = {n: want[n].length() for n in PRIORITY_ORDER}
            return limit(self.vel + limit(total, p.authority), p.v_max)

        # Autre choix par allocation prioritaire :
        total = Vector2(0, 0)
        remaining = p.authority
        alloc: dict[str, float] = {}
        for name in PRIORITY_ORDER:
            dv = want[name]
            n = dv.length()
            if remaining <= 1e-6 or n <= 1e-9:
                alloc[name] = 0.0
                continue
            if n > remaining:
                dv = dv * (remaining / n)
                n = remaining
            total += dv
            remaining -= n
            alloc[name] = n
        self.alloc = alloc
        return limit(self.vel + total, p.v_max)

    #  Cinematique du drone : saturation en cap, en vitesse, en accel.

    def apply(self, v_des: Vector2, dt: float, bounds: pygame.Rect) -> None:
        p = self.params
        if v_des.length_squared() > 1e-9:
            d_th = wrap_pi(math.atan2(v_des.y, v_des.x) - self.heading)
            lim = math.radians(p.omega_max) * dt
            self.heading = wrap_pi(self.heading + max(-lim, min(lim, d_th)))

        sp_des = min(p.v_max, max(p.v_min, v_des.length()))
        d_sp = sp_des - self.speed
        lim_a = p.a_max * dt
        self.speed = min(
            p.v_max, max(p.v_min, self.speed + max(-lim_a, min(lim_a, d_sp)))
        )

        self.vel = Vector2(math.cos(self.heading), math.sin(self.heading)) * self.speed
        self.pos += self.vel * dt
        self._boundary(bounds)

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

    # ------------------------------------------------------------------ #
    def draw(self, screen: pygame.Surface, show_sensor: bool) -> None:
        if show_sensor:
            pygame.draw.circle(
                screen,
                C_COV,
                (int(self.pos.x), int(self.pos.y)),
                int(self.params.sensor_radius),
                1,
            )
        ca, sa = math.cos(self.heading), math.sin(self.heading)
        pts = [
            (self.pos.x + lx * ca - ly * sa, self.pos.y + lx * sa + ly * ca)
            for lx, ly in ((9, 0), (-5, 4), (-2.5, 0), (-5, -4))
        ]
        pygame.draw.polygon(screen, self.color, pts)


#  Le simulateur :


class Swarm:
    def __init__(self, params: Params, bounds: pygame.Rect, seed: int | None = None):
        self.params = params
        self.bounds = bounds
        self.rng = random.Random(seed)
        self.neigh = Neighborhood()
        self.drones: list[Drone] = []
        self.coverage = CoverageMap(bounds.width, bounds.height, params.cell_size)
        self.frame = 0
        self.time = 0.0
        self.done = False
        self.completion_time: float | None = None
        self.alloc_mean: dict[str, float] = {n: 0.0 for n in PRIORITY_ORDER}
        self.last_nbs: list[list[tuple]] = []
        self.reset()

    def reset(self) -> None:
        self.drones.clear()
        self.time = 0.0
        self.frame = 0
        self.done = False
        self.completion_time = None
        self.coverage = CoverageMap(
            self.bounds.width, self.bounds.height, self.params.cell_size
        )
        for i in range(int(self.params.n_drones)):
            self._spawn(i)

    def _spawn(self, index: int) -> None:
        pos = Vector2(
            self.rng.uniform(self.bounds.left + 20, self.bounds.right - 20),
            self.rng.uniform(self.bounds.top + 20, self.bounds.bottom - 20),
        )
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
        while len(self.drones) < n:
            self._spawn(len(self.drones))
        if len(self.drones) > n:
            del self.drones[n:]

    def step(self, dt: float) -> None:
        self.params.sanitize()
        self.frame += 1
        self.time += dt
        self._sync_population()

        nbs = self.neigh.build(
            self.drones, self.params, with_metrics=(self.frame % 10 == 0)
        )
        self.last_nbs = nbs

        desired = [
            d.decide(nbs[i], self.coverage, self.bounds, self.frame)
            for i, d in enumerate(self.drones)
        ]
        for d, v in zip(self.drones, desired):
            d.apply(v, dt, self.bounds)

        self.coverage.mark_all(self.drones, self.params.sensor_radius)

        # Fin de mission : toute la zone a ete decouverte.
        if not self.done and self.coverage.is_complete():
            self.done = True
            self.completion_time = self.time

        if self.frame % 10 == 0 and self.drones:
            n = float(len(self.drones))
            self.alloc_mean = {
                k: sum(d.alloc.get(k, 0.0) for d in self.drones) / n
                for k in PRIORITY_ORDER
            }

    def polarization(self) -> float:
        """Parametre d'ordre : 1 = essaim parfaitement aligne, 0 = desordonne."""
        if not self.drones:
            return 0.0
        sx = sum(math.cos(d.heading) for d in self.drones)
        sy = sum(math.sin(d.heading) for d in self.drones)
        return math.hypot(sx, sy) / len(self.drones)

    def metrics(self) -> dict:
        m = self.coverage.stats()
        m.update(
            {
                "polar": self.polarization(),
                "degree": self.neigh.mean_degree,
                "nn_dist": self.neigh.mean_nn_dist,
                "violations": self.neigh.violations,
                "comps": float(self.neigh.n_components),
                "largest": self.neigh.largest_frac,
            }
        )
        return m


#  Interface :


def draw_hud(
    screen, font, params: Params, swarm: Swarm, selected: int, fps: float
) -> None:
    m = swarm.metrics()
    if swarm.done and swarm.completion_time is not None:
        temps_txt = f"temps {swarm.completion_time:6.1f} s  [TERMINEE]"
    else:
        temps_txt = f"temps {swarm.time:6.1f} s"
    header = [
        (
            f"FPS {fps:5.1f}   drones {len(swarm.drones):3d}   {temps_txt}",
            (200, 200, 205),
        ),
        (
            f"voisinage  : {params.neighborhood}"
            + (
                f" (k={params.k_neighbors})"
                if params.neighborhood == "knn"
                else f" (R={params.perception_radius:.0f})"
            ),
            (150, 200, 240),
        ),
        (f"influence  : {params.influence}", (150, 200, 240)),
        (f"arbitrage  : {params.arbitration}", (150, 200, 240)),
        (f"bords      : {params.boundary_mode}", (240, 180, 120)),
        ("", None),
        (
            f"couverture  {m['explored'] * 100:5.1f} %   restant {m['remaining']:4d} cell",
            (120, 220, 190),
        ),
        (
            f"polarisation {m['polar']:5.2f}   voisins {m['degree']:4.1f}",
            (200, 200, 205),
        ),
        (
            f"groupes {int(m['comps']):3d}   plus grand {m['largest'] * 100:5.1f} %",
            (200, 200, 205),
        ),
        (
            f"d_min moyen {m['nn_dist']:5.1f} px   conflits {m['violations']:3d}",
            (220, 140, 140),
        ),
        ("", None),
    ]
    h = 24 + 19 * (len(header) + len(TUNABLES) + len(PRIORITY_ORDER) + 2)
    panel = pygame.Surface((268, h), pygame.SRCALPHA)
    panel.fill((18, 18, 20, 210))
    screen.blit(panel, (10, 10))

    y = 20
    for txt, col in header:
        if txt:
            screen.blit(font.render("  " + txt, True, col), (18, y))
        y += 19

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
            (18, y),
        )
        y += 19

    y += 10
    screen.blit(font.render("  budget alloue (px/s)", True, (170, 170, 178)), (18, y))
    y += 19
    total = max(1e-6, params.authority)
    for name in PRIORITY_ORDER:
        v = swarm.alloc_mean.get(name, 0.0)
        w = int(90 * min(1.0, v / total))
        screen.blit(font.render(f"  {name[:9]:<10}", True, (150, 150, 158)), (18, y))
        pygame.draw.rect(screen, (55, 55, 62), pygame.Rect(120, y + 4, 90, 8), 1)
        if w:
            pygame.draw.rect(screen, (120, 200, 170), pygame.Rect(120, y + 4, w, 8))
        screen.blit(font.render(f"{v:5.0f}", True, (150, 150, 158)), (216, y))
        y += 19


def draw_done_banner(screen, font_big, swarm: Swarm, W: int) -> None:
    if not swarm.done or swarm.completion_time is None:
        return
    msg = (
        f"MISSION TERMINEE — 100 % couvert en "
        f"{swarm.completion_time:.1f} s   (R : relancer)"
    )
    surf = font_big.render(msg, True, (60, 190, 150))
    rect = surf.get_rect(center=(W // 2, 40))
    bgr = pygame.Surface((rect.width + 24, rect.height + 16), pygame.SRCALPHA)
    bgr.fill((18, 18, 20, 225))
    screen.blit(bgr, (rect.left - 12, rect.top - 8))
    screen.blit(surf, rect)


def cycle(seq, cur):
    return seq[(seq.index(cur) + 1) % len(seq)]


def main(headless: bool = False, max_frames: int = 0, seed: int | None = None) -> None:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    W, H = 1180, 740
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Essaim de drones — decouverte d'une zone fixe 2D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)
    font_big = pygame.font.SysFont("monospace", 24, bold=True)

    params = Params()
    bounds = pygame.Rect(0, 0, W, H)
    swarm = Swarm(params, bounds, seed)

    selected = 0
    paused = False
    show_cov, show_vision, show_sensor = True, False, False
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
                    show_cov = not show_cov
                elif ev.key == pygame.K_v:
                    show_vision = not show_vision
                    if not show_vision and swarm.drones:
                        for d in swarm.drones:
                            d.color = C_DRONE
                elif ev.key == pygame.K_s:
                    show_sensor = not show_sensor
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

        # La mission se fige une fois la zone entierement decouverte.
        if not paused and not swarm.done:
            swarm.step(dt)

        screen.fill(C_BG)
        if show_cov:
            swarm.coverage.render(screen, swarm.frame)
        if params.boundary_mode == "steer":
            m = int(params.boundary_margin)
            pygame.draw.rect(
                screen, (55, 55, 62), pygame.Rect(m, m, W - 2 * m, H - 2 * m), 1
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
            for other, _d, _w in (swarm.last_nbs[0] if swarm.last_nbs else []):
                other.color = (200, 160, 90)
                pygame.draw.line(screen, (90, 70, 50), d0.pos, other.pos, 1)
            if d0.target is not None:
                pygame.draw.line(screen, (60, 190, 150), d0.pos, d0.target, 1)

        # Rayon de perception de chaque drone (touche P).
        if show_radii and swarm.drones:
            for i, d in enumerate(swarm.drones):
                if params.neighborhood == "metric":
                    r = int(params.perception_radius)
                else:
                    # en kNN, le rayon effectif = distance au voisin le plus loin
                    nb = swarm.last_nbs[i] if i < len(swarm.last_nbs) else []
                    r = int(max((dd for _o, dd, _w in nb), default=0))
                if r > 0:
                    pygame.draw.circle(
                        screen, (45, 70, 90), (int(d.pos.x), int(d.pos.y)), r, 1
                    )

        for d in swarm.drones:
            d.draw(screen, show_sensor)
        draw_hud(screen, font, params, swarm, selected, clock.get_fps())
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
