"""
Essaim de drones : ratissage par PATROUILLES, cycles exploration / ralliement.

Echelle physique : toutes les grandeurs internes sont en METRES et en SECONDES ;
seul l'affichage convertit en pixels (PX_PER_M). La zone couvre 400 x 250 m
(10 ha), la grille a une resolution de 2 m, si bien qu'une cellule cible
represente approximativement un individu au sol.

L'UNITE DE MANOEUVRE EST LA PATROUILLE, PAS LE DRONE
  Version precedente : en exploration la cohesion etait coupee et l'alignement
  residuel, si bien que l'essaim n'existait plus pendant 85 % de la mission ; il
  ne restait que N agents de couverture independants. Les cartes etant
  identiques a la sortie d'un partage et le choix de cible etant un argmax
  deterministe, il fallait injecter un biais angulaire exogene (theta_i =
  2 pi i / N) pour eviter que tous ne repartent vers la meme cellule. Cette
  assignation centralisee deguisee est precisement ce que l'on veut eviter.

  Ici les drones sont repartis en PATROUILLES de `squad_size` appareils. Cette
  partition est deterministe (indice // taille), donc connue avant le decollage
  au meme titre que la suite des points de rendez-vous : elle ne suppose aucun
  echange. Le voisinage d'interaction est alors scinde :

    - cohesion et alignement  -> membres de MA patrouille uniquement ;
    - dispersion longue portee -> membres des AUTRES patrouilles uniquement ;
    - separation               -> tout le monde, c'est de la securite ;
    - anti-redondance          -> tout le monde (voir ci-dessous).

  Consequence en cascade : si l'espacement intra-patrouille reste inferieur a la
  portee radio, les membres d'une patrouille sont connectes en permanence, donc
  la fusion leur donne des cartes identiques, donc le ciblage leur renvoie la
  MEME cellule. Ce qui etait le defaut que l'eventail corrigeait devient le
  comportement voulu : la patrouille se deplace comme un corps. La brisure de
  symetrie n'est plus necessaire qu'ENTRE patrouilles.

ANTI-REDONDANCE : REPULSION ENTRE EMPREINTES, PAS ENTRE POSITIONS
  Ce que l'on veut eviter n'est pas la proximite mais le RECOUVREMENT DES
  EMPREINTES CAPTEUR. La repulsion s'exerce donc entre les centres d'empreinte
  c_i = pos_i + (R_capteur / 2) u(cap_i), avec pour portee la largeur de
  fauchee w = f 2 R sin(demi-angle).

  Geometriquement : l'alignement fait converger les caps, donc les empreintes se
  placent cote a cote, donc la repulsion devient purement laterale. L'equilibre
  entre cohesion et anti-redondance produit spontanement une LIGNE DE FRONT
  perpendiculaire au cap, a espacement egal a la fauchee. On obtient un
  ratissage en ligne, qui est a la fois du Reynolds au sens strict (separation
  modifiee, alignement, cohesion) et la maniere dont un ratissage se fait
  reellement.

RESERVATION D'INTENTION AU LIEU DE L'EVENTAIL
  Pour differencier deux PATROUILLES qui sortent d'un partage avec la meme
  carte, on echange ce que des drones connectes peuvent reellement s'echanger :
  leur cible courante, quelques octets contre des cartes entieres. Le score de
  ciblage est penalise par une gaussienne autour des cibles annoncees par les
  drones de MA composante de communication appartenant a une AUTRE patrouille.

  Le desordre necessaire est deja present : `next_eval` desynchronise les
  reevaluations. Le premier qui evalue reserve, les suivants voient la
  reservation et se reportent ailleurs. C'est de l'asynchronisme, pas un angle
  code en dur, et le mecanisme se degrade proprement : deux patrouilles hors de
  portee radio peuvent reserver la meme cellule, ce qui est exactement
  l'inefficacite honnete que le modele veut representer.

  L'eventail est conserve mais DESACTIVE par defaut (touche E) : il sert de
  condition temoin, la brisure de symetrie exogene contre laquelle on mesure la
  brisure endogene. Quand il est actif, le secteur est attribue par patrouille.

TIRAGE HIERARCHIQUE DES CIBLES
  Les cibles ne sont pas tirees en grappes gaussiennes independantes mais par
  une DISPERSION HIERARCHIQUE ORIENTEE, qui imite l'articulation d'une unite en
  progression : une section engendre des groupes, qui engendrent des equipes,
  qui engendrent des soldats. Chaque noeud place ses enfants dans le demi-ovale
  AVANT de sa direction de propagation, de sorte que la formation possede un axe
  d'avance et une structure emboitee a plusieurs echelles.

  Seuls les noeuds TERMINAUX sont des cibles : les niveaux intermediaires sont
  des centres de groupe abstraits, pas des individus au sol. Avec la
  configuration retenue (voir HIER_CONFIG), on obtient une vingtaine de soldats
  repartis dans une empreinte d'environ 35 x 145 m, orientee aleatoirement dans
  la zone.

  Interet pour l'etude : la correlation spatiale n'est pas a une seule echelle
  mais a trois (equipe, groupe, section). La dopamine devient donc informative a
  plusieurs portees, et le cortisol peut a l'inverse condamner a tort un secteur
  situe entre deux groupes.

  Quelques cibles ISOLEES sont ajoutees (eclaireurs detaches) pour empecher
  l'essaim de se contenter d'exploiter la formation principale.

BOUCLE DE MISSION EN TROIS PHASES
  1. EXPLORATION  : les patrouilles ratissent en ligne de front. La portee radio
     (45 m) est tres inferieure a l'espacement moyen entre patrouilles, donc
     l'essaim se fragmente EN PATROUILLES et chacune accumule une connaissance
     PROPRE, ignoree des autres.
  2. RALLIEMENT   : a instant fixe, tous convergent vers un point de rendez-vous.
     Des que le graphe de communication ne forme plus qu'une seule composante,
     les cartes fusionnent : confiance, dopamine, cortisol, cibles connues.
  3. PARTAGE      : bref maintien en formation, puis redispersion. Les drones
     repartent avec la connaissance agregee : les zones deja ratissees par
     d'autres ne les attirent plus, et les foyers de dopamine decouverts par
     d'autres les attirent, y compris a longue distance.
  Puis retour en 1, jusqu'a ce que toutes les cibles soient trouvees.

L'arbitrage est adapte a chaque phase de DEUX facons simultanees :
  - les gains de base sont modules par un profil de phase (multiplicateurs) ;
  - l'ORDRE DE PRIORITE lui-meme est permute. En exploration la couverture et
    l'anti-redondance passent devant l'alignement et la cohesion ; en ralliement
    le comportement de ralliement remonte juste apres la securite, et
    l'anti-redondance comme la dispersion sont coupees.

COUPLAGE DE LA DOPAMINE  (parametre `dopa_saturate`, touche D)
  La dopamine agit a trois endroits : attirance dans le score de ciblage, gain
  du taux de detection, et ralentissement du scan. Deux reponses possibles :
    - LINEAIRE (defaut) : l'effet est proportionnel a la concentration, qui
      n'est pas bornee. L'essaim converge massivement sur les foyers decouverts
      et s'y attarde : emballement par retroaction positive (plus on ralentit,
      plus on detecte, plus on depose), conserve ici parce qu'il est
      precisement l'objet d'etude.
    - SATUREE : on remplace la concentration x par x/(1+x), borne a 1. Pour une
      amplitude comparable il faut alors remonter les gains d'un ordre de
      grandeur (Attir. dopa ~ 30, Gain dopa ~ 30).

Le point de rendez-vous ne peut pas etre negocie : pendant l'exploration les
patrouilles sont deconnectees entre elles. Il doit donc etre deductible sans
echange.
  - "truck"    : retour a la base, trivialement connu de tous.
  - "schedule" : suite de points fixee AVANT le decollage a partir de la
                 geometrie de la zone (boustrophedon). Mode par defaut.
  - "centroid" : barycentre de l'essaim. ORACLE, borne de comparaison seulement.
La phase est globale, ce qui suppose des horloges synchronisees (GPS).

Autres choix de modelisation :
  - Entite      : drone a voilure tournante, sature en vitesse, en acceleration
                  longitudinale et en taux de virage. Vitesse non constante.
  - Voisinage   : au temps courant, metrique (disque R) ou k plus proches voisins.
  - Influence   : uniforme, ou ponderation a decroissance quadratique.
  - Commande    : raisonnement en vitesse (1er ordre), pas en force.
  - Obstacles   : disques aleatoires. Bloquent le deplacement ET occultent le
                  capteur. La radio passe au travers.
  - Capteur     : cone oriente selon le cap (portee + demi-angle).
  - Connaissance: carte de confiance dans [0,1], c <- 1 - (1-c) exp(-lambda dt).
  - Pheromones  : DOPAMINE (orange) sur cible detectee, CORTISOL (bleu) sur zone
                  observee et sterile. Depot -> diffusion -> evaporation.

Commandes clavier :
    HAUT/BAS         selectionner un parametre
    GAUCHE/DROITE    ajuster le parametre selectionne
    N   voisinage    : metrique <-> k plus proches voisins
    I   influence    : aucune -> quadratique -> quadratique a support compact
    A   arbitrage    : allocation prioritaire <-> somme ponderee
    D   reponse a la dopamine : lineaire <-> saturee
    E   eventail (condition temoin) : actif <-> inactif
    B   bords        : steer / wrap / bounce
    F   rendez-vous  : schedule -> truck -> centroid
    G   forcer le passage a la phase suivante
    C   couche       : tout -> confiance -> dopamine -> cortisol -> aucune
    M   vue          : connaissance collective <-> celle du drone 0
    L   liens de communication      V  voisinage du drone 0
    S   cone capteur                P  rayons de communication
    W   empreintes capteur et fauchee
    T   structure hierarchique des cibles
    H   afficher / masquer le panneau
    ESPACE pause     R reinitialiser      ECHAP quitter
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pygame
from pygame import Vector2

#  Echelle et parametrage :

ZONE_W_M = 400  # largeur de la zone de mission, en metres
ZONE_H_M = 250  # hauteur de la zone de mission, en metres

# L'echelle d'affichage est deduite d'une taille de fenetre cible, pour que la
# zone puisse etre agrandie sans que la fenetre deborde de l'ecran.
WIN_MAX_W, WIN_MAX_H = 1150, 700
PX_PER_M = min(WIN_MAX_W / ZONE_W_M, WIN_MAX_H / ZONE_H_M)

NEIGHBORHOODS = ("metric", "knn")
INFLUENCES = ("none", "quad", "quad_compact")
ARBITRATIONS = ("priority", "weighted")
BOUNDARIES = ("steer", "wrap", "bounce")
LAYERS = ("tout", "confiance", "dopamine", "cortisol", "aucune")
RALLY_MODES = ("schedule", "truck", "centroid")

# Tous les comportements existants. L'ordre effectif depend de la phase.
# "swath" : anti-redondance entre empreintes capteur (nouveau).
# "spread": dispersion longue portee, desormais INTER-PATROUILLES uniquement.
ALL_BEHAVIORS = (
    "separation",
    "obstacle",
    "boundary",
    "rally",
    "coverage",
    "swath",
    "spread",
    "alignment",
    "cohesion",
)


# =========================================================================== #
#  TIRAGE HIERARCHIQUE DES CIBLES                                             #
#                                                                             #
#  Dispersion hierarchique multi-echelle avec propagation d'une direction de   #
#  croissance. Chaque noeud de niveau i place ses N_i enfants dans le          #
#  DEMI-OVALE AVANT d'une ellipse (L_i, l_i) dont il occupe le sommet arriere. #
#                                                                             #
#  Notations : c_i centre, d_i direction unitaire, v_i = rot(d_i, +90),        #
#  R_i distance max entre deux enfants voisins, L_i = m_i R_i N_i longueur de  #
#  l'ellipse le long de v_i, l_i = k_i L_i sa largeur le long de d_i,          #
#  O_i = c_i + (l_i/2) d_i son centre geometrique, R_ext(n) le rayon           #
#  d'encombrement reel du sous-arbre, MESURE et non majore analytiquement.     #
#                                                                             #
#  Construction ASCENDANTE : chaque sous-arbre est assemble dans son propre    #
#  repere (origine, +x), son R_ext exact est mesure, puis le parent le place   #
#  en connaissant son encombrement reel et l'amene par rotation-translation.   #
# =========================================================================== #

FACTEUR_PAS = 0.90  # f_pas : pas nominal vise, en fraction de R_i
FACTEUR_AXIAL = 0.95  # f_ax : ecart axial maximal tolere entre deux voisins
POIDS_RADIAL = 0.25  # w : 0 = direction du parent, 1 = axe sortant


@dataclass
class ConfigNiveau:
    """Parametres utilises par un noeud de niveau i pour placer ses enfants."""

    N: int | Sequence[int]  # entier fixe, ou liste de valeurs possibles
    R: float  # R_i, distance max entre deux enfants voisins (m)
    nom: str = ""
    dtheta: float = 0.0  # variation angulaire max (rad)
    marge: float = 0.7  # m_i, marge de l'ellipse
    ratio_l: float = 1.0 / 3.0  # k_i, largeur / longueur
    facteur_Rmax: float = 1.0  # q_i, compacite (1 = contrainte inactive)
    enfant_axial: bool = False  # un enfant place sur l'axe, en avant du centre
    position_axiale: float = 0.5  # f_i, fraction de l'avance l_i / 2

    def tirer_N(self, rng) -> int:
        if isinstance(self.N, (int, np.integer)):
            return int(self.N)
        return int(rng.choice(np.asarray(self.N, dtype=int)))

    def dimensions(self, N):
        L = self.marge * self.R * float(N)
        return L, self.ratio_l * L

    def Rmax(self, N) -> float:
        if N < 2:
            return math.inf
        return self.facteur_Rmax * (N - 1) * self.R


@dataclass
class Noeud:
    """Un element de la hierarchie, une fois place."""

    niveau: int
    centre: np.ndarray
    direction: np.ndarray
    rayon: float  # R_ext reel
    L: float = 0.0
    l: float = 0.0
    R: float = math.inf
    Rmax: float = math.inf
    nom: str = ""
    enfants: list[Noeud] = field(default_factory=list)

    @property
    def est_terminal(self) -> bool:
        return len(self.enfants) == 0

    @property
    def avance(self) -> float:
        return self.l / 2.0


# Configuration retenue, calibree pour la zone 400 x 250 m : environ 23 soldats
# dans une empreinte de 35 x 145 m.
HIER_CONFIG = [
    ConfigNiveau(
        N=[3, 4],
        R=60.0,
        nom="section",
        dtheta=math.radians(8.0),
        enfant_axial=True,
        position_axiale=0.2,
    ),
    ConfigNiveau(N=[2, 3], R=22.0, nom="groupe", dtheta=math.radians(15.0)),
    ConfigNiveau(N=[2, 3], R=7.0, nom="equipe", dtheta=math.radians(35.0)),
]
HIER_R_TERMINAL = 1.0  # rayon propre d'un soldat (m)


def _unitaire(x):
    n = float(np.linalg.norm(x))
    return np.array([1.0, 0.0]) if n < 1e-12 else np.asarray(x) / n


def _perpendiculaire(d):
    return np.array([-d[1], d[0]])


def parcourir(noeud):
    """Generateur : tous les noeuds de l'arbre."""
    yield noeud
    for e in noeud.enfants:
        yield from parcourir(e)


def transformer(noeud, origine, d_cible):
    """Amene un sous-arbre du repere local a sa place definitive."""
    d = _unitaire(d_cible)
    M = np.array([[d[0], -d[1]], [d[1], d[0]]])
    origine = np.asarray(origine, dtype=float)
    for n in parcourir(noeud):
        n.centre = origine + M @ n.centre
        n.direction = M @ n.direction
    return noeud


def mesurer_R_ext(noeud, R_terminal):
    """Rayon du plus petit disque centre en c_n contenant tout le sous-arbre."""
    d_max = max(
        float(np.linalg.norm(m.centre - noeud.centre)) for m in parcourir(noeud)
    )
    return d_max + float(R_terminal)


def _contraintes_ok(p, r_p, centre, d, places, rayons_places, R_voisin, Rmax, s):
    """(C2) non-chevauchement, (C3) voisinage, (C4) etendue, (C5) demi-ovale."""
    if float(np.dot(p - centre, d)) < s - 1e-9:
        return False
    if not places:
        return True
    dist = [float(np.linalg.norm(p - q)) for q in places]
    for dk, rk in zip(dist, rayons_places):
        if dk < r_p + rk:
            return False
    if min(dist) > R_voisin:
        return False
    if max(dist) > Rmax:
        return False
    return True


def placer_enfants(centre, d, L, l, rayons, R_voisin, Rmax, rng, essais_max=300):
    """Place les enfants dans le demi-ovale avant. Retourne (positions, echecs)."""
    N = len(rayons)
    if N <= 0:
        return [], 0

    d = _unitaire(d)
    v = _perpendiculaire(d)
    avance = l / 2.0
    O = centre + avance * d
    r_max = max(rayons)

    a_demi = max(L / 2.0 - r_max, 1e-9)
    b_demi = max(l / 2.0 - r_max, 1e-9)

    if N == 1:
        pas, a_nom = 0.0, np.zeros(1)
    else:
        pas = min(FACTEUR_PAS * R_voisin, 2.0 * a_demi / (N - 1))
        a_nom = (np.arange(N) - (N - 1) / 2.0) * pas

    jeu = max(0.0, pas - 2.0 * r_max)
    da_max = max(
        0.0, min(0.30 * pas, 0.45 * jeu, 0.5 * (FACTEUR_AXIAL * R_voisin - pas))
    )
    ecart_axial = pas + 2.0 * da_max
    b_secu = math.sqrt(max(0.0, R_voisin**2 - ecart_axial**2))

    positions, echecs = [], 0
    for k in range(N):
        p_final, ok = None, False
        a = float(a_nom[k])
        for essai in range(essais_max):
            amp = max(0.0, 1.0 - essai / (0.9 * essais_max))
            a = float(
                np.clip(
                    a_nom[k] + amp * rng.uniform(-1.0, 1.0) * da_max, -a_demi, a_demi
                )
            )
            b_lim = min(b_demi * math.sqrt(max(0.0, 1.0 - (a / a_demi) ** 2)), b_secu)
            b = amp * rng.uniform(0.0, 1.0) * b_lim
            p = O + a * v + b * d
            if _contraintes_ok(
                p, rayons[k], centre, d, positions, rayons[:k], R_voisin, Rmax, avance
            ):
                p_final, ok = p, True
                break
        if not ok:
            p_final = O + a_nom[k] * v  # repli deterministe
            echecs += 1
        positions.append(p_final)
    return positions, echecs


def placer_enfant_axial(
    centre, d, l, r_axial, position_axiale, places, rayons_places, R_voisin
):
    """Place un enfant sur l'axe, entre le centre et le demi-ovale avant."""
    d = _unitaire(d)
    f0 = float(np.clip(position_axiale, 0.05, 0.95))
    candidats = [f0]
    for k in range(1, 19):
        for f in (f0 - 0.05 * k, f0 + 0.05 * k):
            if 0.05 <= f <= 0.95:
                candidats.append(f)
    for f in candidats:
        p = centre + f * (l / 2.0) * d
        if not places:
            return p, True
        dist = [float(np.linalg.norm(p - q)) for q in places]
        if any(dk < r_axial + rk for dk, rk in zip(dist, rayons_places)):
            continue
        if min(dist) > R_voisin:
            continue
        return p, True
    return centre + f0 * (l / 2.0) * d, False


def direction_enfant(centre, d, p, dtheta, rng, w=POIDS_RADIAL):
    """Melange entre direction du parent et axe sortant, plus perturbation."""
    d = _unitaire(d)
    r = _unitaire(p - centre)
    alpha = math.atan2(d[1], d[0])
    beta = math.atan2(r[1], r[0])
    delta = (beta - alpha + math.pi) % (2.0 * math.pi) - math.pi
    eps = rng.uniform(-dtheta, dtheta) if dtheta > 0.0 else 0.0
    a = alpha + w * delta + eps
    return np.array([math.cos(a), math.sin(a)])


def construire_local(niveau, configs, rng, R_terminal=0.0):
    """Construit le sous-arbre de niveau `niveau` dans son repere local."""
    noeud = Noeud(
        niveau=niveau,
        centre=np.zeros(2),
        direction=np.array([1.0, 0.0]),
        rayon=float(R_terminal),
        nom=configs[niveau].nom if niveau < len(configs) else "terminal",
    )
    if niveau >= len(configs):
        return noeud

    cfg = configs[niveau]
    N = cfg.tirer_N(rng)
    if N <= 0:
        return noeud

    L, l = cfg.dimensions(N)
    noeud.L, noeud.l, noeud.R = L, l, cfg.R
    noeud.Rmax = cfg.Rmax(N)

    sous_arbres = [
        construire_local(niveau + 1, configs, rng, R_terminal) for _ in range(N)
    ]
    rayons = [mesurer_R_ext(sa, R_terminal) for sa in sous_arbres]
    for sa, r in zip(sous_arbres, rayons):
        sa.rayon = r

    idx_axial = int(rng.integers(N)) if (cfg.enfant_axial and N >= 2) else None
    arc = [k for k in range(N) if k != idx_axial]

    positions_arc, _ = placer_enfants(
        noeud.centre,
        noeud.direction,
        L,
        l,
        [rayons[k] for k in arc],
        cfg.R,
        noeud.Rmax,
        rng,
    )
    positions = [None] * N
    for k, p in zip(arc, positions_arc):
        positions[k] = p

    if idx_axial is not None:
        p_ax, _ = placer_enfant_axial(
            noeud.centre,
            noeud.direction,
            l,
            rayons[idx_axial],
            cfg.position_axiale,
            positions_arc,
            [rayons[k] for k in arc],
            cfg.R,
        )
        positions[idx_axial] = p_ax

    for p, sa in zip(positions, sous_arbres):
        d_fils = direction_enfant(noeud.centre, noeud.direction, p, cfg.dtheta, rng)
        transformer(sa, p, d_fils)
        noeud.enfants.append(sa)

    noeud.rayon = mesurer_R_ext(noeud, R_terminal)
    return noeud


def generer_structure(configs, centre=(0.0, 0.0), theta0=0.0, R_terminal=0.0, rng=None):
    """Construit la hierarchie puis l'amene a sa position et orientation."""
    if rng is None:
        rng = np.random.default_rng()
    racine = construire_local(0, configs, rng, R_terminal)
    transformer(racine, centre, np.array([math.cos(theta0), math.sin(theta0)]))
    return racine


def positions_terminales(racine) -> np.ndarray:
    """Centres des seuls noeuds terminaux : les individus au sol."""
    pts = [n.centre for n in parcourir(racine) if n.est_terminal]
    return np.array(pts, dtype=float).reshape(-1, 2)


def aretes_hierarchie(racine) -> list:
    """Segments parent -> enfant, avec le niveau, pour l'affichage."""
    out = []
    for n in parcourir(racine):
        for e in n.enfants:
            out.append((n.centre, e.centre, n.niveau))
    return out


# =========================================================================== #


@dataclass(frozen=True)
class PhaseProfile:
    """
    Profil d'une phase de mission.

    `order` est l'ordre de priorite de l'allocation : c'est LUI qui porte
    l'essentiel de l'adaptation, puisqu'un comportement place en fin de file ne
    recoit que le reliquat de budget laisse par les precedents.

    `mul` multiplie les gains de base definis dans Params.
    """

    label: str
    color: tuple
    order: tuple
    mul: dict


PHASES = {
    # Exploration : la patrouille ratisse en ligne de front. La cohesion et
    # l'alignement sont ACTIFS (contrairement a la version precedente) mais
    # restreints aux membres de la patrouille ; l'anti-redondance ecarte les
    # empreintes capteur, ce qui produit la ligne ; la dispersion longue portee
    # ne s'applique qu'aux autres patrouilles.
    "explore": PhaseProfile(
        label="EXPLORATION",
        color=(120, 220, 190),
        order=(
            "separation",
            "obstacle",
            "boundary",
            "coverage",
            "swath",
            "spread",
            "alignment",
            "cohesion",
            "rally",
        ),
        mul={
            "separation": 1.4,
            "obstacle": 1.0,
            "boundary": 1.0,
            "rally": 0.0,
            "coverage": 1.2,
            "swath": 1.0,
            "spread": 1.0,
            "alignment": 1.0,
            "cohesion": 0.8,
        },
    ),
    # Ralliement : le ralliement remonte juste apres la securite, la couverture
    # tombe en fin de file (on scrute encore un peu en chemin). Anti-redondance
    # et dispersion sont coupees : on veut converger, pas etaler la fauchee.
    "rally": PhaseProfile(
        label="RALLIEMENT",
        color=(240, 180, 120),
        order=(
            "separation",
            "obstacle",
            "boundary",
            "rally",
            "cohesion",
            "alignment",
            "coverage",
            "swath",
            "spread",
        ),
        mul={
            "separation": 1.0,
            "obstacle": 1.0,
            "boundary": 1.0,
            "rally": 1.0,
            "coverage": 0.25,
            "swath": 0.0,
            "spread": 0.0,
            "alignment": 1.2,
            "cohesion": 2.0,
        },
    ),
    # Partage : maintien serre le temps de l'echange.
    "share": PhaseProfile(
        label="PARTAGE",
        color=(150, 200, 240),
        order=(
            "separation",
            "obstacle",
            "boundary",
            "cohesion",
            "rally",
            "alignment",
            "coverage",
            "swath",
            "spread",
        ),
        mul={
            "separation": 1.0,
            "obstacle": 1.0,
            "boundary": 0.8,
            "rally": 0.3,
            "coverage": 0.1,
            "swath": 0.0,
            "spread": 0.0,
            "alignment": 1.5,
            "cohesion": 3.0,
        },
    ),
}

PHASE_CYCLE = {"explore": "rally", "rally": "share", "share": "explore"}


def px(v: float) -> int:
    """Metres -> pixels ecran."""
    return round(v * PX_PER_M)


@dataclass
class Params:
    """Tous les parametres du modele. Unites SI : metres, secondes, m/s."""

    # --- 1. entite logique : le drone -------------------------------------- #
    n_drones: int = 12
    v_max: float = 16.0  # m/s (~58 km/h)
    v_cruise: float = 11.0  # m/s, vitesse de croisiere visee
    v_min: float = 0.0  # m/s, 0 = vol stationnaire permis (multirotor)
    a_max: float = 6.0  # m/s2
    omega_max: float = 70.0  # deg/s (rayon ~13 m a 16 m/s)
    turn_slow_min: float = 0.30  # vitesse residuelle en virage serre (fraction)
    safety_radius: float = 3.0  # m, en deca : quasi-collision comptabilisee

    # --- 2. patrouilles ------------------------------------------------------ #
    # Partition DETERMINISTE de l'essaim, deductible sans echange (au meme titre
    # que la suite des rendez-vous). C'est l'unite de manoeuvre : cohesion et
    # alignement s'exercent a l'interieur, dispersion a l'exterieur.
    squad_size: int = 4

    # --- 3. capteur conique -------------------------------------------------- #
    sensor_range: float = 40.0  # m, portee du cone
    sensor_half_angle: float = 30.0  # deg, demi-angle d'ouverture
    sensor_gain: float = 1.0  # 1/s, taux de detection sur l'axe a d=0
    swath_frac: float = 0.70  # fraction utile de la largeur geometrique

    # --- 4. voisinage (au temps courant) ------------------------------------- #
    neighborhood: str = "metric"
    perception_radius: float = 70.0  # m
    k_neighbors: int = 6

    # --- 5. fonction d'influence ---------------------------------------------- #
    influence: str = "none"
    influence_scale: float = 0.5

    # --- 6. communication ------------------------------------------------------ #
    # Volontairement inferieure a l'espacement moyen ENTRE PATROUILLES, mais
    # superieure a l'espacement intra-patrouille (la fauchee, ~28 m) : une
    # patrouille reste connectee en permanence, deux patrouilles non.
    comm_radius: float = 60.0  # m

    # --- 7. phases de mission --------------------------------------------------- #
    rally_mode: str = "schedule"
    t_explore: float = 60.0  # s, duree d'une phase d'exploration
    t_rally_max: float = 50.0  # s, delai au-dela duquel on renonce au ralliement
    t_share: float = 5.0  # s, maintien en formation pendant l'echange
    rally_arrival: float = 25.0  # m, distance de debut de ralentissement
    rv_cols: int = 3  # points de rendez-vous en x
    rv_rows: int = 2  # points de rendez-vous en y

    # --- 8. brisure de symetrie ---------------------------------------------- #
    # `fan_enabled` = CONDITION TEMOIN. Par defaut la differenciation des
    # patrouilles est endogene (reservations d'intention) ; l'eventail est le
    # biais exogene contre lequel on la compare. Touche E.
    fan_enabled: bool = False
    g_fan: float = 0.6  # amplitude du biais de secteur a la dispersion
    fan_tau: float = 20.0  # s, attenuation de ce biais
    claim_strength: float = 0.90  # profondeur de la penalite de reservation
    claim_radius: float = 45.0  # m, portee de la penalite de reservation

    # --- 9. arbitrage et gains de base ------------------------------------------- #
    arbitration: str = "priority"
    authority: float = 22.0  # m/s, budget total de correction de vitesse
    separation_radius: float = 9.0  # m, securite (courte portee)
    spread_radius: float = 95.0  # m, dispersion INTER-PATROUILLES
    g_separation: float = 1.6
    g_obstacle: float = 2.0
    g_boundary: float = 1.2
    g_coverage: float = 1.0
    g_swath: float = 2.5  # anti-redondance des empreintes
    g_spread: float = 1.2
    g_rally: float = 1.6
    g_alignment: float = 0.9
    g_cohesion: float = 0.9

    # --- 10. bords de la zone ------------------------------------------------------ #
    boundary_mode: str = "steer"
    boundary_margin: float = 20.0  # m

    # --- 11. obstacles --------------------------------------------------------------- #
    n_obstacles_min: int = 2
    n_obstacles_max: int = 6
    obstacle_r_min: float = 10.0  # m
    obstacle_r_max: float = 28.0  # m
    obstacle_margin: float = 9.0  # m, zone de rappel autour du disque
    obstacle_gap: float = 20.0  # m, ecart minimal entre deux obstacles

    # --- 12. deploiement depuis le camion ---------------------------------------------- #
    launch_margin: float = 12.0  # m, recul du camion par rapport au bord
    truck_len: float = 7.0  # m
    truck_wid: float = 2.6  # m
    launch_per_row: int = 4  # drones par rangee (une rangee ~ une patrouille)
    launch_spacing: float = 6.0  # m, pas entre drones sur l'aire
    launch_clearance: float = 25.0  # m, degagement impose autour de l'aire

    # --- 13. cibles : tirage hierarchique ------------------------------------------ #
    hier_tries: int = 80  # essais de placement de la formation dans la zone
    hier_margin: float = 25.0  # m, marge au bord pour l'empreinte de la formation
    n_isolated: int = 3  # eclaireurs detaches, hors formation

    # --- 14. carte de confiance ------------------------------------------------------------ #
    cell_m: float = 2.0  # m, cote d'une cellule (~1 individu)
    detect_threshold: float = 0.5  # confiance validant une cible
    explore_threshold: float = 0.55  # en deca : la cellule merite encore un survol
    sterile_threshold: float = 0.92  # au-dela, sans cible : cellule sterile

    # --- 15. pheromones ------------------------------------------------------------- #
    dopa_saturate: bool = False
    dopa_deposit: float = 40.0  # depot par cible detectee
    dopa_diffuse: float = 0.40  # 1/s, coefficient de diffusion
    dopa_tau: float = 80.0  # s, constante de temps d'evaporation
    dopa_attract: float = 15.5  # poids dans le score de ciblage
    dopa_gain: float = 2.5  # multiplicateur du taux de detection lambda
    dopa_scan_slow: float = 0.35  # ralentissement du scan en zone dopaminee

    cort_deposit: float = 0.45
    cort_diffuse: float = 0.15
    cort_tau: float = 50.0
    cort_repel: float = 0.5

    # --- 16. ciblage -------------------------------------------------------------------------- #
    target_horizon: float = 140.0  # m
    coverage_every: int = 8  # reevaluation de la cible 1 frame sur N
    pher_every: int = 5  # diffusion des pheromones 1 frame sur N
    capture_radius: float = 4.0  # m, cible consideree atteinte
    arrival_radius: float = 14.0  # m, debut de ralentissement sur la cible

    def sanitize(self) -> None:
        """Maintient les invariants apres une edition clavier."""
        self.v_cruise = min(self.v_cruise, self.v_max)
        self.v_min = min(self.v_min, self.v_cruise)
        self.n_drones = max(1, int(self.n_drones))
        self.squad_size = max(1, min(int(self.squad_size), int(self.n_drones)))
        self.k_neighbors = max(1, min(int(self.k_neighbors), max(1, self.n_drones - 1)))
        self.launch_per_row = max(1, int(self.launch_per_row))
        self.cell_m = max(0.5, float(self.cell_m))
        self.rv_cols = max(1, int(self.rv_cols))
        self.rv_rows = max(1, int(self.rv_rows))
        self.n_obstacles_min = max(0, int(self.n_obstacles_min))
        self.n_obstacles_max = max(self.n_obstacles_min, int(self.n_obstacles_max))
        self.claim_strength = min(0.99, max(0.0, float(self.claim_strength)))
        # detect <= explore < sterile : sans cet ordre, une cellule pourrait
        # cesser d'etre attractive avant d'avoir valide la cible qu'elle porte.
        self.detect_threshold = min(0.97, max(0.05, self.detect_threshold))
        self.explore_threshold = min(
            0.98, max(self.explore_threshold, self.detect_threshold)
        )
        self.sterile_threshold = min(
            0.995, max(self.sterile_threshold, self.explore_threshold + 0.03)
        )

    def dopa_response(self, x):
        """
        Reponse aux concentrations de dopamine. En mode lineaire l'effet n'est
        pas borne, ce qui produit l'emballement decrit en en-tete ; en mode
        sature x/(1+x) le plafonne a 1.
        """
        return x / (1.0 + x) if self.dopa_saturate else x

    def swath_width(self) -> float:
        """
        Largeur utile de fauchee : w = f 2 R sin(demi-angle).

        C'est la portee de la repulsion entre empreintes, donc l'espacement
        d'equilibre en ligne de front. Le facteur f < 1 tient compte du fait que
        lambda s'annule au bord du cone : la largeur reellement exploitable est
        inferieure a la largeur geometrique.
        """
        half = math.radians(min(90.0, max(0.0, self.sensor_half_angle)))
        return max(1e-6, self.swath_frac * 2.0 * self.sensor_range * math.sin(half))


# (libelle, attribut, min, max, pas)
TUNABLES = [
    ("Patrouille", "squad_size", 1, 12, 1),
    ("T explor. s", "t_explore", 10.0, 400.0, 5.0),
    ("T rally s", "t_rally_max", 10.0, 200.0, 5.0),
    ("T partage s", "t_share", 1.0, 60.0, 1.0),
    ("R comm. m", "comm_radius", 10.0, 400.0, 5.0),
    ("Anti-redond.", "g_swath", 0.0, 6.0, 0.1),
    ("Fauchee f", "swath_frac", 0.10, 2.0, 0.05),
    ("Reservation", "claim_strength", 0.0, 0.99, 0.05),
    ("R reserv. m", "claim_radius", 5.0, 200.0, 5.0),
    ("Eventail", "g_fan", 0.0, 1.0, 0.05),
    ("Ralliement", "g_rally", 0.0, 4.0, 0.1),
    ("Dispersion", "g_spread", 0.0, 6.0, 0.1),
    ("R disper. m", "spread_radius", 10.0, 250.0, 5.0),
    ("Separation", "g_separation", 0.0, 4.0, 0.1),
    ("Alignement", "g_alignment", 0.0, 4.0, 0.1),
    ("Cohesion", "g_cohesion", 0.0, 4.0, 0.05),
    ("Couverture", "g_coverage", 0.0, 4.0, 0.1),
    ("Obstacle", "g_obstacle", 0.0, 6.0, 0.2),
    ("Autorite m/s", "authority", 4.0, 60.0, 2.0),
    ("R percept. m", "perception_radius", 5.0, 200.0, 5.0),
    ("k voisins", "k_neighbors", 1, 30, 1),
    ("R capteur m", "sensor_range", 10.0, 120.0, 2.0),
    ("Demi-angle", "sensor_half_angle", 5.0, 180.0, 5.0),
    ("Gain capt.", "sensor_gain", 0.05, 4.0, 0.05),
    ("Horizon m", "target_horizon", 20.0, 600.0, 10.0),
    ("Depot dopa", "dopa_deposit", 0.0, 60.0, 1.0),
    ("Attir. dopa", "dopa_attract", 0.0, 40.0, 0.25),
    ("Gain dopa", "dopa_gain", 0.0, 40.0, 0.25),
    ("Scan dopa", "dopa_scan_slow", 0.0, 2.0, 0.05),
    ("Tau dopa s", "dopa_tau", 5.0, 400.0, 5.0),
    ("Rep. cortis.", "cort_repel", 0.0, 5.0, 0.1),
    ("V crois. m/s", "v_cruise", 2.0, 25.0, 0.5),
    ("Virage d/s", "omega_max", 15.0, 360.0, 5.0),
    ("Effectif", "n_drones", 1, 90, 1),
]

# Nombre de lignes que le panneau doit afficher : en-tete + reglages + budget.
HUD_HEADER_LINES = 18
HUD_LINES = HUD_HEADER_LINES + len(TUNABLES) + len(ALL_BEHAVIORS) + 2

C_BG = (16, 16, 19)
C_CONF = (60, 190, 150)
C_DOPA = (240, 150, 35)  # orange
C_CORT = (70, 115, 225)  # bleu
C_DRONE = (127, 119, 221)
C_HL = (216, 90, 48)
C_OBST = (52, 52, 60)
C_TARGET = (200, 200, 205)
C_FOUND = (120, 220, 150)
C_TRUCK = (150, 140, 110)
C_RV = (240, 180, 120)
C_HIER = ((150, 90, 90), (120, 80, 100), (95, 75, 105))  # par niveau

# Une teinte par patrouille : c'est le seul moyen de lire a l'oeil si la
# structure en patrouilles tient ou si elle s'est melangee.
C_SQUADS = (
    (127, 119, 221),
    (110, 200, 175),
    (225, 150, 110),
    (200, 130, 190),
    (150, 195, 235),
    (215, 200, 110),
    (130, 220, 140),
    (230, 130, 130),
)


def squad_color(k: int) -> tuple:
    return C_SQUADS[k % len(C_SQUADS)]


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
    """Composantes connexes d'un graphe donne par ses aretes."""
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


def partition_equilibree(n: int, taille: int) -> list[int]:
    """
    Decoupe n drones en patrouilles de taille voisine de `taille`.

    On arrondit le NOMBRE de patrouilles plutot que de decouper en tranches
    fixes : avec 10 drones et une taille de 3, un decoupage par tranches
    laisserait une patrouille orpheline d'un seul appareil, qui ne beneficierait
    d'aucune cohesion et se comporterait comme dans la version precedente.
    """
    n = max(1, int(n))
    taille = max(1, int(taille))
    n_sq = max(1, round(n / float(taille)))
    n_sq = min(n_sq, n)
    base, reste = divmod(n, n_sq)
    return [base + (1 if k < reste else 0) for k in range(n_sq)]


#  Voisinage d'interaction (metrique ou topologique) et fonction d'influence


class Neighborhood:
    """
    Determine, une fois par frame, avec qui chaque drone interagit.

      - "metric" : tous les drones dans un disque de rayon R. Le nombre de
        voisins depend de la densite locale, donc l'essaim se fragmente quand il
        se dilue.
      - "knn"    : les k plus proches, sans contrainte de distance. Le nombre de
        voisins est constant par construction.

    Le filtrage par patrouille n'est PAS fait ici mais dans Drone.decide : le
    voisinage reste une notion purement geometrique, chaque comportement decide
    ensuite quelle partie il consomme.
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
                         d=0 et jamais nulle.
        "quad_compact" : w = (1-d/R)^2, meme ordre mais a support compact.
        """
        if mode == "none":
            return np.ones_like(d)
        if mode == "quad_compact":
            return np.clip(1.0 - d / np.maximum(d0, 1e-6), 0.0, 1.0) ** 2
        return 1.0 / (1.0 + (d / np.maximum(d0, 1e-6)) ** 2)

    def build(self, drones: list[Drone], p: Params, dist: np.ndarray) -> list[list]:
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


#  Le monde : aire de decollage, obstacles, cibles, points de rendez-vous


class World:
    """
    Geometrie de la mission, en metres. Contient ce qui EXISTE reellement :
    l'aire de decollage, les obstacles, les cibles, les points de rendez-vous et
    la trace des cibles trouvees. Les drones n'accedent jamais a `target_mask`
    autrement qu'a travers leur propre observation.

    Note : `bounds` est un pygame.Rect, donc a coordonnees entieres.
    """

    def __init__(self, bounds: pygame.Rect, p: Params, rng: random.Random):
        self.cell = float(p.cell_m)
        self.bounds = bounds
        self.cols = int(bounds.width / self.cell) + 1
        self.rows = int(bounds.height / self.cell) + 1

        # grilles de coordonnees des centres de cellules (reutilisees partout)
        xs = (np.arange(self.cols) + 0.5) * self.cell
        ys = (np.arange(self.rows) + 0.5) * self.cell
        # float32 : meme dtype que les cartes, deux fois moins de trafic memoire
        self.cx = np.repeat(xs[:, None], self.rows, axis=1).astype(np.float32)
        self.cy = np.repeat(ys[None, :], self.cols, axis=0).astype(np.float32)

        # --- aire de decollage (camion + rangees de drones) ---------------- #
        n = int(p.n_drones)
        per_row = max(1, int(p.launch_per_row))
        n_rows = (n + per_row - 1) // per_row
        self.truck = (
            p.launch_margin,
            bounds.height / 2.0 - p.truck_wid / 2.0,
            p.truck_len,
            p.truck_wid,
        )
        pad_x0 = p.launch_margin + p.truck_len + p.launch_spacing
        pad_w = max(p.launch_spacing, (n_rows - 1) * p.launch_spacing)
        pad_h = max(p.launch_spacing, (per_row - 1) * p.launch_spacing)
        self.pad = (
            pad_x0 - p.launch_spacing * 0.5,
            bounds.height / 2.0 - pad_h / 2.0 - p.launch_spacing * 0.5,
            pad_w + p.launch_spacing,
            pad_h + p.launch_spacing,
        )
        self._pad_x0 = pad_x0

        # --- obstacles ------------------------------------------------------ #
        self.obstacles: list[tuple[float, float, float]] = []
        self._make_obstacles(p, rng)

        # cellules recouvertes par un obstacle : ni observables, ni ciblables
        self.blocked = np.zeros((self.cols, self.rows), dtype=bool)
        for ox, oy, orad in self.obstacles:
            self.blocked |= ((self.cx - ox) ** 2 + (self.cy - oy) ** 2) <= orad * orad

        # aire de decollage : exclue du tirage des cibles (detection triviale)
        px0, py0, pw, ph = self.pad
        self.no_target = (
            (self.cx >= px0 - p.launch_clearance)
            & (self.cx <= px0 + pw + p.launch_clearance)
            & (self.cy >= py0 - p.launch_clearance)
            & (self.cy <= py0 + ph + p.launch_clearance)
        )

        # --- points de rendez-vous ------------------------------------------- #
        # Calcules a partir de la seule geometrie de la zone, donc identiques
        # pour tous les drones sans le moindre echange. Parcours en
        # boustrophedon : les rendez-vous successifs balaient la zone.
        self.rv_points: list[Vector2] = self._make_rv_points(p)

        # --- cibles ---------------------------------------------------------- #
        self.target_mask = np.zeros((self.cols, self.rows), dtype=bool)
        self.found = np.zeros((self.cols, self.rows), dtype=bool)
        self.hier_root = None  # racine de la formation, pour l'affichage
        self.hier_edges: list = []
        self._make_targets(p, rng)
        self.n_targets = int(self.target_mask.sum())

    # -- aire de decollage ---------------------------------------------------- #

    def launch_pose(self, index: int, p: Params) -> tuple[Vector2, float]:
        """
        Position et cap du drone `index` sur l'aire, range en rangees.

        Les rangees sont dans l'axe de la zone : avec `launch_per_row` egal a la
        taille d'une patrouille, une rangee = une patrouille, et la formation en
        ligne de front est deja en place au decollage.
        """
        per_row = max(1, int(p.launch_per_row))
        row, col = divmod(index, per_row)
        y0 = self.bounds.height / 2.0
        x = self._pad_x0 + row * p.launch_spacing
        y = y0 + (col - (per_row - 1) / 2.0) * p.launch_spacing
        y = max(2.0, min(self.bounds.height - 2.0, y))
        x = max(2.0, min(self.bounds.width - 2.0, x))
        return Vector2(x, y), 0.0  # cap vers +x, c'est-a-dire vers la zone

    def base_point(self) -> Vector2:
        ax, ay, aw, ah = self.pad
        return Vector2(ax + aw / 2.0, ay + ah / 2.0)

    @staticmethod
    def _rect_dist(cx: float, cy: float, rect: tuple) -> float:
        rx, ry, rw, rh = rect
        dx = max(rx - cx, 0.0, cx - (rx + rw))
        dy = max(ry - cy, 0.0, cy - (ry + rh))
        return math.hypot(dx, dy)

    # -- generation ------------------------------------------------------------ #

    def _make_rv_points(self, p: Params) -> list[Vector2]:
        b = self.bounds
        nx, ny = max(1, int(p.rv_cols)), max(1, int(p.rv_rows))
        pts: list[Vector2] = []
        for j in range(ny):
            cols = range(nx) if j % 2 == 0 else reversed(range(nx))
            for i in cols:
                x = (i + 0.5) / nx * b.width
                y = (j + 0.5) / ny * b.height
                pt = Vector2(x, y)
                hit = self.inside_obstacle(pt)
                if hit is not None:
                    ox, oy, orad = hit
                    off = pt - Vector2(ox, oy)
                    if off.length_squared() < 1e-9:
                        off = Vector2(1, 0)
                    pt = Vector2(ox, oy) + set_mag(off, orad + p.obstacle_margin + 5.0)
                    pt.x = max(15.0, min(b.width - 15.0, pt.x))
                    pt.y = max(15.0, min(b.height - 15.0, pt.y))
                pts.append(pt)
        return pts

    def _make_obstacles(self, p: Params, rng: random.Random) -> None:
        b = self.bounds
        n_obs = rng.randint(int(p.n_obstacles_min), int(p.n_obstacles_max))
        tries = 0
        while len(self.obstacles) < n_obs and tries < 800:
            tries += 1
            r = rng.uniform(p.obstacle_r_min, p.obstacle_r_max)
            x = rng.uniform(r + 6.0, b.width - r - 6.0)
            y = rng.uniform(r + 6.0, b.height - r - 6.0)
            if self._rect_dist(x, y, self.pad) < r + p.launch_clearance:
                continue
            ok = True
            for ox, oy, orad in self.obstacles:
                if math.hypot(x - ox, y - oy) < r + orad + p.obstacle_gap:
                    ok = False
                    break
            if ok:
                self.obstacles.append((x, y, r))

    def _free_cells(self) -> np.ndarray:
        """Indices des cellules ou une cible peut etre posee."""
        return np.argwhere(~self.blocked & ~self.no_target)

    def _cell_ok(self, x: float, y: float) -> tuple[int, int] | None:
        """Cellule correspondant a (x, y) si elle peut porter une cible."""
        i, j = int(x // self.cell), int(y // self.cell)
        if not (0 <= i < self.cols and 0 <= j < self.rows):
            return None
        if self.blocked[i, j] or self.no_target[i, j]:
            return None
        return i, j

    def _make_targets(self, p: Params, rng: random.Random) -> None:
        """
        Tirage hierarchique. On genere la formation en repere local, puis on
        cherche une rotation et une translation qui la font tenir dans la zone
        sans qu'un trop grand nombre de soldats tombe dans un obstacle ou sur
        l'aire de decollage. On garde le meilleur essai.
        """
        free = self._free_cells()
        if free.size == 0:
            return  # zone entierement bloquee : aucune cible posable

        nprng = np.random.default_rng(rng.randrange(2**31))
        b = self.bounds
        m = float(p.hier_margin)

        best_cells, best_root, best_score = [], None, -1
        for _ in range(max(1, int(p.hier_tries))):
            root = generer_structure(
                HIER_CONFIG,
                centre=(0.0, 0.0),
                theta0=0.0,
                R_terminal=HIER_R_TERMINAL,
                rng=nprng,
            )
            pts = positions_terminales(root)
            if pts.size == 0:
                continue
            # centrage sur le barycentre des soldats, puis rotation aleatoire
            pts = pts - pts.mean(axis=0)
            ang = nprng.uniform(0.0, 2.0 * math.pi)
            ca, sa = math.cos(ang), math.sin(ang)
            rot = np.array([[ca, -sa], [sa, ca]])
            q = pts @ rot.T
            lo, hi = q.min(axis=0), q.max(axis=0)
            span = hi - lo
            if span[0] > b.width - 2 * m or span[1] > b.height - 2 * m:
                continue  # empreinte trop grande pour la zone
            ox = nprng.uniform(m - lo[0], b.width - m - hi[0])
            oy = nprng.uniform(m - lo[1], b.height - m - hi[1])
            q = q + np.array([ox, oy])

            cells = []
            for x, y in q:
                c = self._cell_ok(float(x), float(y))
                if c is not None:
                    cells.append(c)
            score = len(set(cells))
            if score > best_score:
                # la formation affichee doit correspondre aux cibles retenues
                transformer(root, (0.0, 0.0), np.array([1.0, 0.0]))
                shift = positions_terminales(root).mean(axis=0)
                for nd in parcourir(root):
                    nd.centre = rot @ (nd.centre - shift) + np.array([ox, oy])
                    nd.direction = rot @ nd.direction
                best_cells, best_root, best_score = cells, root, score
            if score == len(q):
                break  # placement parfait, inutile de chercher mieux

        for c in best_cells:
            self.target_mask[c] = True
        self.hier_root = best_root
        self.hier_edges = aretes_hierarchie(best_root) if best_root else []

        # Eclaireurs detaches : empechent l'essaim de se contenter d'exploiter
        # la formation principale une fois qu'il l'a trouvee.
        for _ in range(int(p.n_isolated)):
            k = rng.randrange(len(free))
            self.target_mask[int(free[k, 0]), int(free[k, 1])] = True

    # -- requetes geometriques -------------------------------------------------- #

    def inside_obstacle(self, pos: Vector2) -> tuple[float, float, float] | None:
        for ox, oy, orad in self.obstacles:
            if (pos.x - ox) ** 2 + (pos.y - oy) ** 2 < orad * orad:
                return (ox, oy, orad)
        return None


#  Le drone


class Drone:
    """
    Agent autonome. Etat interne : position (m), cap (rad), module de vitesse.

    On raisonne en vitesse et non en acceleration : chaque comportement exprime
    une correction de vitesse dv (m/s), l'arbitre en retient une combinaison, et
    la cinematique saturee du drone dit ce qui est reellement realisable dans le
    pas de temps. C'est ce qu'expose l'autopilote d'un multirotor.

    La vitesse n'est pas constante : profil d'arrivee sur la cible,
    ralentissement en virage (cosinus de l'erreur de cap), et ralentissement de
    scan en zone dopaminee.

    `squad` : indice de patrouille. C'est la seule donnee sociale portee par le
    drone, et elle est deterministe (donc connue avant le decollage).
    """

    __slots__ = (
        "pos",
        "heading",
        "speed",
        "vel",
        "params",
        "target",
        "next_eval",
        "scan_factor",
        "fan_dir",
        "index",
        "squad",
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
        self.scan_factor = 1.0
        self.fan_dir = 0.0
        self.index = index
        self.squad = 0
        self.alloc: dict[str, float] = {}
        self.color = C_DRONE

    # -- geometrie du capteur ---------------------------------------------------- #

    def footprint(self) -> Vector2:
        """
        Centre de l'empreinte capteur : le milieu de l'axe du cone.

        C'est CE point, et non la position du drone, qui sert de reference a
        l'anti-redondance. Deux drones cote a cote avec des caps opposes ne se
        genent pas ; deux drones eloignes mais convergents, si.
        """
        r = 0.5 * self.params.sensor_range
        return self.pos + Vector2(math.cos(self.heading), math.sin(self.heading)) * r

    # -- comportements : chacun renvoie une correction de vitesse -------------- #

    def _w_separation(self, nb, g: float) -> Vector2:
        """Securite a courte portee. S'applique a TOUS, patrouille ou non."""
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
        return set_mag(push, p.v_max * urgency) * g

    def _w_swath(self, nb, g: float) -> Vector2:
        """
        ANTI-REDONDANCE. Repulsion entre CENTRES D'EMPREINTE capteur, de portee
        egale a la largeur de fauchee.

        C'est le comportement qui remplace la dispersion aveugle de la version
        precedente. Ce que l'on veut eviter n'est pas la proximite mais le
        recouvrement des cones : deux drones qui se suivent en file ratissent la
        meme bande et l'un des deux ne sert a rien.

        Combine a l'alignement, il produit la ligne de front : les caps
        convergeant, la composante utile de la repulsion devient laterale, et
        l'equilibre avec la cohesion fixe l'espacement a la fauchee.
        """
        p = self.params
        if g <= 0.0 or not nb:
            return Vector2(0, 0)
        w_sw = p.swath_width()
        ci = self.footprint()
        push = Vector2(0, 0)
        for other, _d, _w in nb:
            off = ci - other.footprint()
            dd = off.length()
            if dd >= w_sw or dd < 1e-6:
                continue
            push += off / dd * (1.0 - dd / w_sw)
        # Intensite GRADUEE. set_mag() seul normaliserait le vecteur et
        # rendrait la repulsion constante quelle que soit la profondeur du
        # recouvrement : il n'y aurait alors aucun equilibre avec la cohesion,
        # seulement une bascule entre deux regimes. On conserve donc la norme
        # accumulee (somme des 1 - d/w) comme mesure d'intensite.
        inten = min(1.0, push.length())
        if inten < 1e-6:
            return Vector2(0, 0)
        return set_mag(push, p.v_cruise * inten) * g

    def _w_spread(self, nb_out, g: float) -> Vector2:
        """
        Dispersion longue portee, desormais INTER-PATROUILLES uniquement.

        Elle empeche deux patrouilles de ratisser le meme secteur quand elles
        sont hors de portee radio (donc quand les reservations d'intention ne
        peuvent pas jouer). Appliquee a l'interieur d'une patrouille, elle la
        ferait exploser : c'est ce qui detruisait le flocking.

        Le rayon de reference est `spread_radius` en mode metrique, mais la
        distance du voisin le plus eloigne en mode topologique : sinon, avec un
        essaim dilue, aucun des k voisins ne passerait le filtre.
        """
        p = self.params
        if g <= 0.0 or not nb_out:
            return Vector2(0, 0)
        if p.neighborhood == "knn":
            R = max(d for _o, d, _w in nb_out) * 1.001
        else:
            R = p.spread_radius
        if R <= 1e-6:
            return Vector2(0, 0)
        push = Vector2(0, 0)
        for other, d, _w in nb_out:
            if d >= R or d < 1e-6:
                continue
            push += (self.pos - other.pos) / d * (1.0 - d / R)
        inten = min(1.0, push.length())
        if inten < 1e-6:
            return Vector2(0, 0)
        return set_mag(push, p.v_cruise * inten) * g

    def _w_obstacle(self, world: World, g: float) -> Vector2:
        p = self.params
        push = Vector2(0, 0)
        urgency = 0.0
        for ox, oy, orad in world.obstacles:
            off = self.pos - Vector2(ox, oy)
            d = off.length()
            if d >= orad + p.obstacle_margin:
                continue
            u = 1.0 if d <= orad else 1.0 - (d - orad) / max(1e-6, p.obstacle_margin)
            if d < 1e-6:
                off, d = Vector2(1, 0), 1.0
            push += off / d * u
            urgency = max(urgency, u)
        if urgency <= 0.0:
            return Vector2(0, 0)
        return set_mag(push, p.v_max * urgency) * g

    def _w_alignment(self, nb_in, g: float) -> Vector2:
        """Alignement sur la PATROUILLE seulement."""
        acc = Vector2(0, 0)
        tot = 0.0
        for other, _d, w in nb_in:
            acc += other.vel * w
            tot += w
        if tot <= 1e-9 or g <= 0.0:
            return Vector2(0, 0)
        return (acc / tot - self.vel) * g

    def _w_cohesion(self, nb_in, g: float) -> Vector2:
        """Cohesion sur la PATROUILLE seulement."""
        p = self.params
        if g <= 0.0:
            return Vector2(0, 0)
        acc = Vector2(0, 0)
        tot = 0.0
        for other, _d, w in nb_in:
            acc += other.pos * w
            tot += w
        if tot <= 1e-9:
            return Vector2(0, 0)
        offset = acc / tot - self.pos
        dist = offset.length()
        if dist < 1e-6:
            return Vector2(0, 0)
        slow = min(1.0, dist / max(1e-6, p.separation_radius * 2.0))
        return (set_mag(offset, p.v_cruise * slow) - self.vel) * g

    def _w_boundary(self, bounds: pygame.Rect, g: float) -> Vector2:
        p = self.params
        if p.boundary_mode != "steer" or g <= 0.0:
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
        return (set_mag(push, p.v_max) - self.vel) * (g * min(1.0, depth))

    def _w_rally(self, rally_pt: Vector2 | None, g: float) -> Vector2:
        """Rejoindre le point de rendez-vous, avec profil d'arrivee."""
        p = self.params
        if rally_pt is None or g <= 0.0:
            return Vector2(0, 0)
        off = rally_pt - self.pos
        dist = off.length()
        if dist < 1e-6:
            return Vector2(0, 0)
        v_want = p.v_cruise * min(1.0, dist / max(1e-6, p.rally_arrival))
        return (set_mag(off, v_want) - self.vel) * g

    def _w_coverage(self, g: float) -> Vector2:
        """
        La cible est choisie par le simulateur a partir de la carte du drone.

        Les membres d'une patrouille connectee portent la MEME carte, donc la
        meme cible : la couverture agit alors comme une cohesion implicite, et
        c'est l'anti-redondance qui les empeche de se superposer.
        """
        p = self.params
        if g <= 0.0 or self.target is None:
            return Vector2(0, 0)
        offset = self.target - self.pos
        dist = offset.length()
        if dist < p.capture_radius:
            self.target = None
            return Vector2(0, 0)
        # profil d'arrivee + ralentissement de scan
        v_want = p.v_cruise * min(1.0, dist / max(1e-6, p.arrival_radius))
        v_want *= self.scan_factor
        return (set_mag(offset, v_want) - self.vel) * g

    # -- arbitrage, module par la phase ------------------------------------------ #

    def decide(
        self,
        nb,
        world: World,
        bounds: pygame.Rect,
        profile: PhaseProfile,
        rally_pt: Vector2 | None,
    ) -> Vector2:
        """
        Scission du voisinage : c'est ici que la patrouille devient l'unite de
        manoeuvre. La securite et l'anti-redondance ne connaissent pas les
        patrouilles (un drone d'une autre patrouille peut tout aussi bien vous
        percuter ou ratisser votre bande) ; la cohesion et l'alignement, si.
        """
        p = self.params
        mul = profile.mul

        nb_in = [t for t in nb if t[0].squad == self.squad]
        nb_out = [t for t in nb if t[0].squad != self.squad]

        want = {
            "separation": self._w_separation(nb, p.g_separation * mul["separation"]),
            "obstacle": self._w_obstacle(world, p.g_obstacle * mul["obstacle"]),
            "boundary": self._w_boundary(bounds, p.g_boundary * mul["boundary"]),
            "rally": self._w_rally(rally_pt, p.g_rally * mul["rally"]),
            "coverage": self._w_coverage(p.g_coverage * mul["coverage"]),
            "swath": self._w_swath(nb, p.g_swath * mul["swath"]),
            "spread": self._w_spread(nb_out, p.g_spread * mul["spread"]),
            "alignment": self._w_alignment(nb_in, p.g_alignment * mul["alignment"]),
            "cohesion": self._w_cohesion(nb_in, p.g_cohesion * mul["cohesion"]),
        }

        if p.arbitration == "weighted":
            # Reference : somme ponderee. Tout s'exprime, la saturation finale
            # ecrete tout de la meme facon, donc une exigence de securite peut
            # etre diluee par plusieurs exigences de confort.
            total = Vector2(0, 0)
            for name in ALL_BEHAVIORS:
                total += want[name]
            self.alloc = {k: want[k].length() for k in ALL_BEHAVIORS}
            return limit(self.vel + limit(total, p.authority), p.v_max)

        # Allocation prioritaire. L'ordre depend de la phase : c'est la que se
        # joue l'essentiel de l'adaptation.
        total = Vector2(0, 0)
        remaining = p.authority
        alloc = {k: 0.0 for k in ALL_BEHAVIORS}
        for name in profile.order:
            dv = want[name]
            m = dv.length()
            if remaining <= 1e-6 or m <= 1e-9:
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
        turn_factor = 1.0
        if v_des.length_squared() > 1e-9:
            err = wrap_pi(math.atan2(v_des.y, v_des.x) - self.heading)
            lim = math.radians(p.omega_max) * dt
            self.heading = wrap_pi(self.heading + max(-lim, min(lim, err)))
            # un virage serre coute de la vitesse
            turn_factor = max(p.turn_slow_min, math.cos(err))

        sp_des = min(p.v_max, max(p.v_min, v_des.length() * turn_factor))
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
        self.pos = Vector2(ox, oy) + set_mag(off, orad + 0.5)

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

    def draw(self, screen: pygame.Surface, show_cone: bool, show_swath: bool) -> None:
        p = self.params
        cx, cy = px(self.pos.x), px(self.pos.y)
        if show_cone:
            half = math.radians(p.sensor_half_angle)
            pts = [(cx, cy)]
            steps = 12
            for s in range(steps + 1):
                a = self.heading - half + 2 * half * s / steps
                pts.append(
                    (
                        cx + px(p.sensor_range * math.cos(a)),
                        cy + px(p.sensor_range * math.sin(a)),
                    )
                )
            pygame.draw.polygon(screen, (40, 110, 95), pts, 1)
        if show_swath:
            # disque d'exclusion de l'anti-redondance, centre sur l'empreinte :
            # deux disques qui se recoupent = deux cones redondants.
            f = self.footprint()
            pygame.draw.circle(
                screen,
                (70, 60, 45),
                (px(f.x), px(f.y)),
                max(1, px(0.5 * p.swath_width())),
                1,
            )
            pygame.draw.circle(screen, (110, 95, 70), (px(f.x), px(f.y)), 2)
        # taille fixe (~9 px) et non a l'echelle : un multirotor de 0,5 m ferait
        # 2 px et serait illisible.
        ca, sa = math.cos(self.heading), math.sin(self.heading)
        tri = [
            (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)
            for lx, ly in ((9, 0), (-5, 4), (-2.5, 0), (-5, -4))
        ]
        pygame.draw.polygon(screen, self.color, tri)


#  Le simulateur


class Swarm:
    """
    Porte les cartes de connaissance, la partition en patrouilles et la machine
    a etats des phases.

    Ordre d'un pas :
      1. voisinage d'interaction
      2. graphe de communication -> composantes connexes (transitivite)
      3. fusion par composante (maximum, idempotent)
      4. machine a etats des phases
      5. choix de cible (avec reservations d'intention), arbitrage module par la
         phase, cinematique
      6. observation dans le cone, sommee par composante
      7. depots de pheromones
      8. diffusion + evaporation
      9. verite terrain et test de fin

    La fusion precede l'observation et le ciblage : un drone qui rejoint un
    groupe doit profiter de sa connaissance avant de choisir ou regarder.
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
        self.alloc_mean = {k: 0.0 for k in ALL_BEHAVIORS}
        self.last_nbs: list[list] = []
        self.comm_pairs: list[tuple[int, int]] = []
        self.n_components = 1
        # --- patrouilles ---
        self.squads: list[list[int]] = []
        self.n_squads = 1
        self._squad_sig: tuple = ()
        # indicateurs mis a jour periodiquement
        self.mean_speed = 0.0
        self.conf_col = 0.0
        self.conf_solo = 0.0
        self.conf_gap = 0.0
        self.dopa_max = 0.0
        self.d_squad = 0.0  # distance moyenne au plus proche coequipier
        self.squad_ratio = 0.0  # la meme, rapportee a la fauchee
        self.overlap = 0.0  # redondance moyenne des empreintes
        self.squad_split = 0  # patrouilles coupees en deux (radio)
        # --- etat des phases ---
        self.phase = "explore"
        self.phase_t = 0.0
        self.cycle = 0
        self.rally_pt: Vector2 | None = None
        self.shares_ok = 0
        self.shares_failed = 0
        self.gap_before = 0.0
        self.gap_after = 0.0
        self.conf = np.zeros((0, 0, 0), dtype=np.float32)
        self.dopa = np.zeros((0, 0, 0), dtype=np.float32)
        self.cort = np.zeros((0, 0, 0), dtype=np.float32)
        self.known_t = np.zeros((0, 0, 0), dtype=bool)
        self.known_s = np.zeros((0, 0, 0), dtype=bool)
        self._lam = np.zeros((0, 0, 0), dtype=np.float32)
        self._pher_dt = 0.0
        self.reset()

    # -- initialisation --------------------------------------------------------- #

    def reset(self) -> None:
        self.params.sanitize()
        self.rng = random.Random(self.seed)
        self.drones.clear()
        self.time = 0.0
        self.frame = 0
        self.done = False
        self.completion_time = None
        self.phase = "explore"
        self.phase_t = 0.0
        self.cycle = 0
        self.rally_pt = None
        self.shares_ok = 0
        self.shares_failed = 0
        self.gap_before = 0.0
        self.gap_after = 0.0
        self._pher_dt = 0.0
        self._squad_sig = ()
        self.world = World(self.bounds, self.params, self.rng)
        self._alloc_layers(int(self.params.n_drones))
        for i in range(int(self.params.n_drones)):
            self._spawn(i)
        self._assign_squads()

    def _alloc_layers(self, n: int) -> None:
        w = self.world
        shape = (n, w.cols, w.rows)
        self.conf = np.zeros(shape, dtype=np.float32)
        self.dopa = np.zeros(shape, dtype=np.float32)
        self.cort = np.zeros(shape, dtype=np.float32)
        self.known_t = np.zeros(shape, dtype=bool)
        self.known_s = np.zeros(shape, dtype=bool)
        self._lam = np.zeros(shape, dtype=np.float32)

    def _spawn(self, index: int) -> None:
        """Deploiement depuis le camion : rangees sur l'aire de decollage."""
        pos, heading = self.world.launch_pose(index, self.params)
        self.drones.append(Drone(pos, heading, 0.0, self.params, index))

    # -- patrouilles ------------------------------------------------------------- #

    def _assign_squads(self) -> None:
        """
        Partition DETERMINISTE de l'essaim en patrouilles.

        Aucun tirage aleatoire, aucune negociation : la partition se deduit du
        seul couple (effectif, taille de patrouille), tous deux fixes avant le
        decollage. C'est ce qui la rend legitime au regard de la contrainte
        "deductible sans echange" que respectent deja les points de rendez-vous.
        """
        n = len(self.drones)
        if n == 0:
            self.squads, self.n_squads = [], 1
            return
        sig = (n, int(self.params.squad_size))
        if sig == self._squad_sig:
            return
        self._squad_sig = sig

        tailles = partition_equilibree(n, self.params.squad_size)
        self.squads = []
        i = 0
        for k, t in enumerate(tailles):
            membres = list(range(i, i + t))
            for m in membres:
                self.drones[m].squad = k
                self.drones[m].color = squad_color(k)
            self.squads.append(membres)
            i += t
        self.n_squads = len(self.squads)
        self._assign_fan_sectors()

    def _sync_population(self) -> None:
        n = int(self.params.n_drones)
        cur = len(self.drones)
        if n != cur:
            if n > cur:
                pad = ((0, n - cur), (0, 0), (0, 0))
                self.conf = np.pad(self.conf, pad)
                self.dopa = np.pad(self.dopa, pad)
                self.cort = np.pad(self.cort, pad)
                self.known_t = np.pad(self.known_t, pad)
                self.known_s = np.pad(self.known_s, pad)
                while len(self.drones) < n:
                    self._spawn(len(self.drones))  # renfort depuis le camion
            else:
                self.conf = self.conf[:n]
                self.dopa = self.dopa[:n]
                self.cort = self.cort[:n]
                self.known_t = self.known_t[:n]
                self.known_s = self.known_s[:n]
                del self.drones[n:]
            w = self.world
            self._lam = np.zeros((n, w.cols, w.rows), dtype=np.float32)
        # la taille de patrouille est reglable en vol : on repasse toujours
        self._assign_squads()

    def _assign_fan_sectors(self) -> None:
        """
        CONDITION TEMOIN. Secteur angulaire prefere, attribue PAR PATROUILLE.

        C'est le mecanisme de la version precedente, conserve pour comparaison
        mais desactive par defaut. Il est ici purement deterministe (aucun
        tirage), ce qui evite qu'un simple changement de partition ne decale le
        flux aleatoire et ne casse la reproductibilite a graine fixee.
        """
        n_sq = max(1, self.n_squads)
        for d in self.drones:
            d.fan_dir = 2.0 * math.pi * d.squad / n_sq

    # -- phases ------------------------------------------------------------------- #

    def _compute_rally_point(self) -> Vector2:
        """
        Le point doit etre deductible SANS communication, puisque les
        patrouilles sont justement deconnectees entre elles au moment ou la
        phase s'enclenche.
        """
        p, w = self.params, self.world
        if p.rally_mode == "truck":
            return w.base_point()
        if p.rally_mode == "centroid":
            # ORACLE : suppose que chaque drone connait la position de tous les
            # autres. Borne de comparaison uniquement.
            if not self.drones:
                return w.base_point()
            sx = sum(d.pos.x for d in self.drones) / len(self.drones)
            sy = sum(d.pos.y for d in self.drones) / len(self.drones)
            pt = Vector2(sx, sy)
            hit = w.inside_obstacle(pt)
            if hit is not None:
                ox, oy, orad = hit
                off = pt - Vector2(ox, oy)
                if off.length_squared() < 1e-9:
                    off = Vector2(1, 0)
                pt = Vector2(ox, oy) + set_mag(off, orad + p.obstacle_margin + 4.0)
            return pt
        # "schedule" : suite fixee avant le decollage, donc connue de tous.
        if not w.rv_points:
            return w.base_point()
        return w.rv_points[self.cycle % len(w.rv_points)]

    def _knowledge_gap(self) -> float:
        """
        Ecart entre ce que sait l'essaim et ce que sait un drone en moyenne.
        Il croit pendant l'exploration et retombe d'un coup au partage.
        """
        if not self.conf.size:
            return 0.0
        return max(0.0, float(self.conf.max(axis=0).mean()) - float(self.conf.mean()))

    def _advance_phase(self) -> None:
        nxt = PHASE_CYCLE[self.phase]
        if self.phase == "explore":
            self.gap_before = self._knowledge_gap()
            self.rally_pt = self._compute_rally_point()
        elif self.phase == "rally":
            if self.n_components == 1 and len(self.drones) > 1:
                self.shares_ok += 1
            else:
                self.shares_failed += 1
        elif self.phase == "share":
            self.gap_after = self._knowledge_gap()
            self.cycle += 1
            self.rally_pt = None
        self.phase = nxt
        self.phase_t = 0.0

    def _update_phase(self, dt: float) -> None:
        p = self.params
        self.phase_t += dt
        if self.phase == "explore":
            if self.phase_t >= p.t_explore:
                self._advance_phase()
        elif self.phase == "rally":
            # on passe au partage des que l'essaim ne forme plus qu'une seule
            # composante connexe : la transitivite fait le reste.
            if self.n_components <= 1 or self.phase_t >= p.t_rally_max:
                self._advance_phase()
        elif self.phase == "share":
            if self.phase_t >= p.t_share:
                self._advance_phase()

    def fan_weight(self) -> float:
        """Amplitude du biais de secteur : nulle si la condition temoin est off."""
        p = self.params
        if not p.fan_enabled:
            return 0.0
        if self.phase != "explore" or p.g_fan <= 0.0 or p.fan_tau <= 1e-6:
            return 0.0
        return p.g_fan * math.exp(-self.phase_t / p.fan_tau)

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
        l'information soit recomptee. Il est aussi commutatif et associatif.
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

        lambda decroit avec la distance et avec l'ecart angulaire a l'axe,
        s'annule au bord du cone, et est multiplie par
        (1 + gain * reponse(dopamine)).
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

        DX = w.cx[i0:i1, j0:j1] - d.pos.x
        DY = w.cy[i0:i1, j0:j1] - d.pos.y
        dist = np.sqrt(DX * DX + DY * DY)
        safe = np.maximum(dist, 1e-6)

        hx, hy = math.cos(d.heading), math.sin(d.heading)
        cosang = (DX * hx + DY * hy) / safe
        cos_half = math.cos(math.radians(p.sensor_half_angle))

        # la cellule sous le drone est toujours observee (visee nadir)
        mask = (dist <= R) & ((cosang >= cos_half) | (dist <= cell))
        mask &= ~w.blocked[i0:i1, j0:j1]

        # occultation : un obstacle sur le segment drone -> cellule coupe la vue.
        # Un obstacle entierement au-dela de la portee ne peut masquer aucune
        # cellule visible : on l'ecarte avant les calculs sur tableau.
        for ox, oy, orad in w.obstacles:
            wx, wy = ox - d.pos.x, oy - d.pos.y
            if math.hypot(wx, wy) - orad > R:
                continue
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
        lam = lam * (1.0 + p.dopa_gain * p.dopa_response(dopa[i0:i1, j0:j1]))
        return i0, i1, j0, j1, np.where(mask, lam, 0.0).astype(np.float32)

    # -- ciblage ---------------------------------------------------------------------- #

    def _claims_for(self, comp: int, squad: int, comp_of: np.ndarray) -> list[Vector2]:
        """
        Reservations d'intention visibles par une patrouille donnee.

        Trois filtres, et ils portent tout le mecanisme :
          - MEME COMPOSANTE de communication : une intention ne se propage que
            par la radio. Deux patrouilles hors de portee peuvent donc reserver
            la meme cellule, ce qui est l'inefficacite honnete que le modele
            doit representer, et non un bug a corriger.
          - AUTRE PATROUILLE : a l'interieur d'une patrouille on VEUT la cible
            commune, c'est elle qui fait tenir la formation. La reservation ne
            sert qu'a differencier les patrouilles entre elles.
          - DEDUPLICATION : les membres d'une meme patrouille annoncent tous la
            meme cible. Sans ce filtre la penalite serait appliquee autant de
            fois qu'il y a de membres, soit (1 - k)^4 au lieu de (1 - k), ce qui
            creuserait un trou infranchissable au lieu d'une simple dissuasion.

        Le desordre necessaire est deja la : `next_eval` desynchronise les
        reevaluations d'une patrouille a l'autre, donc la premiere qui evalue
        reserve et les suivantes voient la reservation. Aucun tirage aleatoire,
        aucun angle impose.
        """
        uniq: list[Vector2] = []
        for j, dj in enumerate(self.drones):
            if dj.target is None or comp_of[j] != comp or dj.squad == squad:
                continue
            if any((dj.target - u).length_squared() < 1e-6 for u in uniq):
                continue
            uniq.append(dj.target)
        return uniq

    def _targeting(self, comp_of: np.ndarray, fan_w: float) -> None:
        """
        UNE decision de cible PAR PATROUILLE CONNECTEE, et non par drone.

        Point important : meme avec des cartes rigoureusement identiques, deux
        drones ne choisissaient pas la meme cellule, parce que le score depend
        aussi de leur position et de leur cap (termes `dist` et `bias`). La
        patrouille se serait donc effilochee malgre la fusion. On evalue ici le
        score depuis une POSE DE REFERENCE commune -- barycentre et cap moyen
        des membres joignables -- ce qui ne suppose que l'echange de quelques
        octets entre appareils deja connectes.

        Effet de bord appreciable : le balayage de la carte entiere, qui est le
        deuxieme poste de cout du simulateur, n'est plus fait qu'une fois par
        patrouille au lieu d'une fois par drone.

        Une patrouille coupee en deux par la radio forme deux groupes, qui
        divergent : c'est le mode de defaillance que l'on veut pouvoir observer.
        """
        p = self.params
        par_groupe: dict[tuple, list[int]] = {}
        for i, d in enumerate(self.drones):
            par_groupe.setdefault((int(comp_of[i]), int(d.squad)), []).append(i)

        for (comp, squad), membres in par_groupe.items():
            besoin = any(
                self.frame >= self.drones[m].next_eval or self.drones[m].target is None
                for m in membres
            )
            if not besoin:
                continue

            k = float(len(membres))
            cx = sum(self.drones[m].pos.x for m in membres) / k
            cy = sum(self.drones[m].pos.y for m in membres) / k
            hx = sum(math.cos(self.drones[m].heading) for m in membres)
            hy = sum(math.sin(self.drones[m].heading) for m in membres)
            href = (
                math.atan2(hy, hx)
                if (hx * hx + hy * hy) > 1e-9
                else self.drones[membres[0]].heading
            )

            rep = membres[0]  # cartes identiques dans une composante
            cible = self._choose_target(
                Vector2(cx, cy),
                href,
                self.drones[rep].fan_dir,
                self.conf[rep],
                self.dopa[rep],
                self.cort[rep],
                fan_w,
                self._claims_for(comp, squad, comp_of),
            )
            suivant = self.frame + max(1, int(p.coverage_every))
            for m in membres:
                d = self.drones[m]
                d.target = cible
                d.next_eval = suivant
                # ralentissement de scan : plus la zone est dopaminee, plus le
                # drone leve le pied pour allonger son temps d'observation.
                d.scan_factor = 1.0 / (
                    1.0 + p.dopa_scan_slow * self._local_dopa(m, d.pos)
                )

    def _choose_target(
        self,
        pos: Vector2,
        heading: float,
        fan_dir: float,
        conf,
        dopa,
        cort,
        fan_w: float,
        claims,
    ) -> Vector2 | None:
        """
        Cellule maximisant

            besoin x (1 + k_D.rep(dopa) - k_C.cortisol) x biais_de_cap
            x reservations x eventail x exp(-dist/L) / sqrt(dist)

        evaluee sur TOUTE la carte propre au drone. C'est ce qui permet, apres un
        partage, de repartir vers un foyer de dopamine decouvert par un autre a
        l'autre bout de la zone.

        Le besoin vaut 1 pour une cellule jamais observee et tombe a 0 quand la
        confiance atteint le seuil d'exploration. Une cellule revenue d'un
        partage avec une confiance PARTIELLE garde donc un besoin non nul.

        La pose (pos, heading) est celle de la PATROUILLE, pas d'un drone
        particulier : tous ses membres joignables recoivent donc la meme cible.
        C'est voulu, et ce sont les reservations qui differencient les
        patrouilles entre elles.
        """
        p = self.params
        w = self.world

        need = np.clip(
            (p.explore_threshold - conf) / max(1e-6, p.explore_threshold), 0.0, 1.0
        )
        attract = np.clip(
            1.0 + p.dopa_attract * p.dopa_response(dopa) - p.cort_repel * cort,
            0.0,
            None,
        )

        DX = w.cx - pos.x
        DY = w.cy - pos.y
        dist = np.sqrt(DX * DX + DY * DY) + 1.0
        hx, hy = math.cos(heading), math.sin(heading)
        bias = 0.55 + 0.45 * (DX * hx + DY * hy) / dist

        score = need * attract * bias * np.exp(-dist / max(1.0, p.target_horizon))
        score /= np.sqrt(dist)

        # penalite gaussienne autour des cellules deja reservees par une autre
        # patrouille de ma composante radio
        if claims and p.claim_strength > 1e-6:
            s2 = 2.0 * max(1.0, p.claim_radius) ** 2
            for t in claims:
                r2 = (w.cx - t.x) ** 2 + (w.cy - t.y) ** 2
                score = score * (1.0 - p.claim_strength * np.exp(-r2 / s2))

        if fan_w > 1e-6:
            fx, fy = math.cos(fan_dir), math.sin(fan_dir)
            fan = 1.0 - fan_w + fan_w * (0.5 + 0.5 * (DX * fx + DY * fy) / dist)
            score *= fan

        score = np.where(w.blocked, 0.0, score)

        k = int(np.argmax(score))
        if score.flat[k] <= 1e-12:
            return None
        i, j = np.unravel_index(k, score.shape)
        return Vector2(float(w.cx[i, j]), float(w.cy[i, j]))

    def _local_dopa(self, i_drone: int, pos: Vector2) -> float:
        w = self.world
        i = min(w.cols - 1, max(0, int(pos.x // w.cell)))
        j = min(w.rows - 1, max(0, int(pos.y // w.cell)))
        return float(self.params.dopa_response(self.dopa[i_drone, i, j]))

    # -- pheromones -------------------------------------------------------------------- #

    @staticmethod
    def _diffuse_evaporate(A: np.ndarray, rate: float, tau: float, dt: float) -> None:
        """
        Laplacien a 5 points sur toutes les couches d'un coup, puis evaporation
        exponentielle. Le coefficient de diffusion est borne a 0,24 pour rester
        dans le domaine de stabilite du schema explicite.
        """
        if A.size == 0:
            return
        coef = min(0.24, max(0.0, rate * dt))
        if coef > 0.0:
            P = np.pad(A, ((0, 0), (1, 1), (1, 1)), mode="edge")
            lap = (
                P[:, :-2, 1:-1]
                + P[:, 2:, 1:-1]
                + P[:, 1:-1, :-2]
                + P[:, 1:-1, 2:]
                - 4.0 * A
            )
            A += coef * lap
        if tau > 1e-6:
            A *= math.exp(-dt / tau)

    # -- indicateurs de formation ------------------------------------------------- #

    def _formation_metrics(self, dist: np.ndarray) -> None:
        """
        Deux indicateurs pour piloter le reglage :

          - `squad_ratio` : distance moyenne au plus proche coequipier,
            rapportee a la fauchee. Une ligne de front bien formee le place au
            voisinage de 1. Nettement en dessous : la cohesion l'emporte et les
            cones se recouvrent. Nettement au-dessus : la patrouille se delite,
            et si le rapport depasse comm_radius / fauchee, elle se coupe.
          - `overlap` : nombre moyen d'empreintes qui recouvrent la mienne,
            pondere par la profondeur du recouvrement. C'est exactement ce que
            l'anti-redondance minimise.
        """
        n = len(self.drones)
        p = self.params
        if n < 2:
            self.d_squad = self.squad_ratio = self.overlap = 0.0
            self.squad_split = 0
            return

        sq = np.array([d.squad for d in self.drones])
        same = sq[:, None] == sq[None, :]
        np.fill_diagonal(same, False)
        d_in = np.where(same, dist, np.inf).min(axis=1)
        fin = d_in[np.isfinite(d_in)]
        self.d_squad = float(fin.mean()) if fin.size else 0.0
        self.squad_ratio = self.d_squad / p.swath_width()

        fp = np.array([[f.x, f.y] for f in (d.footprint() for d in self.drones)])
        diff = fp[None, :, :] - fp[:, None, :]
        dfp = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(dfp, np.inf)
        self.overlap = float(
            np.clip(1.0 - dfp / p.swath_width(), 0.0, 1.0).sum(axis=1).mean()
        )

        # patrouilles coupees : membres repartis sur plusieurs composantes radio
        split = 0
        for members in self.squads:
            if len(members) < 2:
                continue
            sub = dist[np.ix_(members, members)] <= p.comm_radius
            rows, cols = np.nonzero(sub)
            _c, g = union_find_groups(len(members), rows, cols)
            if len(g) > 1:
                split += 1
        self.squad_split = split

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

        # 2-3. communication puis fusion (avant le ciblage et l'observation)
        comp_of, groups = self._communication(dist)
        self._fuse(groups)

        # 4. machine a etats des phases
        self._update_phase(dt)
        profile = PHASES[self.phase]
        fan_w = self.fan_weight()

        # 5. ciblage, arbitrage module par la phase, cinematique.
        #    Les reservations sont lues au moment de l'evaluation : un drone qui
        #    reevalue plus tard dans la meme frame voit deja la cible que ses
        #    voisins viennent de choisir.
        self._targeting(comp_of, fan_w)
        desired = [
            d.decide(nbs[i], w, self.bounds, profile, self.rally_pt)
            for i, d in enumerate(self.drones)
        ]
        for d, v in zip(self.drones, desired):
            d.apply(v, dt, self.bounds, w)

        # 6. observation : les contributions de tous les membres d'une composante
        #    sont sommees avant d'etre appliquees a leur carte commune. La
        #    summabilite est donc exacte pour des drones qui volent ensemble --
        #    c'est-a-dire, desormais, pour une patrouille entiere.
        ng = len(groups)
        lam = self._lam[:ng]
        lam.fill(0.0)
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

            # 7. depots. On ne depose qu'a la transition, sinon une grappe deja
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

        # 8. diffusion puis evaporation. Groupees sur pher_every frames : le
        #    coefficient de diffusion cumule reste tres en deca de la limite de
        #    stabilite 0,24, et l'evaporation exponentielle est exacte quel que
        #    soit le pas. Le laplacien sur toute la pile de cartes est le poste
        #    le plus couteux du simulateur.
        self._pher_dt += dt
        if self.frame % max(1, int(p.pher_every)) == 0:
            self._diffuse_evaporate(
                self.dopa, p.dopa_diffuse, p.dopa_tau, self._pher_dt
            )
            self._diffuse_evaporate(
                self.cort, p.cort_diffuse, p.cort_tau, self._pher_dt
            )
            self._pher_dt = 0.0

        # 9. verite terrain (invisible aux drones) et fin de mission
        best = self.conf.max(axis=0)
        w.found |= w.target_mask & (best >= p.detect_threshold)
        if not self.done and int(w.found.sum()) >= w.n_targets:
            # n_targets == 0 compte comme un succes immediat, sans quoi la
            # mission ne pourrait jamais s'achever.
            self.done = True
            self.completion_time = self.time

        if self.frame % 10 == 0:
            self.alloc_mean = {
                k: sum(d.alloc.get(k, 0.0) for d in self.drones) / n
                for k in ALL_BEHAVIORS
            }
            self.mean_speed = sum(d.speed for d in self.drones) / n
            self.conf_col = float(best.mean())
            self.conf_solo = float(self.conf[0].mean())
            self.conf_gap = max(0.0, self.conf_col - float(self.conf.mean()))
            self.dopa_max = float(self.dopa.max()) if self.dopa.size else 0.0
            self._formation_metrics(dist)

    # -- indicateurs -------------------------------------------------------------------- #

    def polarization(self) -> float:
        if not self.drones:
            return 0.0
        sx = sum(math.cos(d.heading) for d in self.drones)
        sy = sum(math.sin(d.heading) for d in self.drones)
        return math.hypot(sx, sy) / len(self.drones)

    def squad_polarization(self) -> float:
        """Polarisation moyenne INTRA-PATROUILLE : elle, doit rester elevee."""
        if not self.squads:
            return 0.0
        acc, cnt = 0.0, 0
        for members in self.squads:
            if not members:
                continue
            sx = sum(math.cos(self.drones[i].heading) for i in members)
            sy = sum(math.sin(self.drones[i].heading) for i in members)
            acc += math.hypot(sx, sy) / len(members)
            cnt += 1
        return acc / max(1, cnt)

    def phase_remaining(self) -> float:
        p = self.params
        limit_s = {
            "explore": p.t_explore,
            "rally": p.t_rally_max,
            "share": p.t_share,
        }[self.phase]
        return max(0.0, limit_s - self.phase_t)

    def metrics(self) -> dict:
        w = self.world
        if self.drones:
            sp_min = min(d.speed for d in self.drones)
            sp_max = max(d.speed for d in self.drones)
        else:
            sp_min = sp_max = 0.0
        return {
            "found": int(w.found.sum()),
            "targets": w.n_targets,
            "conf_col": self.conf_col,
            "conf_solo": self.conf_solo,
            "gap": self.conf_gap,
            "gap_before": self.gap_before,
            "gap_after": self.gap_after,
            "dopa_max": self.dopa_max,
            "polar": self.polarization(),
            "polar_sq": self.squad_polarization(),
            "degree": self.neigh.mean_degree,
            "nn_dist": self.neigh.mean_nn_dist,
            "violations": self.neigh.violations,
            "comps": self.n_components,
            "squads": self.n_squads,
            "d_squad": self.d_squad,
            "ratio": self.squad_ratio,
            "overlap": self.overlap,
            "split": self.squad_split,
            "swath": self.params.swath_width(),
            "v_mean": self.mean_speed,
            "v_min": sp_min,
            "v_max": sp_max,
        }

    def layer(self, name: str, collective: bool) -> np.ndarray | None:
        if not self.conf.size:
            return None
        src = {"confiance": self.conf, "dopamine": self.dopa, "cortisol": self.cort}
        if name not in src:
            return None
        return src[name].max(axis=0) if collective else src[name][0]


#  Interface


def render_fields(
    screen: pygame.Surface, layers, cols: int, rows: int, cell_m: float
) -> None:
    """
    Compose plusieurs champs scalaires en une seule image de fond.
    `layers` est une liste de (champ, couleur, echelle, alpha_max).

    La mise a l'echelle passe par px() sur la taille TOTALE de la grille en
    metres, et non par un nombre entier de pixels par cellule : sans cela un
    arrondi decale la carte de plusieurs dizaines de pixels au bord droit.
    """
    rgb = np.empty((cols, rows, 3), dtype=np.float64)
    rgb[:] = np.array(C_BG, dtype=np.float64)
    drawn = False
    for fld, color, scale, alpha in layers:
        if fld is None:
            continue
        a = (np.clip(fld * scale, 0.0, 1.0) * alpha)[..., None]
        rgb = rgb * (1.0 - a) + np.array(color, dtype=np.float64) * a
        drawn = True
    if not drawn:
        return
    surf = pygame.surfarray.make_surface(rgb.astype(np.uint8))
    target = (px(cols * cell_m), px(rows * cell_m))
    screen.blit(pygame.transform.scale(surf, target), (0, 0))


def draw_hierarchy(screen, world: World) -> None:
    """Arbre de la formation : segments parent -> enfant, teintes par niveau."""
    for a, b, niv in world.hier_edges:
        col = C_HIER[min(niv, len(C_HIER) - 1)]
        pygame.draw.line(screen, col, (px(a[0]), px(a[1])), (px(b[0]), px(b[1])), 1)
    if world.hier_root is not None:
        for nd in parcourir(world.hier_root):
            if nd.est_terminal:
                continue
            col = C_HIER[min(nd.niveau, len(C_HIER) - 1)]
            pygame.draw.circle(screen, col, (px(nd.centre[0]), px(nd.centre[1])), 3, 1)


def draw_squad_links(screen, swarm: Swarm) -> None:
    """
    Enveloppe de chaque patrouille : segments entre membres, dans la teinte de
    la patrouille. C'est le moyen le plus direct de voir si la structure tient
    ou si deux patrouilles se sont interpenetrees.
    """
    for k, members in enumerate(swarm.squads):
        if len(members) < 2:
            continue
        col = squad_color(k)
        faded = tuple(int(c * 0.45) for c in col)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pa = swarm.drones[members[a]].pos
                pb = swarm.drones[members[b]].pos
                pygame.draw.line(
                    screen, faded, (px(pa.x), px(pa.y)), (px(pb.x), px(pb.y)), 1
                )


def draw_phase_banner(screen, font_big, font, swarm: Swarm, W: int) -> None:
    """Bandeau de phase, visible meme quand le panneau est masque."""
    prof = PHASES[swarm.phase]
    # en ralliement, le compte a rebours est un delai d'abandon, pas une duree
    mot = "limite" if swarm.phase == "rally" else "reste"
    txt = (
        f"{prof.label}   cycle {swarm.cycle}   "
        f"{mot} {swarm.phase_remaining():4.1f} s   "
        f"patrouilles {swarm.n_squads}   groupes {swarm.n_components}"
    )
    surf = font_big.render(txt, True, prof.color)
    rect = surf.get_rect(midtop=(W // 2, 12))
    bgr = pygame.Surface((rect.width + 26, rect.height + 12), pygame.SRCALPHA)
    bgr.fill((18, 18, 20, 225))
    screen.blit(bgr, (rect.left - 13, rect.top - 6))
    screen.blit(surf, rect)
    sub = font.render(
        f"partages reussis {swarm.shares_ok}   echoues {swarm.shares_failed}"
        f"   patrouilles coupees {swarm.squad_split}",
        True,
        (150, 150, 158),
    )
    screen.blit(sub, sub.get_rect(midtop=(W // 2, rect.bottom + 8)))


def draw_rally_marker(screen, swarm: Swarm) -> None:
    if swarm.rally_pt is None:
        return
    x, y = px(swarm.rally_pt.x), px(swarm.rally_pt.y)
    r = px(swarm.params.rally_arrival)
    pygame.draw.circle(screen, C_RV, (x, y), r, 1)
    pygame.draw.line(screen, C_RV, (x - 9, y), (x + 9, y), 2)
    pygame.draw.line(screen, C_RV, (x, y - 9), (x, y + 9), 2)


def draw_hud(
    screen,
    font,
    params: Params,
    swarm: Swarm,
    selected: int,
    fps: float,
    layer: str,
    collective: bool,
    lh: int,
) -> None:
    m = swarm.metrics()
    prof = PHASES[swarm.phase]
    if swarm.done and swarm.completion_time is not None:
        t_txt = f"t {swarm.completion_time:6.1f}s FINI"
    else:
        t_txt = f"t {swarm.time:6.1f}s"
    header = [
        (f"FPS {fps:5.1f}  drones {len(swarm.drones):3d}  {t_txt}", (200, 200, 205)),
        (f"cibles {m['found']:2d} / {m['targets']:2d}", (120, 220, 190)),
        (
            f"phase {prof.label[:11]:<11} cycle {swarm.cycle:2d} "
            f"{swarm.phase_remaining():4.1f}s",
            prof.color,
        ),
        (f"rendez-vous : {params.rally_mode}", C_RV),
        (
            f"radio R={params.comm_radius:.0f} m  groupes {m['comps']:2d}",
            (150, 200, 240),
        ),
        # --- bloc patrouilles : c'est ce qui se lit pour regler la formation
        (
            f"patrouilles {m['squads']:2d} x ~{params.squad_size:d}"
            f"   coupees {m['split']:2d}",
            (200, 200, 205),
        ),
        (
            f"fauchee {m['swath']:5.1f} m  d.equipier {m['d_squad']:5.1f} m "
            f"(x{m['ratio']:4.2f})",
            (200, 200, 205),
        ),
        (
            f"recouvrement {m['overlap']:4.2f}  polar patr. {m['polar_sq']:4.2f}",
            (200, 200, 205),
        ),
        (
            "brisure : "
            + (
                f"eventail TEMOIN g={params.g_fan:.2f}"
                if params.fan_enabled
                else f"reservations {params.claim_strength:.2f}"
            ),
            (240, 180, 120) if params.fan_enabled else (120, 220, 190),
        ),
        (
            f"voisinage {params.neighborhood}"
            + (
                f" k={params.k_neighbors}"
                if params.neighborhood == "knn"
                else f" R={params.perception_radius:.0f} m"
            ),
            (150, 200, 240),
        ),
        (
            f"influence {params.influence}  arb. {params.arbitration[:4]}",
            (150, 200, 240),
        ),
        (
            "dopamine : "
            + ("saturee" if params.dopa_saturate else "lineaire")
            + f"  max {m['dopa_max']:5.1f}",
            C_DOPA,
        ),
        (
            f"couche : {layer} ({'collectif' if collective else 'drone 0'})",
            (240, 180, 120),
        ),
        ("", None),
        (
            f"confiance {m['conf_col'] * 100:5.1f} %  (d0 {m['conf_solo'] * 100:4.1f} %)",
            (120, 220, 190),
        ),
        (f"ecart de connaissance {m['gap'] * 100:5.2f} %", C_DOPA),
        (
            f"  dernier partage {m['gap_before'] * 100:5.2f} -> "
            f"{m['gap_after'] * 100:5.2f} %",
            (150, 150, 158),
        ),
        (
            f"vitesse {m['v_mean']:4.1f} m/s [{m['v_min']:4.1f}-{m['v_max']:4.1f}]",
            (200, 200, 205),
        ),
        (
            f"polar {m['polar']:5.2f}  voisins {m['degree']:4.1f}  "
            f"dmin {m['nn_dist']:5.1f} m",
            (200, 200, 205),
        ),
    ]
    h = 16 + lh * (len(header) + len(TUNABLES) + len(ALL_BEHAVIORS) + 3)
    panel = pygame.Surface((300, h), pygame.SRCALPHA)
    panel.fill((18, 18, 20, 220))
    screen.blit(panel, (10, 8))

    y = 12
    for txt, col in header:
        if txt:
            screen.blit(font.render("  " + txt, True, col), (16, y))
        y += lh

    y += lh // 2
    for i, (label, attr, _lo, _hi, _st) in enumerate(TUNABLES):
        val = getattr(params, attr)
        txt = (
            f"{label:<13}{val:7.2f}"
            if isinstance(val, float)
            else f"{label:<13}{val:7d}"
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

    y += lh // 2
    screen.blit(
        font.render("  budget alloue (m/s), ordre de la phase", True, (170, 170, 178)),
        (16, y),
    )
    y += lh
    total = max(1e-6, params.authority)
    for name in prof.order:
        v = swarm.alloc_mean.get(name, 0.0)
        bw = int(76 * min(1.0, v / total))
        screen.blit(font.render(f"  {name[:10]:<11}", True, (150, 150, 158)), (16, y))
        pygame.draw.rect(screen, (55, 55, 62), pygame.Rect(134, y + 3, 76, 8), 1)
        if bw:
            pygame.draw.rect(screen, prof.color, pygame.Rect(134, y + 3, bw, 8))
        screen.blit(font.render(f"{v:5.1f}", True, (150, 150, 158)), (216, y))
        y += lh


def draw_scale_bar(screen, font, W: int, H: int) -> None:
    """Echelle graphique : 50 m."""
    length = px(50.0)
    x0, y0 = W - length - 30, H - 28
    pygame.draw.line(screen, (170, 170, 178), (x0, y0), (x0 + length, y0), 2)
    pygame.draw.line(screen, (170, 170, 178), (x0, y0 - 5), (x0, y0 + 5), 2)
    pygame.draw.line(
        screen, (170, 170, 178), (x0 + length, y0 - 5), (x0 + length, y0 + 5), 2
    )
    screen.blit(
        font.render("50 m", True, (170, 170, 178)), (x0 + length // 2 - 14, y0 + 8)
    )


def draw_done_banner(screen, font_big, swarm: Swarm, W: int, H: int) -> None:
    if not swarm.done or swarm.completion_time is None:
        return
    msg = (
        f"MISSION TERMINEE — {swarm.world.n_targets} cibles en "
        f"{swarm.completion_time:.1f} s, {swarm.cycle} cycles   (R : relancer)"
    )
    surf = font_big.render(msg, True, C_FOUND)
    rect = surf.get_rect(center=(W // 2, H - 60))
    bgr = pygame.Surface((rect.width + 24, rect.height + 14), pygame.SRCALPHA)
    bgr.fill((18, 18, 20, 235))
    screen.blit(bgr, (rect.left - 12, rect.top - 7))
    screen.blit(surf, rect)


def cycle_of(seq, cur):
    return seq[(seq.index(cur) + 1) % len(seq)]


def main(headless: bool = False, max_frames: int = 0, seed: int | None = None) -> None:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    W, H = px(ZONE_W_M), px(ZONE_H_M)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        f"Essaim de drones — {ZONE_W_M} x {ZONE_H_M} m, ratissage par patrouilles"
    )
    clock = pygame.time.Clock()

    # La hauteur de ligne du panneau est deduite de la place disponible : avec
    # 34 reglages et 9 barres de budget, une valeur fixe deborderait.
    lh = max(10, min(15, (H - 40) // HUD_LINES))
    font = pygame.font.SysFont("monospace", max(9, lh - 3))
    font_big = pygame.font.SysFont("monospace", 19, bold=True)

    params = Params()
    bounds = pygame.Rect(0, 0, ZONE_W_M, ZONE_H_M)  # en METRES
    swarm = Swarm(params, bounds, seed)

    selected = 0
    paused = False
    layer = "tout"
    collective = True
    show_links = True
    show_vision = False
    show_cone = False
    show_comm_radii = False
    show_swath = False
    show_hud = True
    show_hier = False
    frames = 0
    running = True

    while running:
        # En headless le pas est FIXE : sans cela la trajectoire dependrait de
        # la charge de la machine et une meme graine ne redonnerait pas la meme
        # mission. On ne bride pas non plus la cadence, il n'y a rien a afficher.
        if headless:
            clock.tick()
            dt = 1.0 / 60.0
        else:
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
                elif ev.key == pygame.K_g:
                    swarm._advance_phase()
                elif ev.key == pygame.K_f:
                    params.rally_mode = cycle_of(RALLY_MODES, params.rally_mode)
                elif ev.key == pygame.K_d:
                    params.dopa_saturate = not params.dopa_saturate
                elif ev.key == pygame.K_e:
                    params.fan_enabled = not params.fan_enabled
                elif ev.key == pygame.K_t:
                    show_hier = not show_hier
                elif ev.key == pygame.K_c:
                    layer = cycle_of(LAYERS, layer)
                elif ev.key == pygame.K_m:
                    collective = not collective
                elif ev.key == pygame.K_l:
                    show_links = not show_links
                elif ev.key == pygame.K_v:
                    show_vision = not show_vision
                    for d in swarm.drones:
                        d.color = squad_color(d.squad)
                elif ev.key == pygame.K_s:
                    show_cone = not show_cone
                elif ev.key == pygame.K_w:
                    show_swath = not show_swath
                elif ev.key == pygame.K_p:
                    show_comm_radii = not show_comm_radii
                elif ev.key == pygame.K_h:
                    show_hud = not show_hud
                elif ev.key == pygame.K_n:
                    params.neighborhood = cycle_of(NEIGHBORHOODS, params.neighborhood)
                elif ev.key == pygame.K_i:
                    params.influence = cycle_of(INFLUENCES, params.influence)
                elif ev.key == pygame.K_a:
                    params.arbitration = cycle_of(ARBITRATIONS, params.arbitration)
                elif ev.key == pygame.K_b:
                    params.boundary_mode = cycle_of(BOUNDARIES, params.boundary_mode)
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

        # --- champs scalaires : confiance (vert), cortisol (bleu), dopamine (orange)
        if layer == "tout":
            render_fields(
                screen,
                [
                    (swarm.layer("confiance", collective), C_CONF, 1.0, 0.28),
                    (swarm.layer("cortisol", collective), C_CORT, 0.5, 0.50),
                    (swarm.layer("dopamine", collective), C_DOPA, 0.25, 0.80),
                ],
                w.cols,
                w.rows,
                w.cell,
            )
        elif layer == "confiance":
            render_fields(
                screen,
                [(swarm.layer(layer, collective), C_CONF, 1.0, 0.55)],
                w.cols,
                w.rows,
                w.cell,
            )
        elif layer == "dopamine":
            render_fields(
                screen,
                [(swarm.layer(layer, collective), C_DOPA, 0.25, 0.85)],
                w.cols,
                w.rows,
                w.cell,
            )
        elif layer == "cortisol":
            render_fields(
                screen,
                [(swarm.layer(layer, collective), C_CORT, 0.5, 0.75)],
                w.cols,
                w.rows,
                w.cell,
            )

        # --- obstacles
        for ox, oy, orad in w.obstacles:
            pygame.draw.circle(screen, C_OBST, (px(ox), px(oy)), px(orad))
            pygame.draw.circle(screen, (80, 80, 92), (px(ox), px(oy)), px(orad), 1)

        # --- structure hierarchique de la formation (touche T)
        if show_hier:
            draw_hierarchy(screen, w)

        # --- points de rendez-vous programmes
        if params.rally_mode == "schedule" and w.rv_points:
            active = swarm.cycle % len(w.rv_points)
            for k, pt in enumerate(w.rv_points):
                col = C_RV if k == active else (70, 62, 48)
                pygame.draw.circle(screen, col, (px(pt.x), px(pt.y)), 4, 1)

        # --- camion et aire de decollage
        tx, ty, tw, th = w.truck
        pygame.draw.rect(screen, C_TRUCK, pygame.Rect(px(tx), px(ty), px(tw), px(th)))
        ax, ay, aw, ah = w.pad
        pygame.draw.rect(
            screen, (70, 68, 58), pygame.Rect(px(ax), px(ay), px(aw), px(ah)), 1
        )

        # --- cibles : cercle gris = non trouvee, disque vert = trouvee
        for i, j in np.argwhere(w.target_mask):
            x = px((i + 0.5) * w.cell)
            y = px((j + 0.5) * w.cell)
            if w.found[i, j]:
                pygame.draw.circle(screen, C_FOUND, (x, y), 5)
            else:
                pygame.draw.circle(screen, C_TARGET, (x, y), 4, 1)

        # --- marge de rappel
        if params.boundary_mode == "steer":
            mg = px(params.boundary_margin)
            pygame.draw.rect(
                screen, (55, 55, 62), pygame.Rect(mg, mg, W - 2 * mg, H - 2 * mg), 1
            )

        # --- structure des patrouilles, puis liaisons radio actives
        draw_squad_links(screen, swarm)
        if show_links:
            for i, j in swarm.comm_pairs:
                a, b = swarm.drones[i].pos, swarm.drones[j].pos
                pygame.draw.line(
                    screen, (50, 80, 105), (px(a.x), px(a.y)), (px(b.x), px(b.y)), 1
                )

        # --- rayons de communication (touche P) : disque de portee radio,
        #     identique pour tous les drones puisque comm_radius est global.
        #     A croiser avec la touche L : deux cercles qui se recoupent
        #     correspondent exactement a une liaison active.
        if show_comm_radii:
            for d in swarm.drones:
                pygame.draw.circle(
                    screen,
                    (45, 70, 90),
                    (px(d.pos.x), px(d.pos.y)),
                    px(params.comm_radius),
                    1,
                )

        # --- voisinage du drone 0
        if show_vision and swarm.drones:
            d0 = swarm.drones[0]
            for d in swarm.drones:
                d.color = squad_color(d.squad)
            d0.color = C_HL
            if params.neighborhood == "metric":
                pygame.draw.circle(
                    screen,
                    C_HL,
                    (px(d0.pos.x), px(d0.pos.y)),
                    px(params.perception_radius),
                    1,
                )
            for other, _d, _wt in (swarm.last_nbs[0] if swarm.last_nbs else []):
                other.color = (200, 160, 90)
                pygame.draw.line(
                    screen,
                    (90, 70, 50),
                    (px(d0.pos.x), px(d0.pos.y)),
                    (px(other.pos.x), px(other.pos.y)),
                    1,
                )
            if d0.target is not None:
                pygame.draw.line(
                    screen,
                    C_CONF,
                    (px(d0.pos.x), px(d0.pos.y)),
                    (px(d0.target.x), px(d0.target.y)),
                    1,
                )

        draw_rally_marker(screen, swarm)

        for d in swarm.drones:
            d.draw(screen, show_cone, show_swath)

        draw_phase_banner(screen, font_big, font, swarm, W)
        if show_hud:
            draw_hud(
                screen,
                font,
                params,
                swarm,
                selected,
                clock.get_fps(),
                layer,
                collective,
                lh,
            )
        draw_scale_bar(screen, font, W, H)
        draw_done_banner(screen, font_big, swarm, W, H)
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
