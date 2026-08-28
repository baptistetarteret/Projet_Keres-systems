"""RQ: pour l'importation de flocking, fn doit respecter la signature update
(drones, surface_rect, dt) -> None."""

import pygame
import config
from drone_state import DroneState
 
 
class Simulator:
    def __init__(self, num_drones=None, update_fn=None, init_positions=None):
        """
        num_drones      : nombre de drones à créer si init_positions n'est
                           pas fourni (positions générées automatiquement)
        update_fn       : fonction respectant le contrat décrit plus haut.
                           Peut aussi être fournie/remplacée plus tard via
                           set_update_function().
        init_positions  : liste optionnelle de tuples (x, y) pour fixer
                           les positions de départ des drones.
        """
        self.surface_rect = pygame.Rect(
            config.SURFACE_MARGIN,
            config.SURFACE_MARGIN,
            config.SCREEN_WIDTH - 2 * config.SURFACE_MARGIN,
            config.SCREEN_HEIGHT - 2 * config.SURFACE_MARGIN,
        )
 
        if init_positions is not None:
            self.drones = [
                DroneState(i, x, y) for i, (x, y) in enumerate(init_positions)
            ]
        else:
            num_drones = num_drones or config.NUM_DRONES_DEMO
            self.drones = [self._default_spawn(i) for i in range(num_drones)]
 
        self.update_fn = update_fn
 
        self._screen = None
        self._coverage_overlay = None
        self._coverage_cells = None  # set des cellules couvertes pour l'affichage
 
    def _default_spawn(self, i):        # Si non renseigné les drones sont réparties au centre de 
        cx = self.surface_rect.centerx  # la simu au départ        
        cy = self.surface_rect.centery
        return DroneState(i, cx + (i % 5) * 15 - 30, cy + (i // 5) * 15 - 15)
 
    # ------------------------------------------------------------------
    # Interface pour remplacer la logique de flocking
    # ------------------------------------------------------------------

    def set_update_function(self, fn):
        self.update_fn = fn
 
    def add_drone(self, x, y, drone_id=None):
        drone_id = drone_id if drone_id is not None else len(self.drones)
        d = DroneState(drone_id, x, y)
        self.drones.append(d)
        return d
 
    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self):
        pygame.init()
        self._screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Simulateur d'essaim de drones")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)
 
        if config.SHOW_COVERAGE:
            self._coverage_overlay = pygame.Surface(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA
            )
            self._coverage_cells = set()
 
        paused = False
        running = True
 
        while running:
            dt = clock.tick(config.FPS) / 1000.0
 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self._reset_coverage()
 
            if not paused:
                if self.update_fn is not None:
                    self.update_fn(self.drones, self.surface_rect, dt)
                self._update_trails()
                if config.SHOW_COVERAGE:
                    self._update_coverage()
 
            self._render(font, clock, paused)
 
        pygame.quit()
 
    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------
    def _render(self, font, clock, paused):
        screen = self._screen
        screen.fill(config.BACKGROUND_COLOR)
 
        pygame.draw.rect(screen, config.SURFACE_BORDER_COLOR, self.surface_rect, width=2)
 
        if config.SHOW_COVERAGE and self._coverage_overlay is not None:
            screen.blit(self._coverage_overlay, (0, 0))
 
        if config.SHOW_TRAIL:
            for drone in self.drones:
                if len(drone.trail) >= 2:
                    pygame.draw.lines(screen, config.TRAIL_COLOR, False, drone.trail, 1)
 
        for drone in self.drones:
            self._draw_drone(screen, drone, font)
 
        self._draw_hud(screen, font, clock, paused)
 
        pygame.display.flip()
 
    def _draw_drone(self, screen, drone, font):         #Definition forme des drones
        import math

        angle = drone.heading
        r = config.DRONE_RADIUS
        pos = pygame.math.Vector2(drone.x, drone.y)
 
        tip = pos + pygame.math.Vector2(r * 2.2, 0).rotate_rad(angle)
        left = pos + pygame.math.Vector2(-r * 1.4, r * 1.4).rotate_rad(angle)
        right = pos + pygame.math.Vector2(-r * 1.4, -r * 1.4).rotate_rad(angle)
        pygame.draw.polygon(screen, config.DRONE_COLOR, [tip, left, right])
        
        #pygame.draw.circle(screen, config.DRONE_COLOR, (int(drone.x), int(drone.y)), config.DRONE_RADIUS)
        pygame.draw.circle(
        screen,
        (220, 60, 60),                                    
        (int(drone.x), int(drone.y)),
        int(config.SENSOR_RADIUS_VISUAL),
        width=1,                                          
        )
    
        if config.DRONE_ID_LABEL:
            label = font.render(str(drone.id), True, (200, 200, 200))
            screen.blit(label, (drone.x + r, drone.y - r))
 
    def _draw_hud(self, screen, font, clock, paused):
        lines = [
            f"FPS: {clock.get_fps():.0f}",
            f"Drones: {len(self.drones)}",
        ]
        if config.SHOW_COVERAGE and self._coverage_cells is not None:
            total_cols = self.surface_rect.width // config.GRID_CELL_SIZE + 1
            total_rows = self.surface_rect.height // config.GRID_CELL_SIZE + 1
            total = max(1, total_cols * total_rows)
            pct = 100.0 * len(self._coverage_cells) / total
            lines.append(f"Couverture: {pct:.1f}%")
        if paused:
            lines.append("PAUSE (Espace pour reprendre)")
 
        for i, line in enumerate(lines):
            surf = font.render(line, True, (230, 230, 230))
            screen.blit(surf, (10, 10 + i * 20))
 
    # ------------------------------------------------------------------
    # Traînée (pas necessaire)
    # ------------------------------------------------------------------
    def _update_trails(self):
        for drone in self.drones:
            drone.trail.append((drone.x, drone.y))
            if len(drone.trail) > config.TRAIL_LENGTH:
                drone.trail.pop(0)
 
    # ------------------------------------------------------------------
    # Visualisation et calcul de la couverture
    # ------------------------------------------------------------------
    def _reset_coverage(self):
        if self._coverage_overlay is not None:
            self._coverage_overlay.fill((0, 0, 0, 0))
            self._coverage_cells = set()
 
    def _update_coverage(self):
        cell = config.GRID_CELL_SIZE
        r = config.SENSOR_RADIUS_VISUAL
        r_cells = int(r // cell) + 1
        r_sq = r * r
 
        for drone in self.drones:
            cx = int(drone.x // cell)
            cy = int(drone.y // cell)
            for gx in range(cx - r_cells, cx + r_cells + 1):
                for gy in range(cy - r_cells, cy + r_cells + 1):
                    if (gx, gy) in self._coverage_cells:
                        continue
                    px, py = gx * cell + cell / 2, gy * cell + cell / 2
                    if (px - drone.x) ** 2 + (py - drone.y) ** 2 <= r_sq:
                        if self.surface_rect.collidepoint(px, py):
                            self._coverage_cells.add((gx, gy))
                            pygame.draw.rect(
                                self._coverage_overlay,
                                config.COVERED_COLOR,
                                (gx * cell, gy * cell, cell, cell),
                            )