#!/usr/bin/env python3
"""
Pac-Man  –  Python translation of the original Atari 1982 source code
Original:  Developed by Roklan Corp for Atari Inc., 10/03/82  (Revision 3.0)
Translated from 6502 assembly into human-readable Python.

Key assembly files and what they map to here:
  PACMAN.ASM  → Game class (main loop, option/attract, init, DLI/VBI stubs)
  PAC1.ASM    → Game._update_* (VBlank subroutines: attract, collision, death)
  PAC2.ASM    → Game._check_* (rerack/level-clear, ready seq, sounds→omitted)
  PAC3.ASM    → PacMan class (PMSTIK movement, MUNCHY dot eating, PSCORE)
  PAC4.ASM    → Ghost class (MONSTR/MCHASE/GOHOME/MDIRCT/EYONLY AI)
  PACDAT1.ASM → MAZE_DEF (translated from DATMAZ tile data)

Controls:  Arrow keys / WASD – move    P – pause    Q / Esc – quit
Deps:      pip install pygame
"""

import math, sys, random
import pygame

# ══════════════════════════════════════════════════════════════════
#  DISPLAY & TIMING
# ══════════════════════════════════════════════════════════════════
CELL      = 16           # pixels per maze tile
COLS, ROWS = 28, 31
SCREEN_W  = COLS * CELL  # 448 px
SCREEN_H  = ROWS * CELL + 72  # 568 px  (maze + HUD strip)
FPS       = 60

# ══════════════════════════════════════════════════════════════════
#  PALETTE  – approximate Atari NTSC register values
#  (ACOLR1–4, PCOLR0–3, COLPF0–3 referenced throughout assembly)
# ══════════════════════════════════════════════════════════════════
BLACK   = (  0,   0,   0)
WHITE   = (255, 255, 255)
YELLOW  = (255, 255,   0)   # Pac-Man  (COLPF3 = $2A)
WALL_C  = ( 33,  33, 255)   # maze walls  (COLPF2 = $3A → blue)
RED     = (220,   0,   0)   # Blinky   PCOLR0
PINK    = (255, 184, 255)   # Pinky    PCOLR1
CYAN    = (  0, 255, 255)   # Inky     PCOLR2
ORANGE  = (255, 184,  82)   # Clyde    PCOLR3
FRIGHT  = (  0,   0, 160)   # frightened ghost body  (COLPF0 = $84 in flight)
FRIGHT2 = (255, 255, 255)   # flashing-white near end
DOT_C   = (255, 184, 255)   # small dot  (COLPF1)
DOOR_C  = (255, 184, 255)   # ghost house door line
EYE_W   = (255, 255, 255)   # ghost eye white
EYE_P   = (  0,   0, 220)   # ghost eye pupil
HUD_C   = (255, 255, 255)

# ══════════════════════════════════════════════════════════════════
#  DIRECTION FLAGS  (bit masks matching assembly: UP=1 DN=2 LF=4 RT=8)
# ══════════════════════════════════════════════════════════════════
UP, DOWN, LEFT, RIGHT, NONE = 1, 2, 4, 8, 0
ALL_DIRS = (UP, DOWN, LEFT, RIGHT)
DELTA    = {UP:(0,-1), DOWN:(0,1), LEFT:(-1,0), RIGHT:(1,0), NONE:(0,0)}
OPPOSITE = {UP:DOWN, DOWN:UP, LEFT:RIGHT, RIGHT:LEFT, NONE:NONE}

# ══════════════════════════════════════════════════════════════════
#  GHOST STATES  (M1STAT register values, PAC4.ASM)
# ══════════════════════════════════════════════════════════════════
GS_INACTIVE   = 0    # waiting in ghost house; M1TIMR counting down
GS_LEAVING    = 1    # exiting ghost house  (MSTRTP / PNKMOT path)
GS_SCATTER    = 2    # patrol home corner   (GOHOME / MNTST2)
GS_CHASE      = 8    # chase Pac-Man        (MCHASE)
GS_FRIGHTENED = 128  # blue/flight mode     (DOTTST → ORA #$80)
GS_EATEN      = 66   # eyes only, returning (ZAPGST → M1STAT = $42)

# ══════════════════════════════════════════════════════════════════
#  MAZE CELL TYPES  (DOT=1 / POWER=2 match MUNCHY checks in PAC3.ASM)
# ══════════════════════════════════════════════════════════════════
EMPTY      = 0
DOT_CELL   = 1
POWER_CELL = 2
WALL_CELL  = 3
DOOR_CELL  = 4   # ghost door: ghosts pass, Pac-Man blocked
GHOST_AREA = 5   # ghost house interior: only ghosts enter/exit

# ══════════════════════════════════════════════════════════════════
#  MAZE DEFINITION  28 cols × 31 rows
#
#  Translated from DATMAZ tile data in PACDAT1.ASM.
#  '#' wall   '.' dot (10 pts)   'o' power pellet (50 pts)
#  ' ' empty  '-' ghost door     '~' ghost-house interior
# ══════════════════════════════════════════════════════════════════
MAZE_DEF = [
    "############################",   #  0
    "#............##............#",   #  1
    "#.####.#####.##.#####.####.#",   #  2
    "#o####.#####.##.#####.####o#",   #  3
    "#.####.#####.##.#####.####.#",   #  4
    "#..........................#",   #  5
    "#.####.##.########.##.####.#",   #  6
    "#.####.##.########.##.####.#",   #  7
    "#......##....##....##......#",   #  8
    "######.#####.##.#####.######",   #  9
    "######.#####.##.#####.######",   # 10
    "######.##          ##.######",   # 11
    "######.## ###--### ##.######",   # 12
    "######.## #~~~~~~# ##.######",   # 13
    "      .   #~~~~~~#   .      ",   # 14  ← tunnel exits at col 0 & col 27
    "######.## #~~~~~~# ##.######",   # 15
    "######.## ######## ##.######",   # 16
    "######.##          ##.######",   # 17
    "######.## ######## ##.######",   # 18
    "######.## ######## ##.######",   # 19
    "#............##............#",   # 20
    "#.####.#####.##.#####.####.#",   # 21
    "#.####.#####.##.#####.####.#",   # 22
    "#o..##.......  .......##..o#",   # 23
    "###.##.##.########.##.##.###",   # 24
    "###.##.##.########.##.##.###",   # 25
    "#......##....##....##......#",   # 26
    "#.##########.##.##########.#",   # 27
    "#.##########.##.##########.#",   # 28
    "#..........................#",   # 29
    "############################",   # 30
]

# ══════════════════════════════════════════════════════════════════
#  LEVEL PARAMETERS
#  Derived from PACSPD / MONSPD / BLUTIM / SPEED1 / STARTV tables.
#  (pac_pps, ghost_pps, fright_frames, [ghost_release_delays_in_frames])
# ══════════════════════════════════════════════════════════════════
LEVEL_PARAMS = [
    ( 90,  75, 360, [  0,  60, 120, 240]),   # level 1
    ( 95,  80, 300, [  0,  50, 100, 180]),   # level 2
    (100,  85, 240, [  0,  40,  80, 140]),   # level 3
    (100,  90, 180, [  0,  30,  60, 100]),   # level 4
    (100,  95, 120, [  0,  20,  50,  80]),   # level 5+
]

# FRUCHR / FRSTAB tables: (name, score, colour)
FRUITS = [
    ("CHERRY",     100, (255,   0,   0)),
    ("STRAWBERRY", 300, (255,  50,  50)),
    ("ORANGE",     500, (255, 140,   0)),
    ("APPLE",      700, ( 50, 200,  50)),
    ("MELON",     1000, ( 50, 200,  50)),
    ("GALAXIAN",  2000, (200,  50, 200)),
    ("BELL",      3000, (255, 220,   0)),
    ("KEY",       5000, (200, 200, 200)),
]

GHOST_SCORES  = [200, 400, 800, 1600]   # GLPCNT-indexed, ZAPGST / PSCORE
BONUS_LIFE_AT = 10_000                   # BPACP1 threshold


# ══════════════════════════════════════════════════════════════════
#  MAZE UTILITIES
# ══════════════════════════════════════════════════════════════════

def build_maze(template):
    grid = []
    for row_s in template:
        row = []
        for ch in row_s:
            if   ch == '#': row.append(WALL_CELL)
            elif ch == '.': row.append(DOT_CELL)
            elif ch == 'o': row.append(POWER_CELL)
            elif ch == '-': row.append(DOOR_CELL)
            elif ch == '~': row.append(GHOST_AREA)
            else:            row.append(EMPTY)
        grid.append(row)
    return grid

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def pac_walkable(grid, r, c):
    """Pac-Man may enter tile (c,r). Out-of-bounds columns = tunnel side."""
    if c < 0 or c >= COLS: return True   # tunnel
    if not in_bounds(r, c): return False
    return grid[r][c] not in (WALL_CELL, DOOR_CELL, GHOST_AREA)

def ghost_walkable(grid, r, c, eaten=False):
    """Ghost may enter tile. Eaten ghosts can re-enter GHOST_AREA."""
    if c < 0 or c >= COLS: return True
    if not in_bounds(r, c): return False
    t = grid[r][c]
    if t == WALL_CELL: return False
    if t == GHOST_AREA and not eaten: return False
    return True


# ══════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════

def draw_pac(surface, cx, cy, r, facing_deg, mouth_deg, color=YELLOW):
    """
    Pac-Man sprite as a filled pie arc.
    Mirrors PACIDX / MOVPAC animation logic from PAC3.ASM:
      mouth cycles open↔closed; direction sets facing angle.
    """
    if r <= 0: return
    if mouth_deg < 2:
        pygame.draw.circle(surface, color, (int(cx), int(cy)), r)
        return
    sa = math.radians(facing_deg + mouth_deg)
    ea = math.radians(facing_deg + 360 - mouth_deg)
    pts = [(cx, cy)]
    n = max(16, r * 2)
    for i in range(n + 1):
        a = sa + (ea - sa) * i / n
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    if len(pts) >= 3:
        pygame.draw.polygon(surface, color, pts)


def draw_ghost_sprite(surface, cx, cy, r, color, direction, flash=False):
    """
    Ghost sprite: rounded head, rectangular body, wavy skirt, eyes.
    References MONSUP/MONSDN/MONSLF/MONSRT/MONSFL/MONSEY data in PAC4.ASM.
    """
    if r <= 0: return
    top_cy = int(cy) - r // 3
    pygame.draw.circle(surface, color, (int(cx), top_cy), r)
    body_rect = pygame.Rect(int(cx) - r, top_cy, r * 2, r + r // 2)
    pygame.draw.rect(surface, color, body_rect)
    # wavy skirt – 3 teeth (MSKIRT flag toggles skirt animation)
    br = max(1, r // 3)
    for i in range(3):
        bx = int(cx) - r + br + i * br * 2
        by = int(cy) + r // 2 + br // 2
        pygame.draw.circle(surface, BLACK, (bx, by), br)
    # eyes
    for sign in (-1, 1):
        ex = int(cx) + sign * (r // 3)
        ey = int(cy) - r // 5
        pygame.draw.circle(surface, EYE_W, (ex, ey), max(1, r // 4))
        if flash:
            pygame.draw.circle(surface, (255, 255, 255), (ex, ey), max(1, r // 6))
        else:
            dx, dy = DELTA.get(direction, (1, 0))
            px = ex + dx * max(1, r // 8)
            py = ey + dy * max(1, r // 8)
            pygame.draw.circle(surface, EYE_P, (px, py), max(1, r // 8))


def draw_ghost_eyes(surface, cx, cy, r, direction):
    """Eyes-only sprite for eaten ghost (MONSEY in PAC4.ASM)."""
    for sign in (-1, 1):
        ex = int(cx) + sign * (r // 3)
        ey = int(cy) - r // 5
        pygame.draw.circle(surface, EYE_W, (ex, ey), max(1, r // 4))
        dx, dy = DELTA.get(direction, (1, 0))
        px = ex + dx * max(1, r // 8)
        py = ey + dy * max(1, r // 8)
        pygame.draw.circle(surface, EYE_P, (px, py), max(1, r // 8))


def draw_maze(surface, grid, power_visible):
    """
    Render tile grid.  Approximates the DLIST / character-set rendering
    from the original Atari display list.
    """
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * CELL, r * CELL
            t = grid[r][c]
            if t == WALL_CELL:
                pygame.draw.rect(surface, WALL_C, (x, y, CELL, CELL))
                # inner recess for depth (mimics the blue border tiles)
                pygame.draw.rect(surface, (0, 0, 90), (x+2, y+2, CELL-4, CELL-4))
            elif t == DOT_CELL:
                pygame.draw.circle(surface, DOT_C, (x+CELL//2, y+CELL//2), 2)
            elif t == POWER_CELL and power_visible:
                pygame.draw.circle(surface, WHITE, (x+CELL//2, y+CELL//2), 5)
            elif t == DOOR_CELL:
                pygame.draw.rect(surface, DOOR_C, (x, y+CELL//2-1, CELL, 3))


# ══════════════════════════════════════════════════════════════════
#  PAC-MAN  (PAC3.ASM: PMSTIK, MUNCHY, PACUP/DN/LF/RT, MOVPAC)
# ══════════════════════════════════════════════════════════════════

class PacMan:
    # Start position (INIDAT in assembly: PMHPOS=PMVPOS init values)
    SPAWN = (13, 23)

    def __init__(self, speed):
        self.speed = speed   # pixels per second
        self.reset()

    def reset(self):
        self.col, self.row = self.SPAWN
        self.px = float(self.col * CELL + CELL // 2)   # PMHPOS
        self.py = float(self.row * CELL + CELL // 2)   # PMVPOS
        self.dir  = NONE   # current direction  (PMODIR)
        self.want = NONE   # buffered direction  (PMNDIR)
        self.alive       = True
        self.death_frame = 0   # VFIZST / VFIZSQ counter
        self.mouth_deg   = 30  # PMSEQU drives this via PACIDX table
        self.mouth_vel   = -1  # opening or closing
        self.mouth_tick  = 0
        self._accum      = 0.0   # sub-pixel movement accumulator

    def set_dir(self, d):
        """Store joystick input (PMNDIR set by PMSTIK scan)."""
        self.want = d

    def update(self, dt, grid):
        if not self.alive: return
        self._mouth()
        self._accum += self.speed * dt
        while self._accum >= 1.0:
            self._accum -= 1.0
            self._step(grid)
        self.col = int(self.px) // CELL
        self.row = int(self.py) // CELL

    def _mouth(self):
        """PMSEQU → PACIDX: 3-frame cycle, 8° per step, range 0-40°."""
        self.mouth_tick += 1
        if self.mouth_tick < 3: return
        self.mouth_tick = 0
        self.mouth_deg += 8 * self.mouth_vel
        if   self.mouth_deg <= 0:  self.mouth_vel =  1
        elif self.mouth_deg >= 40: self.mouth_vel = -1

    def _step(self, grid):
        """
        PMSTIK / MAZHND: one-pixel move.
        At tile centre try the buffered direction; if valid, commit.
        Then advance one pixel in current direction, checking walls.
        """
        # Check if we are at (or very close to) a tile centre
        cx_exact = self.col * CELL + CELL // 2
        cy_exact = self.row * CELL + CELL // 2
        near_cx  = abs(self.px - cx_exact) < 1.5
        near_cy  = abs(self.py - cy_exact) < 1.5
        at_centre = near_cx and near_cy

        # Try buffered direction change (PMNDIR logic in PMSTIK)
        if at_centre and self.want and self.want != self.dir:
            dc, dr = DELTA[self.want]
            if pac_walkable(grid, self.row + dr, self.col + dc):
                self.dir = self.want
                self.px  = float(cx_exact)
                self.py  = float(cy_exact)

        if self.dir == NONE: return

        dc, dr = DELTA[self.dir]
        nx = self.px + dc
        ny = self.py + dr

        # Tunnel wrap (TUNNEL subroutine)
        if nx < 0:        nx = float(SCREEN_W - 1)
        elif nx >= SCREEN_W: nx = 0.0

        nc = int(nx) // CELL
        nr = int(ny) // CELL
        if pac_walkable(grid, nr, nc):
            self.px, self.py = nx, ny
        else:
            # Snap to tile centre when hitting a wall
            self.px = float(cx_exact)
            self.py = float(cy_exact)

    def tile(self):
        return self.col, self.row

    def facing_deg(self):
        return {RIGHT:0, LEFT:180, UP:90, DOWN:270, NONE:0}.get(self.dir, 0)

    def draw(self, surface):
        cx, cy = int(self.px), int(self.py)
        r = CELL // 2 - 1
        if not self.alive:
            # VFIZZL / FIZZIE: shrink-to-nothing death animation
            prog = min(1.0, self.death_frame / 45)
            r = max(0, int(r * (1.0 - prog)))
            draw_pac(surface, cx, cy, r, 0, 0)
        else:
            mdeg = self.mouth_deg if self.dir != NONE else 0
            draw_pac(surface, cx, cy, r, self.facing_deg(), mdeg)


# ══════════════════════════════════════════════════════════════════
#  GHOST  (PAC4.ASM: MONSTR, MCHASE, GOHOME, MDIRCT, EYONLY, BOUNCE)
# ══════════════════════════════════════════════════════════════════

class Ghost:
    # HOMEHV table: scatter-mode corner targets
    SCATTER_TARGETS = [(25, 0), (2, 0), (25, 30), (0, 30)]
    COLORS = [RED, PINK, CYAN, ORANGE]
    NAMES  = ["BLINKY", "PINKY", "INKY", "CLYDE"]

    # Ghost house geometry
    HOUSE_EXIT   = (13, 11)   # tile above the door (where ghosts rejoin maze)
    HOUSE_CENTER = (13, 13)   # bounce origin inside ghost house

    # Pixel start positions for each ghost (INIDAT / M1HPOS / M1VPOS)
    STARTS = [(13, 11), (13, 13), (11, 13), (15, 13)]

    def __init__(self, idx, speed, release_delay):
        self.idx    = idx
        self.color  = self.COLORS[idx]
        self.speed  = speed
        self.delay  = release_delay   # M1TIMR initial value
        self._full_reset()

    # ── reset helpers ─────────────────────────────────────────
    def _full_reset(self):
        sc, sr = self.STARTS[self.idx]
        self.col, self.row   = sc, sr
        self.px = float(sc * CELL + CELL // 2)
        self.py = float(sr * CELL + CELL // 2)
        self.dir          = UP if self.idx == 0 else DOWN
        self.state        = GS_INACTIVE if self.idx > 0 else GS_LEAVING
        self.timer        = self.delay
        self.fright_timer = 0
        self._bounce_dir  = DOWN
        # Grid-based nav: always moving toward a committed next tile
        self.ntc, self.ntr = sc, sr   # next tile col, row

    def reset_after_eaten(self):
        """Reinitialise at ghost house centre; re-enter as GS_LEAVING."""
        hc, hr = self.HOUSE_CENTER
        self.col, self.row   = hc, hr
        self.px = float(hc * CELL + CELL // 2)
        self.py = float(hr * CELL + CELL // 2)
        self.dir          = UP
        self.state        = GS_LEAVING
        self.fright_timer = 0
        self.ntc, self.ntr = hc, hr

    # ── public state changers ─────────────────────────────────
    @property
    def is_frightened(self): return self.state == GS_FRIGHTENED
    @property
    def is_eaten(self):       return self.state == GS_EATEN

    def frighten(self, frames):
        """DOTTST: ORA #$80 – set flight mode, reverse direction."""
        if self.state not in (GS_INACTIVE, GS_EATEN, GS_LEAVING):
            self.state = GS_FRIGHTENED
            self.fright_timer = frames
            self.dir = OPPOSITE.get(self.dir, self.dir)

    def eat(self):
        """ZAPGST: M1STAT = $42 – eaten, eyes-only mode."""
        self.state = GS_EATEN
        self.dir   = UP

    def end_fright(self):
        if self.state == GS_FRIGHTENED:
            self.state = GS_SCATTER

    # ── per-frame update (SPDMON / MONSTR dispatch) ───────────
    def update(self, dt, grid, pac, all_ghosts):
        """
        Dispatches to behaviour methods matching the MONSTR state machine
        and SPDMON speed handler in PAC4.ASM.
        """
        self.col = int(self.px) // CELL
        self.row = int(self.py) // CELL

        if self.state == GS_INACTIVE:
            self._bounce(dt)
            self.timer -= 1
            if self.timer <= 0:
                self.state = GS_LEAVING

        elif self.state == GS_LEAVING:
            self._leave(dt)

        elif self.state == GS_FRIGHTENED:
            self.fright_timer -= 1
            if self.fright_timer <= 0:
                self.end_fright()
            else:
                self._navigate(dt, grid, target=None, random_mode=True)

        elif self.state == GS_EATEN:
            self._navigate(dt, grid, target=self.HOUSE_CENTER,
                           eaten_mode=True)
            # arrived?
            if (self.col, self.row) == self.HOUSE_CENTER:
                self.reset_after_eaten()

        elif self.state in (GS_SCATTER, GS_CHASE):
            target = self._pick_target(pac, all_ghosts)
            self._navigate(dt, grid, target=target)

    # ── movement behaviours ───────────────────────────────────
    def _bounce(self, dt):
        """
        BOUNCE subroutine: ghosts bob up/down while waiting in the ghost house
        (M1TIMR counting down, M1DIRT toggling between UP/DOWN at row limits).
        """
        step = self.speed * 0.4 * dt
        if self._bounce_dir == UP:
            self.py -= step
            if int(self.py) // CELL <= 12:
                self._bounce_dir = DOWN
        else:
            self.py += step
            if int(self.py) // CELL >= 15:
                self._bounce_dir = UP
        self.row = int(self.py) // CELL

    def _leave(self, dt):
        """
        MSTRTP / PNKMOT: navigate directly to HOUSE_EXIT tile, then
        enter scatter mode.  (Ghost house exit sequence from PAC4.ASM.)
        """
        ec, er = self.HOUSE_EXIT
        tx = float(ec * CELL + CELL // 2)
        ty = float(er * CELL + CELL // 2)
        ddx, ddy = tx - self.px, ty - self.py
        dist = math.hypot(ddx, ddy)
        step = self.speed * dt
        if dist <= step + 0.5:
            self.px, self.py = tx, ty
            self.col, self.row = ec, er
            self.ntc, self.ntr = ec, er
            self.dir   = LEFT
            self.state = GS_SCATTER
        else:
            s = step / dist
            self.px += ddx * s
            self.py += ddy * s
            self.dir = UP if ddy < 0 else DOWN

    def _navigate(self, dt, grid, target, random_mode=False, eaten_mode=False):
        """
        Tile-by-tile navigation.  At each tile centre, MDIRCT picks the
        next direction toward 'target' (or random if frightened).
        Cannot reverse direction (AND REVTAB,Y mask in assembly).
        Eaten ghosts move faster and can enter GHOST_AREA.
        """
        spd = self.speed * dt
        if eaten_mode:   spd *= 1.8
        if random_mode:  spd *= 0.6   # frightened ghosts are slower

        # Move toward committed next tile
        tx = float(self.ntc * CELL + CELL // 2)
        ty = float(self.ntr * CELL + CELL // 2)
        ddx = tx - self.px
        ddy = ty - self.py
        dist = math.hypot(ddx, ddy)

        if dist <= spd + 0.5:
            # Arrived – snap and choose next tile (MDIRCT)
            self.px, self.py = tx, ty
            self.col, self.row = self.ntc, self.ntr
            ndir = self._choose_dir(grid, self.ntc, self.ntr,
                                     target, self.dir,
                                     random_mode, eaten_mode)
            self.dir = ndir
            dc, dr = DELTA[ndir]
            self.ntc = self.ntc + dc
            self.ntr = self.ntr + dr
        else:
            s = spd / dist
            self.px += ddx * s
            self.py += ddy * s

        # Tunnel wrap
        if   self.px < 0:         self.px = float(SCREEN_W - 1)
        elif self.px >= SCREEN_W: self.px = 0.0

    def _choose_dir(self, grid, col, row, target, cur_dir,
                    random_mode=False, eaten_mode=False):
        """
        MDIRCT: enumerate allowed directions; filter out reverse;
        pick the one with minimum Manhattan distance to target.
        Random mode (frightened) picks randomly instead.
        """
        allowed = []
        for d in ALL_DIRS:
            dc, dr = DELTA[d]
            nc, nr = col + dc, row + dr
            if ghost_walkable(grid, nr, nc, eaten=eaten_mode):
                allowed.append(d)

        rev = OPPOSITE.get(cur_dir, NONE)
        choices = [d for d in allowed if d != rev]
        if not choices:
            choices = allowed
        if not choices:
            return cur_dir

        if random_mode or target is None:
            return random.choice(choices)

        tc, tr = target
        def manhattan(d):
            dc, dr = DELTA[d]
            return abs((col + dc) - tc) + abs((row + dr) - tr)
        return min(choices, key=manhattan)

    # ── targeting logic (MCHASE / SEEPAC / GOHOME) ───────────
    def _pick_target(self, pac, all_ghosts):
        """
        Each ghost has a unique targeting strategy (MONSTR state machine
        and MCHASE / SCHASE / GOHOME in PAC4.ASM):

          Blinky (0) – direct chase, always targets Pac-Man's tile
          Pinky  (1) – ambush: targets 4 tiles ahead of Pac-Man
          Inky   (2) – flanking: 2 tiles ahead reflected around Blinky
          Clyde  (3) – shy: chases when far (>8 tiles), scatters when close
        """
        if self.state == GS_SCATTER:
            return self.SCATTER_TARGETS[self.idx]

        pcol, prow = pac.tile()
        pdir = pac.dir if pac.dir != NONE else RIGHT
        pdx, pdy = DELTA[pdir]

        if self.idx == 0:   # Blinky: MCHASE – target = Pac-Man
            return (pcol, prow)

        elif self.idx == 1: # Pinky: 4 tiles ahead of Pac-Man
            return (pcol + pdx * 4, prow + pdy * 4)

        elif self.idx == 2: # Inky: 2-ahead mirrored around Blinky
            blinky = all_ghosts[0]
            bc, br = blinky.col, blinky.row
            ac, ar = pcol + pdx * 2, prow + pdy * 2
            return (2 * ac - bc, 2 * ar - br)

        else:               # Clyde: shy ghost (GOHOME when dist <= 8)
            dist = abs(self.col - pcol) + abs(self.row - prow)
            if dist > 8:
                return (pcol, prow)
            return self.SCATTER_TARGETS[self.idx]

    def tile(self):
        return self.col, self.row

    def draw(self, surface, fright_timer):
        cx, cy = int(self.px), int(self.py)
        r = CELL // 2 - 1
        if self.state == GS_EATEN:
            draw_ghost_eyes(surface, cx, cy, r, self.dir)
            return
        if self.state == GS_FRIGHTENED:
            flash = (fright_timer < 90) and ((fright_timer // 10) % 2 == 0)
            col   = FRIGHT2 if flash else FRIGHT
            draw_ghost_sprite(surface, cx, cy, r, col, self.dir, flash=flash)
        else:
            draw_ghost_sprite(surface, cx, cy, r, self.color, self.dir)


# ══════════════════════════════════════════════════════════════════
#  GAME  (PACMAN.ASM main loop + PAC1-PAC2 VBlank subroutines)
# ══════════════════════════════════════════════════════════════════

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(
            "PAC-MAN  –  Atari/Roklan 1982  ·  Python translation")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("monospace", 16, bold=True)
        self.sfont = pygame.font.SysFont("monospace", 12)
        self.high  = 0       # high score  (HISCTX in assembly)
        self._new_game()

    # ── game-level initialisation (NEWGAM / REINIT) ───────────
    def _new_game(self):
        self.score       = 0
        self.lives       = 3   # XPACP1 = 3
        self.level       = 1   # MAZCT1 initial value
        self.bonus_given = False
        self._new_level()

    # ── per-level initialisation (NEWBRD / SETUP / READY1) ────
    def _new_level(self):
        self.grid      = build_maze(MAZE_DEF)
        self.dots_left = sum(1 for row in MAZE_DEF for c in row if c in '.o')
        idx = min(self.level - 1, len(LEVEL_PARAMS) - 1)
        pspd, gspd, self.fright_total, delays = LEVEL_PARAMS[idx]
        self.pac         = PacMan(pspd)
        self.ghosts      = [Ghost(i, gspd, delays[i]) for i in range(4)]
        self.fright_timer  = 0       # FLITMR
        self.gulp_count    = 0       # GLPCNT
        self.fruit_active  = False
        self.fruit_shown   = set()   # which dot-count thresholds showed fruit
        self.fruit_timer   = 0
        self.fruit_score   = 0
        self.fruit_color   = (255, 0, 0)
        self.FRUIT_COL     = 13
        self.FRUIT_ROW     = 17
        self.popups        = []      # (text, px, py, lifetime_frames)
        self.power_vis     = True    # BLINKR: power pellet blink state
        self.blink_tick    = 0
        self.scatter_timer = 300     # CHASET initial value (~5 s scatter)
        self.chase_mode    = False   # MNTST2 / SCHASE flag
        self._state        = "READY"
        self._stimer       = 150     # READYF display timer (~2.5 s)

    # ── main loop ─────────────────────────────────────────────
    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self._events()
            self._update(dt)
            self._draw()
            pygame.display.flip()

    # ── event handling ────────────────────────────────────────
    _KEY_DIR = {
        pygame.K_UP: UP,   pygame.K_DOWN: DOWN,
        pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
        pygame.K_w: UP,    pygame.K_s: DOWN,
        pygame.K_a: LEFT,  pygame.K_d: RIGHT,
    }

    def _events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type != pygame.KEYDOWN: continue
            k = ev.key
            if k in (pygame.K_q, pygame.K_ESCAPE):
                pygame.quit(); sys.exit()
            if k == pygame.K_p:
                if   self._state == "PLAYING": self._state = "PAUSED"
                elif self._state == "PAUSED":  self._state = "PLAYING"
            if k == pygame.K_RETURN and self._state in ("GAMEOVER",):
                self._new_game()
            if k in self._KEY_DIR and self._state == "PLAYING":
                self.pac.set_dir(self._KEY_DIR[k])

    # ── per-frame update ─────────────────────────────────────
    def _update(self, dt):
        if self._state == "READY":
            self._stimer -= 1
            if self._stimer <= 0:
                self._state = "PLAYING"
            return

        if self._state == "DYING":
            self.pac.death_frame += 1
            if self.pac.death_frame >= 50:
                self._on_life_lost()
            return

        if self._state == "LEVEL_CLEAR":
            self._stimer -= 1
            if self._stimer <= 0:
                self.level += 1
                self._new_level()
            return

        if self._state in ("PAUSED", "GAMEOVER"):
            return

        # ── PLAYING ──────────────────────────────────────────
        self._update_scatter_chase()
        self.pac.update(dt, self.grid)
        for g in self.ghosts:
            g.update(dt, self.grid, self.pac, self.ghosts)
        # Fright timer countdown (FLITMR DEC in VBGAME)
        if self.fright_timer > 0:
            self.fright_timer -= 1
            if self.fright_timer == 0:
                for g in self.ghosts:
                    g.end_fright()
        self._blink()
        self._eat_dots()          # MUNCHY / DOTTST
        self._check_collisions()  # COLCHK
        self._update_fruit()      # FRUITY / SETFRT
        self._tick_popups()
        # CKBONS: bonus life at threshold
        if not self.bonus_given and self.score >= BONUS_LIFE_AT:
            self.lives += 1
            self.bonus_given = True

    # ── scatter/chase mode toggle (CHSSEQ / SETMAD) ──────────
    def _update_scatter_chase(self):
        self.scatter_timer -= 1
        if self.scatter_timer > 0: return
        self.chase_mode    = not self.chase_mode
        self.scatter_timer = 420 if self.chase_mode else 300
        ns = GS_CHASE if self.chase_mode else GS_SCATTER
        for g in self.ghosts:
            if g.state in (GS_CHASE, GS_SCATTER):
                g.state = ns

    # ── power pellet blink (BLINKR) ──────────────────────────
    def _blink(self):
        self.blink_tick += 1
        if self.blink_tick >= 30:
            self.blink_tick = 0
            self.power_vis = not self.power_vis

    # ── dot / power pellet consumption (MUNCHY / DOTTST) ─────
    def _eat_dots(self):
        col, row = self.pac.tile()
        if not (0 <= row < ROWS and 0 <= col < COLS): return
        t = self.grid[row][col]
        if t == DOT_CELL:
            self.grid[row][col] = EMPTY
            self.score     += 10
            self.dots_left -= 1
        elif t == POWER_CELL:
            self.grid[row][col] = EMPTY
            self.score     += 50
            self.dots_left -= 1
            self._activate_fright()   # DOTTST activates flight mode
        if self.dots_left <= 0:
            self._state  = "LEVEL_CLEAR"
            self._stimer = 120

    def _activate_fright(self):
        """DOTTST: sets FLASHC=1, GLPCNT=$FF, FLITMR, ORA #$80 per ghost."""
        self.fright_timer = self.fright_total
        self.gulp_count   = 0
        for g in self.ghosts:
            g.frighten(self.fright_total)

    # ── collision check (COLCHK / ZAPGST / PMDEAD) ───────────
    def _check_collisions(self):
        pcol, prow = self.pac.tile()
        for g in self.ghosts:
            gc, gr = g.tile()
            if abs(gc - pcol) > 1 or abs(gr - prow) > 1: continue
            if g.is_frightened:
                # ZAPGST: eat the ghost, award score
                score = GHOST_SCORES[min(self.gulp_count, 3)]
                self.gulp_count += 1
                self.score += score
                g.eat()
                self._add_popup(str(score), g.px, g.py)
            elif not g.is_eaten and g.state not in (GS_INACTIVE, GS_LEAVING):
                # PMDEAD: Pac-Man dies
                self._start_death()
                return

    def _start_death(self):
        """PMDEAD / VFIZZL: begin Pac-Man death animation."""
        self.pac.alive = False
        self.pac.death_frame = 0
        self.fright_timer = 0
        self._state = "DYING"

    def _on_life_lost(self):
        """RST1PG / VRESET: lose a life, reset or game-over."""
        self.lives -= 1
        if self.lives <= 0:
            self.high  = max(self.high, self.score)
            self._state = "GAMEOVER"
        else:
            self._reset_entities()
            self._state  = "READY"
            self._stimer = 120

    def _reset_entities(self):
        """SETUP / READY1: put Pac-Man and ghosts back at start positions."""
        idx = min(self.level - 1, len(LEVEL_PARAMS) - 1)
        pspd, gspd, _, delays = LEVEL_PARAMS[idx]
        self.pac         = PacMan(pspd)
        self.ghosts      = [Ghost(i, gspd, delays[i]) for i in range(4)]
        self.fright_timer = 0
        self.gulp_count   = 0

    # ── fruit logic (FRUITY / SETFRT / DFRTMR) ───────────────
    def _update_fruit(self):
        for threshold in (170, 70):
            if self.dots_left == threshold and threshold not in self.fruit_shown:
                self.fruit_shown.add(threshold)
                fi = min(self.level - 1, len(FRUITS) - 1)
                self.fruit_score = FRUITS[fi][1]
                self.fruit_color = FRUITS[fi][2]
                self.fruit_active = True
                self.fruit_timer  = 600   # ~10 s  (FDELAY × 2 in assembly)
                break
        if not self.fruit_active: return
        self.fruit_timer -= 1
        if self.fruit_timer <= 0:
            self.fruit_active = False
            return
        pcol, prow = self.pac.tile()
        if abs(pcol - self.FRUIT_COL) <= 1 and abs(prow - self.FRUIT_ROW) <= 1:
            self.score += self.fruit_score
            self._add_popup(str(self.fruit_score),
                            self.FRUIT_COL * CELL + CELL // 2,
                            self.FRUIT_ROW * CELL + CELL // 2)
            self.fruit_active = False

    # ── score popup helper ────────────────────────────────────
    def _add_popup(self, text, x, y):
        self.popups.append([text, float(x), float(y), 60])

    def _tick_popups(self):
        self.popups = [[t,x,y,n-1] for t,x,y,n in self.popups if n > 1]

    # ── rendering ────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BLACK)
        draw_maze(self.screen, self.grid, self.power_vis)
        if self.fruit_active:
            fx = self.FRUIT_COL * CELL + CELL // 2
            fy = self.FRUIT_ROW * CELL + CELL // 2
            pygame.draw.circle(self.screen, self.fruit_color, (fx, fy), 7)
        for g in self.ghosts:
            g.draw(self.screen, self.fright_timer)
        self.pac.draw(self.screen)
        for txt, x, y, _ in self.popups:
            s = self.sfont.render(txt, True, WHITE)
            self.screen.blit(s, (int(x)-s.get_width()//2,
                                  int(y)-s.get_height()//2))
        self._draw_hud()
        # Overlay messages
        if self._state == "READY":
            self._centered("READY!", ROWS//2, YELLOW)
        elif self._state == "GAMEOVER":
            self._centered("GAME  OVER", ROWS//2, RED)
            self._centered("PRESS ENTER TO PLAY AGAIN", ROWS//2+2, WHITE)
        elif self._state == "PAUSED":
            self._centered("PAUSED  –  P TO RESUME", ROWS//2, WHITE)
        elif self._state == "LEVEL_CLEAR":
            self._centered(f"LEVEL {self.level} CLEAR!", ROWS//2, YELLOW)

    def _draw_hud(self):
        """
        HUD strip below the maze.
        Mirrors the TEXT buffer rendered at the top/bottom of the Atari
        screen (1UP / 2UP scores, extra-life Pac-Man icons, HI score).
        """
        y0 = ROWS * CELL + 6
        sc = self.font.render(f"1UP  {self.score:06d}", True, HUD_C)
        hi = self.font.render(f"HI  {self.high:06d}",  True, HUD_C)
        self.screen.blit(sc, (8,          y0))
        self.screen.blit(hi, (SCREEN_W - hi.get_width() - 8, y0))
        # Extra-life icons (XPACP1 display, UDXPACS subroutine)
        for i in range(max(0, self.lives - 1)):
            lx = 8 + i * (CELL + 2)
            ly = y0 + 28
            draw_pac(self.screen, lx + CELL//2, ly + CELL//2,
                     CELL//2 - 2, 0, 20)
        # Level fruit icon
        fi  = min(self.level - 1, len(FRUITS) - 1)
        fcl = FRUITS[fi][2]
        pygame.draw.circle(self.screen, fcl,
                            (SCREEN_W - 20, y0 + 34), 7)
        lv = self.sfont.render(f"LV{self.level}", True, ORANGE)
        self.screen.blit(lv, (SCREEN_W//2 - lv.get_width()//2, y0 + 28))

    def _centered(self, text, row, color):
        s = self.font.render(text, True, color)
        self.screen.blit(s, (SCREEN_W//2 - s.get_width()//2, row * CELL))


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Game().run()
