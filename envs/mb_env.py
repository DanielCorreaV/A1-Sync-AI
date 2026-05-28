"""
envs/mb_env.py
--------------
Entorno Gymnasium para el corredor Madre Bernarda — Zona Sur de Cartagena.

CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL:
  1. Modelo de llegada de pasajeros reemplazado por distribución de Poisson
     con tasas realistas por franja horaria, en vez de randint fijo.
     Fuente de referencia: Transcaribe opera con headways de 4-8 min en pico,
     lo que implica que en 5 min se acumulan entre 3-8 pax por parada en pico
     y 1-3 en valle. El randint(6,12) original era el doble de lo razonable.
  2. Se introducen ARRIVAL_PROFILES: tasas lambda de Poisson por franja horaria.
     Cada ruta tiene su propio multiplicador de demanda relativa para
     reflejar que A108 (Blas de Lezo) tiene más demanda que A105 (Las Gaviotas).
  3. SATURATION_THRESHOLD bajado de 40 a 25 para que el agente reaccione
     antes, acorde a la nueva escala de llegadas.
  4. DISPATCH_CAPACITY bajado de 50 a 40 (capacidad real articulado Transcaribe).
  5. MAX_QUEUE bajado de 200 a 120 para mantener la normalización coherente.
  6. La penalización W_WASTE se vuelve más suave (0.15) para no castigar
     tanto despachos preventivos en horas de transición.
"""

import random
import numpy as np
from typing import Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bus import Bus
from core.weather_module import WeatherModule
from core.traffic_module import TrafficModule

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM_AVAILABLE = True
    except ImportError:
        _GYM_AVAILABLE = False

        class _FakeSpace:
            def __init__(self, n=None, shape=None, dtype=None, low=None, high=None):
                self.n = n; self.shape = shape; self.dtype = dtype
            def sample(self):
                if self.n: return random.randint(0, self.n - 1)
                return np.zeros(self.shape, dtype=self.dtype)

        class spaces:
            @staticmethod
            def Discrete(n): return _FakeSpace(n=n)
            @staticmethod
            def Box(low, high, shape, dtype): return _FakeSpace(shape=shape, dtype=dtype)

        class gym:
            class Env:
                metadata = {}


# -----------------------------------------------------------------------
# Constantes del entorno
# -----------------------------------------------------------------------
ROUTES      = ["A103", "A104", "A105", "A107", "A108"]
N_ROUTES    = len(ROUTES)
ROUTE_INDEX = {r: i for i, r in enumerate(ROUTES)}

FLEET_SIZE     = 13
PORTAL_INITIAL = 5
MIN_PLATFORM   = 4

TICK_MINUTES = 5
START_MINUTES = 300    # 05:00
END_MINUTES   = 1380   # 23:00
MAX_STEPS     = (END_MINUTES - START_MINUTES) // TICK_MINUTES  # 216 pasos

DISPATCH_CAPACITY    = 40    # pasajeros que absorbe un articulado (era 50)
SATURATION_THRESHOLD = 25    # fila mínima para considerar ruta saturada (era 40)
MAX_QUEUE            = 120.0 # valor de normalización (era 200)
MAX_TRAFFIC          = 5.0
MAX_WAIT_MIN         = 60.0  # normalización: 60 min de espera = saturación máxima

# Pesos de la función de recompensa
W_QUEUE = 0.4
W_WASTE = 0.15   # más suave que antes (era 0.3) — el sistema no debe temer despachos preventivos
W_BONUS = 0.5
W_DRIFT = 0.2

# Efecto vaso comunicante A103 ↔ A104
SISTER_ROUTES = {"A103": "A104", "A104": "A103"}
SISTER_RELIEF = 0.10

# -----------------------------------------------------------------------
# MODELO DE LLEGADA REALISTA
# -----------------------------------------------------------------------
# Tasas lambda de Poisson (pasajeros / tick de 5 min / parada) por franja.
# Basado en que un articulado en pico toma ~40 pax cada 5 min en la parada
# más cargada; las paradas intermedias reciben menos.
#
# Franjas:
#   "madrugada"  05:00–06:29  — muy poca demanda, trabajadores nocturnos
#   "mañana"     06:30–08:59  — hora pico mañana
#   "media"      09:00–11:59  — valle suave
#   "mediodia"   12:00–13:59  — pico moderado (almuerzo, colegios)
#   "tarde"      14:00–16:29  — valle
#   "pico_tarde" 16:30–19:29  — hora pico tarde (el más intenso)
#   "noche"      19:30–22:59  — caída progresiva
#
ARRIVAL_PROFILES: list[tuple[int, int, float]] = [
    # (inicio_min, fin_min, lambda_base)
    (300,  389, 1.2),   # madrugada
    (390,  539, 4.5),   # pico mañana
    (540,  719, 2.0),   # media mañana / valle
    (720,  839, 3.2),   # mediodía
    (840,  989, 1.8),   # tarde valle
    (990, 1169, 5.5),   # pico tarde — el más intenso del día
    (1170, 1380, 1.5),  # noche
]

# Multiplicador de demanda relativa por ruta
# A108 (Blas de Lezo) y A103 (Olaya) son las más cargadas históricamente
ROUTE_DEMAND_FACTOR: dict[str, float] = {
    "A103": 1.10,   # Olaya Herrera — alta densidad residencial
    "A104": 0.90,   # San Fernando — demanda media
    "A105": 0.75,   # Las Gaviotas — menor densidad
    "A107": 0.95,   # El Pozón — demanda media-alta
    "A108": 1.15,   # Blas de Lezo — terminal de alta rotación
}


def _get_arrival_lambda(time_minutes: int) -> float:
    """Devuelve la tasa lambda de Poisson para el tick actual."""
    for start, end, lam in ARRIVAL_PROFILES:
        if start <= time_minutes < end:
            return lam
    return 1.0  # fallback


class MBEnv(gym.Env):
    """
    Entorno Gymnasium — Corredor Madre Bernarda, Zona Sur de Cartagena.
    """

    metadata = {"render_modes": ["human", "ansi"]}
    OBS_DIM  = 22  # +5 por wait_ticks normalizados (uno por ruta)

    def __init__(self, month: int = 10, seed: int | None = None, verbose: bool = False):
        super().__init__()
        self.month    = month
        self.seed_val = seed
        self.verbose  = verbose

        self.action_space      = spaces.Discrete(N_ROUTES + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

        self._fleet:    list[Bus]       = []
        self._routes:   dict[str, dict] = {}
        self._weather:  WeatherModule | None = None
        self._traffic:  TrafficModule | None = None
        self._time:     int = START_MINUTES
        self._step_count:     int = 0
        self._episode_rewards: list[float] = []

    # ═══════════════════════════════════════════════════════════════════
    # API GYMNASIUM
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_val = seed
        if self.seed_val is not None:
            random.seed(self.seed_val)
            np.random.seed(self.seed_val)

        self._weather = WeatherModule(month=self.month)
        self._traffic = TrafficModule(weather_rain_factor=self._weather.rain_traffic_factor)

        self._fleet = [Bus(i) for i in range(FLEET_SIZE)]
        for i in range(PORTAL_INITIAL):
            self._fleet[i].state = "portal"

        # Fila inicial realista: 2-4 pasajeros residuales nocturnos
        # wait_ticks: ticks consecutivos con al menos 1 pasajero esperando
        self._routes = {r: {"fila": random.randint(2, 4), "wait_ticks": 1} for r in ROUTES}

        self._time            = START_MINUTES
        self._step_count      = 0
        self._episode_rewards = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  NUEVO EPISODIO | mes={self.month} | {self._weather.weather.upper()}")
            print(f"  bloqueo={self._traffic.is_blocked} | "
                  f"p_lluvia={self._weather.rain_probability:.0%}")
            print(f"{'='*60}")

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        self._tick_world()
        reward = self._apply_action(action)
        self._refill_platform()

        self._time       += TICK_MINUTES
        self._step_count += 1
        self._episode_rewards.append(reward)

        obs        = self._get_obs()
        terminated = self._time >= END_MINUTES
        truncated  = False
        info       = self._get_info()
        info["action"] = action
        info["reward"] = reward

        if self.verbose:
            self._print_step(action, reward)

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> str | None:
        h = self._time // 60
        m = self._time % 60

        # Indicador de franja horaria
        lam   = _get_arrival_lambda(self._time)
        if lam >= 5.0:
            franja = "PICO ALTO"
        elif lam >= 3.0:
            franja = "Pico moderado"
        elif lam >= 2.0:
            franja = "Valle"
        else:
            franja = "Madrugada/noche"

        lines = [
            f"\n--- Tick {self._step_count:03d} | {h:02d}:{m:02d} | {franja} ---",
            f"Clima: {self._weather.weather if self._weather else 'N/A'} | "
            f"Bloqueo: {self._traffic.is_blocked if self._traffic else 'N/A'} | "
            f"Traffic mult: {self._get_traffic_mult():.2f}",
            "Rutas:",
        ]
        for r in ROUTES:
            fila = self._routes[r]["fila"]
            bar       = "█" * min(int(fila / 3), 30)
            wait_min  = self._routes[r]["wait_ticks"] * TICK_MINUTES
            wait_str  = f" | espera ~{wait_min}min" if wait_min > 0 else ""
            lines.append(f"  {r}: {fila:4d} pasajeros  {bar}{wait_str}")
        lines.append("Flota:")
        counts = self._fleet_counts()
        lines.append(
            f"  station={counts['station']} | onGoing={counts['onGoing']} | "
            f"portal={counts['portal']} | rest={counts['rest']} | "
            f"stranded={counts['stranded']}"
        )
        output = "\n".join(lines)
        if mode == "human":
            print(output)
            return None
        return output

    def close(self):
        pass

    # ═══════════════════════════════════════════════════════════════════
    # LÓGICA INTERNA DEL MUNDO
    # ═══════════════════════════════════════════════════════════════════

    def _tick_world(self) -> None:
        mult = self._get_traffic_mult()
        for bus in self._fleet:
            bus.update(TICK_MINUTES, mult)

        # Llegada de pasajeros con distribución de Poisson por franja horaria
        lam_base = _get_arrival_lambda(self._time)

        # Factor de lluvia: reduce demanda porque la gente migra a mototaxis/informal
        weather_factor = self._weather.demand_factor if self._weather else 1.0

        for route, route_data in self._routes.items():
            # Lambda ajustado por ruta y clima
            lam = lam_base * ROUTE_DEMAND_FACTOR[route] * weather_factor

            # Poisson da llegadas discretas con varianza natural — más realista
            # que un randint que tiene distribución uniforme
            arrive = int(np.random.poisson(lam))
            route_data["fila"] = min(route_data["fila"] + arrive, int(MAX_QUEUE))

            # Acumular tiempo de espera: si hay pasajeros en la parada, el reloj corre
            if route_data["fila"] > 0:
                route_data["wait_ticks"] += 1
            else:
                route_data["wait_ticks"] = 0

    def _apply_action(self, action: int) -> float:
        reward = 0.0

        total_queue    = sum(d["fila"] for d in self._routes.values())
        queue_pressure = total_queue / (MAX_QUEUE * N_ROUTES)
        reward        -= W_QUEUE * queue_pressure

        available = len([b for b in self._fleet if b.is_available])
        if available == 0:
            reward -= W_DRIFT

        if action > 0:
            target_route = ROUTES[action - 1]
            bus          = self._get_next_available_bus()

            if bus is not None:
                fila = self._routes[target_route]["fila"]

                if fila >= SATURATION_THRESHOLD:
                    self._do_dispatch(bus, target_route)
                    reward += W_BONUS * min(fila / MAX_QUEUE, 1.0)
                else:
                    self._do_dispatch(bus, target_route)
                    reward -= W_WASTE * (1.0 - fila / SATURATION_THRESHOLD)

        return float(np.clip(reward, -2.0, 2.0))

    def _do_dispatch(self, bus: Bus, route: str) -> None:
        bus.dispatch()
        self._routes[route]["fila"] = max(0, self._routes[route]["fila"] - DISPATCH_CAPACITY)
        # Un despacho reinicia el reloj de espera de la ruta
        self._routes[route]["wait_ticks"] = 0

        if route in SISTER_ROUTES:
            sister = SISTER_ROUTES[route]
            relief = int(self._routes[sister]["fila"] * SISTER_RELIEF)
            self._routes[sister]["fila"] = max(0, self._routes[sister]["fila"] - relief)

    def _refill_platform(self) -> None:
        available = len([b for b in self._fleet if b.is_available])
        if available < MIN_PLATFORM:
            portal_bus = next((b for b in self._fleet if b.is_in_portal), None)
            if portal_bus:
                portal_bus.state = "station"

    # ═══════════════════════════════════════════════════════════════════
    # OBSERVACIÓN
    # ═══════════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        counts = self._fleet_counts()
        hour   = self._time / 60.0
        mult   = self._get_traffic_mult()
        total  = sum(d["fila"] for d in self._routes.values())
        avail  = counts["station"]
        sat    = max(1, sum(1 for d in self._routes.values()
                            if d["fila"] >= SATURATION_THRESHOLD))

        obs = np.array([
            *[self._routes[r]["fila"] / MAX_QUEUE for r in ROUTES],
            counts["station"]  / FLEET_SIZE,
            counts["onGoing"]  / FLEET_SIZE,
            counts["portal"]   / FLEET_SIZE,
            counts["rest"]     / FLEET_SIZE,
            counts["stranded"] / FLEET_SIZE,
            (self._time - START_MINUTES) / (END_MINUTES - START_MINUTES),
            1.0 if (self._traffic and self._traffic.is_rush_hour(hour)) else 0.0,
            min(mult / MAX_TRAFFIC, 1.0),
            1.0 if (self._weather and self._weather.is_raining) else 0.0,
            1.0 if (self._traffic and self._traffic.is_blocked) else 0.0,
            total / (MAX_QUEUE * N_ROUTES),
            min(avail / sat, 1.0),
            # 17-21: tiempo de espera normalizado por ruta (wait_ticks * TICK_MINUTES / MAX_WAIT_MIN)
            *[min(self._routes[r]["wait_ticks"] * TICK_MINUTES / MAX_WAIT_MIN, 1.0) for r in ROUTES],
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict[str, Any]:
        counts    = self._fleet_counts()
        total_q   = sum(d["fila"] for d in self._routes.values())
        ep_reward = sum(self._episode_rewards) if self._episode_rewards else 0.0

        return {
            "time_minutes":   self._time,
            "step":           self._step_count,
            "weather":        self._weather.weather if self._weather else "N/A",
            "blocked":        self._traffic.is_blocked if self._traffic else False,
            "traffic_mult":   self._get_traffic_mult(),
            "queues":         {r: self._routes[r]["fila"] for r in ROUTES},
            "total_queue":    total_q,
            "fleet_counts":   counts,
            "episode_reward": ep_reward,
            "arrival_lambda": _get_arrival_lambda(self._time),
            "wait_minutes":   {r: self._routes[r]["wait_ticks"] * TICK_MINUTES for r in ROUTES},
        }

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _get_traffic_mult(self) -> float:
        if self._traffic is None:
            return 1.0
        return self._traffic.get_multiplier(self._time / 60.0)

    def _get_next_available_bus(self) -> Bus | None:
        return next((b for b in self._fleet if b.is_available), None)

    def _fleet_counts(self) -> dict[str, int]:
        counts = {"station": 0, "onGoing": 0, "portal": 0, "rest": 0, "stranded": 0}
        for bus in self._fleet:
            if bus.state in counts:
                counts[bus.state] += 1
        return counts

    def _print_step(self, action: int, reward: float) -> None:
        h   = self._time // 60
        m   = self._time % 60
        act = "no-op" if action == 0 else f"→{ROUTES[action - 1]}"
        print(f"[{h:02d}:{m:02d}] step={self._step_count:3d} | "
              f"action={act:6s} | reward={reward:+.3f} | "
              f"queues={[self._routes[r]['fila'] for r in ROUTES]}")