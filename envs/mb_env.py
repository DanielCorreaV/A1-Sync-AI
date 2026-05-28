"""
envs/mb_env.py
--------------
Entorno Gymnasium para el corredor Madre Bernarda — Zona Sur de Cartagena.

Implementa la API estándar de Gymnasium:
    env = MBEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

═══════════════════════════════════════════════════════════════
ESPACIO DE ACCIONES
═══════════════════════════════════════════════════════════════
Discrete(6):
    0 → no despachar ningún bus este tick
    1 → despachar el siguiente bus disponible a ruta A103
    2 → despachar el siguiente bus disponible a ruta A104
    3 → despachar el siguiente bus disponible a ruta A105
    4 → despachar el siguiente bus disponible a ruta A107
    5 → despachar el siguiente bus disponible a ruta A108

Si la acción pide despachar pero no hay bus disponible, se ejecuta
como acción 0 (no-op) sin penalización extra.

═══════════════════════════════════════════════════════════════
ESPACIO DE OBSERVACIONES  (vector numpy float32, dim=17)
═══════════════════════════════════════════════════════════════
Índices 0-4  : fila normalizada por ruta (fila / MAX_QUEUE)
               orden: A103, A104, A105, A107, A108
Índice  5    : buses disponibles en plataforma / FLEET_SIZE
Índice  6    : buses en ruta / FLEET_SIZE
Índice  7    : buses en portal / FLEET_SIZE
Índice  8    : buses en descanso / FLEET_SIZE
Índice  9    : buses varados / FLEET_SIZE
Índice  10   : hora normalizada (time_minutes / 1440)
Índice  11   : ¿es hora pico? (0.0 o 1.0)
Índice  12   : multiplicador de tráfico normalizado (mult / MAX_TRAFFIC_MULT)
Índice  13   : ¿llueve? (0.0 o 1.0)
Índice  14   : ¿hay bloqueo vial? (0.0 o 1.0)
Índice  15   : fila total normalizada (suma filas / (MAX_QUEUE * N_ROUTES))
Índice  16   : capacidad de respuesta (buses_disponibles / max(1, rutas_saturadas))

═══════════════════════════════════════════════════════════════
RECOMPENSA
═══════════════════════════════════════════════════════════════
La recompensa está diseñada para que el agente aprenda a:
    (a) reducir la acumulación de pasajeros en paradas
    (b) no desperdiciar buses en rutas con poca demanda
    (c) mantener la plataforma con cobertura mínima

  r = - w_queue  * presión_de_cola
      - w_waste  * penalización_despacho_vacío
      + w_bonus  * bonus_despacho_efectivo
      - w_drift  * penalización_plataforma_vacía
"""

import random
import numpy as np
from typing import Any

# Importamos los módulos del proyecto
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bus import Bus
from core.weather_module import WeatherModule
from core.traffic_module import TrafficModule

# -----------------------------------------------------------------------
# Intentar importar gymnasium; si no está disponible (entorno de desarrollo),
# usar la clase base mínima compatible para que el resto del código funcione.
# -----------------------------------------------------------------------
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

        # Shim mínimo para desarrollo sin gymnasium instalado
        class _FakeSpace:
            def __init__(self, n=None, shape=None, dtype=None, low=None, high=None):
                self.n = n
                self.shape = shape
                self.dtype = dtype

            def sample(self):
                if self.n:
                    return random.randint(0, self.n - 1)
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
ROUTES         = ["A103", "A104", "A105", "A107", "A108"]
N_ROUTES       = len(ROUTES)
ROUTE_INDEX    = {r: i for i, r in enumerate(ROUTES)}

FLEET_SIZE     = 25       # total de buses en el sistema
PORTAL_INITIAL = 17       # buses que inician en portal
MIN_PLATFORM   = 4        # mínimo de buses en plataforma antes de traer del portal

TICK_MINUTES   = 5        # duración de cada paso de simulación (minutos)
START_MINUTES  = 300      # inicio del día: 5:00 AM (300 min desde medianoche)
END_MINUTES    = 1380     # fin del día: 11:00 PM (1380 min)
MAX_STEPS      = (END_MINUTES - START_MINUTES) // TICK_MINUTES  # 216 pasos/día

DISPATCH_CAPACITY = 50    # pasajeros que absorbe un despacho
SATURATION_THRESHOLD = 30 # fila mínima para considerar una ruta saturada
MAX_QUEUE      = 200.0    # valor de normalización para las filas
MAX_TRAFFIC    = 5.0      # valor de normalización para el multiplicador

# Pesos de la función de recompensa
W_QUEUE  = 0.4   # penalización por presión de cola
W_WASTE  = 0.3   # penalización por despacho a ruta sin demanda suficiente
W_BONUS  = 0.5   # bonus por despacho efectivo (reduce cola saturada)
W_DRIFT  = 0.2   # penalización por dejar la plataforma sin buses

# Efecto "vaso comunicante" A103↔A104 (Efecto Mandela documentado en el código original)
SISTER_ROUTES  = {"A103": "A104", "A104": "A103"}
SISTER_RELIEF  = 0.10   # 10% de reducción en la ruta hermana al despachar


class MBEnv(gym.Env):
    """
    Entorno Gymnasium — Corredor Madre Bernarda, Zona Sur de Cartagena.

    Parámetros
    ----------
    month : int
        Mes del año que se va a simular (1-12). Determina la probabilidad
        de lluvia y con ello el comportamiento del clima y el tráfico.
    seed : int | None
        Semilla de aleatoriedad. Fijarlo garantiza reproducibilidad total.
    verbose : bool
        Si True, imprime un resumen de cada tick en consola.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    OBS_DIM = 17

    def __init__(self, month: int = 10, seed: int | None = None, verbose: bool = False):
        super().__init__()
        self.month   = month
        self.seed_val = seed
        self.verbose = verbose

        # Espacios de acción y observación (API Gymnasium)
        self.action_space      = spaces.Discrete(N_ROUTES + 1)   # 0=no-op, 1-5=rutas
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32
        )

        # Estado interno (se inicializa en reset)
        self._fleet:    list[Bus]         = []
        self._routes:   dict[str, dict]   = {}
        self._weather:  WeatherModule | None = None
        self._traffic:  TrafficModule | None = None
        self._time:     int               = START_MINUTES
        self._step_count: int             = 0
        self._episode_rewards: list[float] = []

    # ═══════════════════════════════════════════════════════════════════
    # API GYMNASIUM
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reinicia el entorno para un nuevo episodio (día de simulación).
        Genera clima y condición de bloqueo nuevos para el día.
        """
        if seed is not None:
            self.seed_val = seed
        if self.seed_val is not None:
            random.seed(self.seed_val)
            np.random.seed(self.seed_val)

        # Condiciones del día
        self._weather = WeatherModule(month=self.month)
        self._traffic = TrafficModule(weather_rain_factor=self._weather.rain_traffic_factor)

        # Flota
        self._fleet = [Bus(i) for i in range(FLEET_SIZE)]
        for i in range(PORTAL_INITIAL):
            self._fleet[i].state = "portal"
        # Los buses restantes (FLEET_SIZE - PORTAL_INITIAL) inician en station

        # Rutas — fila inicial de 5 pasajeros (demanda residual nocturna)
        self._routes = {r: {"fila": 5} for r in ROUTES}

        # Reloj
        self._time       = START_MINUTES
        self._step_count = 0
        self._episode_rewards = []

        obs  = self._get_obs()
        info = self._get_info()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  NUEVO EPISODIO | mes={self.month} | {self._weather.weather.upper()}")
            print(f"  bloqueo={self._traffic.is_blocked} | "
                  f"p_lluvia={self._weather.rain_probability:.0%}")
            print(f"{'='*60}")

        return obs, info

    def step(self, action: int):
        """
        Ejecuta un tick de simulación y aplica la acción del agente.

        Args:
            action : entero 0-5 (ver definición al inicio del archivo).

        Returns:
            obs        : vector de observación (numpy float32)
            reward     : recompensa escalar
            terminated : True si el episodio terminó (fin del día)
            truncated  : False (no usamos límite de pasos distinto al fin de día)
            info       : diccionario con métricas auxiliares
        """
        # 1. Avanzar el mundo
        self._tick_world()

        # 2. Aplicar acción del agente
        reward = self._apply_action(action)

        # 3. Logística automática de portal → plataforma
        self._refill_platform()

        # 4. Avanzar reloj y contador
        self._time       += TICK_MINUTES
        self._step_count += 1

        self._episode_rewards.append(reward)

        # 5. Construir retorno
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
        """Renderiza el estado actual en consola."""
        h = self._time // 60
        m = self._time % 60
        lines = [
            f"\n--- Tick {self._step_count:03d} | {h:02d}:{m:02d} ---",
            f"Clima: {self._weather.weather if self._weather else 'N/A'} | "
            f"Bloqueo: {self._traffic.is_blocked if self._traffic else 'N/A'} | "
            f"Traffic mult: {self._get_traffic_mult():.2f}",
            "Rutas:",
        ]
        for r in ROUTES:
            fila = self._routes[r]["fila"]
            bar  = "█" * min(int(fila / 5), 30)
            lines.append(f"  {r}: {fila:4d} pasajeros  {bar}")
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
        """
        Avanza la física del mundo: actualiza buses y acumula demanda en paradas.
        Se ejecuta ANTES de que el agente tome su acción.
        """
        mult = self._get_traffic_mult()

        # Actualizar cada bus
        for bus in self._fleet:
            bus.update(TICK_MINUTES, mult)

        # Llegada de pasajeros a paradas
        hour = self._time / 60.0
        rush = self._traffic.is_rush_hour(hour) if self._traffic else False

        for route_data in self._routes.values():
            if rush:
                arrive = random.randint(6, 12)
            else:
                arrive = random.randint(1, 4)
            # Reducción por lluvia: migración al transporte informal
            arrive = int(arrive * self._weather.demand_factor) if self._weather else arrive
            route_data["fila"] = min(route_data["fila"] + arrive, int(MAX_QUEUE))

    def _apply_action(self, action: int) -> float:
        """
        Aplica la decisión del agente y calcula la recompensa.

        action 0    : no despachar
        action 1-5  : despachar a ROUTES[action-1]
        """
        reward = 0.0

        # Componente negativa base: presión acumulada en todas las colas
        total_queue    = sum(d["fila"] for d in self._routes.values())
        saturated      = sum(1 for d in self._routes.values() if d["fila"] >= SATURATION_THRESHOLD)
        queue_pressure = total_queue / (MAX_QUEUE * N_ROUTES)
        reward        -= W_QUEUE * queue_pressure

        # Penalización por plataforma vacía (sistema sin capacidad de respuesta)
        available = len([b for b in self._fleet if b.is_available])
        if available == 0:
            reward -= W_DRIFT

        if action == 0:
            # No despachar: no hay bonus ni penalización adicional
            pass
        else:
            target_route = ROUTES[action - 1]
            bus = self._get_next_available_bus()

            if bus is None:
                # El agente pidió despachar pero no hay bus: acción inválida silenciosa
                pass
            else:
                fila = self._routes[target_route]["fila"]

                if fila >= SATURATION_THRESHOLD:
                    # Despacho efectivo: la ruta lo necesitaba
                    self._do_dispatch(bus, target_route)
                    reward += W_BONUS * min(fila / MAX_QUEUE, 1.0)
                else:
                    # Despacho desperdiciado: la ruta no está saturada
                    self._do_dispatch(bus, target_route)
                    reward -= W_WASTE * (1.0 - fila / SATURATION_THRESHOLD)

        return float(np.clip(reward, -2.0, 2.0))

    def _do_dispatch(self, bus: Bus, route: str) -> None:
        """Despacha el bus a la ruta y aplica efectos sobre las colas."""
        bus.dispatch()

        # Reducir fila de la ruta objetivo
        self._routes[route]["fila"] = max(0, self._routes[route]["fila"] - DISPATCH_CAPACITY)

        # Efecto vaso comunicante A103↔A104
        if route in SISTER_ROUTES:
            sister = SISTER_ROUTES[route]
            relief = int(self._routes[sister]["fila"] * SISTER_RELIEF)
            self._routes[sister]["fila"] = max(0, self._routes[sister]["fila"] - relief)

    def _refill_platform(self) -> None:
        """
        Logística automática: si la plataforma cae por debajo del mínimo,
        traer un bus del portal. Esto no es decisión del agente.
        """
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
        sat    = max(1, sum(1 for d in self._routes.values() if d["fila"] >= SATURATION_THRESHOLD))

        obs = np.array([
            # 0-4: filas por ruta normalizadas
            *[self._routes[r]["fila"] / MAX_QUEUE for r in ROUTES],
            # 5-9: distribución de flota
            counts["station"]  / FLEET_SIZE,
            counts["onGoing"]  / FLEET_SIZE,
            counts["portal"]   / FLEET_SIZE,
            counts["rest"]     / FLEET_SIZE,
            counts["stranded"] / FLEET_SIZE,
            # 10: hora del día normalizada
            (self._time - START_MINUTES) / (END_MINUTES - START_MINUTES),
            # 11: hora pico (binaria)
            1.0 if (self._traffic and self._traffic.is_rush_hour(hour)) else 0.0,
            # 12: multiplicador de tráfico normalizado
            min(mult / MAX_TRAFFIC, 1.0),
            # 13: lluvia (binaria)
            1.0 if (self._weather and self._weather.is_raining) else 0.0,
            # 14: bloqueo vial (binaria)
            1.0 if (self._traffic and self._traffic.is_blocked) else 0.0,
            # 15: fila total normalizada
            total / (MAX_QUEUE * N_ROUTES),
            # 16: capacidad de respuesta
            min(avail / sat, 1.0),
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict[str, Any]:
        counts     = self._fleet_counts()
        total_q    = sum(d["fila"] for d in self._routes.values())
        ep_reward  = sum(self._episode_rewards) if self._episode_rewards else 0.0

        return {
            "time_minutes":    self._time,
            "step":            self._step_count,
            "weather":         self._weather.weather if self._weather else "N/A",
            "blocked":         self._traffic.is_blocked if self._traffic else False,
            "traffic_mult":    self._get_traffic_mult(),
            "queues":          {r: self._routes[r]["fila"] for r in ROUTES},
            "total_queue":     total_q,
            "fleet_counts":    counts,
            "episode_reward":  ep_reward,
        }

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _get_traffic_mult(self) -> float:
        if self._traffic is None:
            return 1.0
        hour = self._time / 60.0
        return self._traffic.get_multiplier(hour)

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
