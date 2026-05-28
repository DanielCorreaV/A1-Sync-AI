# agents/Fleet_optimization_agent.py
#
# CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL:
#   1. FloatOptimizationAgentAdapter unificado (ya no existe duplicado en main.py).
#      Combina el cooldown de main.py con la conciencia operativa de agents/.
#   2. La predicción de demanda ahora realmente influye en la decisión de despacho.
#   3. La consulta de buses disponibles usa directamente obs[5] en vez de re-leer
#      telemetría para el mismo dato que ya tiene el vector de observación.
#   4. Telemetría sigue usándose para la orden de ejecución (separación de roles).

import os
import pickle
import random
import numpy as np
from collections import defaultdict

from core.fipa import ACLMessage, FipaPerformative

DEFAULT_QTABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qtable_checkpoint.pkl"
)

QUEUE_BINS = [0.0, 0.15, 0.40, 0.70, 1.01]
AVAIL_BINS = [0.0, 0.08, 0.20, 0.40, 1.01]
N_ROUTES   = 5


# ──────────────────────────────────────────────────────────────────────
# CEREBRO MATEMÁTICO — Q-LEARNING  (sin cambios en la lógica de RL)
# ──────────────────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(
        self,
        n_actions:     int   = 6,
        alpha:         float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
        qtable_path:   str   = DEFAULT_QTABLE_PATH,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.qtable_path   = qtable_path

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.episode_count    = 0
        self.total_steps      = 0
        self.episode_rewards: list[float] = []

        self.load()

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state_key = self._discretize(obs)
        return int(np.argmax(self.q_table[state_key]))

    def update(
        self,
        obs:        np.ndarray,
        action:     int,
        reward:     float,
        next_obs:   np.ndarray,
        terminated: bool,
    ) -> None:
        s  = self._discretize(obs)
        s_ = self._discretize(next_obs)

        q_current = self.q_table[s][action]
        q_future  = 0.0 if terminated else float(np.max(self.q_table[s_]))
        td_target = reward + self.gamma * q_future
        self.q_table[s][action] += self.alpha * (td_target - q_current)
        self.total_steps += 1

    def end_episode(self, episode_reward: float) -> None:
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self) -> None:
        checkpoint = {
            "q_table":         dict(self.q_table),
            "epsilon":         self.epsilon,
            "episode_count":   self.episode_count,
            "total_steps":     self.total_steps,
            "episode_rewards": self.episode_rewards,
        }
        os.makedirs(os.path.dirname(self.qtable_path) or ".", exist_ok=True)
        with open(self.qtable_path, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"[QLearningAgent] Checkpoint guardado → {self.qtable_path}")

    def load(self) -> None:
        if not os.path.exists(self.qtable_path):
            print(f"[QLearningAgent] Sin checkpoint en {self.qtable_path}. Iniciando desde cero.")
            return
        with open(self.qtable_path, "rb") as f:
            checkpoint = pickle.load(f)

        loaded = checkpoint.get("q_table", {})

        # Verificar compatibilidad: si las claves tienen longitud distinta a la
        # tupla que genera _discretize ahora (6 elementos), el checkpoint es de
        # otra versión del entorno y se descarta para evitar que el agente
        # no reconozca ningún estado y siempre devuelva acción 0.
        EXPECTED_KEY_LEN = 6
        sample_keys = list(loaded.keys())[:5]
        if sample_keys and len(sample_keys[0]) != EXPECTED_KEY_LEN:
            print(f"[QLearningAgent] ADVERTENCIA: checkpoint incompatible "
                  f"(clave len={len(sample_keys[0])}, esperado={EXPECTED_KEY_LEN}). "
                  f"Descartando y empezando desde cero.")
            return

        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64),
            {k: np.array(v) for k, v in loaded.items()}
        )
        self.epsilon         = checkpoint.get("epsilon",        self.epsilon)
        self.episode_count   = checkpoint.get("episode_count",  0)
        self.total_steps     = checkpoint.get("total_steps",    0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        print(f"[QLearningAgent] Checkpoint cargado | "
              f"estados={len(self.q_table):,} | episodios={self.episode_count}")

    def _discretize(self, obs: np.ndarray) -> tuple:
        # Usa solo los primeros 17 índices del obs para mantener compatibilidad
        # con checkpoints existentes. wait_times (obs[17:22]) se usa en el
        # coordinador para decisiones de negocio, pero no como clave de estado QL
        # hasta que se re-entrene desde cero con el nuevo espacio.
        queues = obs[0:N_ROUTES]
        avail  = float(obs[5])
        rush   = int(round(float(obs[11])))
        rain   = int(round(float(obs[13])))
        block  = int(round(float(obs[14])))

        max_q_route = int(np.argmax(queues))
        max_q_val   = float(np.max(queues))
        max_q_bin   = int(np.digitize(max_q_val, QUEUE_BINS[1:]))
        avail_bin   = int(np.digitize(avail,     AVAIL_BINS[1:]))

        return (max_q_bin, max_q_route, avail_bin, rush, rain, block)


# ──────────────────────────────────────────────────────────────────────
# COORDINADOR MAS — VERSIÓN UNIFICADA
#
# Combina:
#   • Cooldown por ruta (venía de main.py) — evita spam de propuestas
#   • Conciencia temporal con ticks_saturados (venía de agents/) — evita
#     reacciones de pánico a picos momentáneos
#   • La predicción de demanda ahora se usa para filtrar despachos prematuros
#   • Buses disponibles se lee de obs[5] (ya está en el vector); telemetría
#     sigue recibiendo la orden de ejecución para mantener separación de roles
# ──────────────────────────────────────────────────────────────────────
class FloatOptimizationAgentAdapter:

    RUTAS_CORREDOR   = ["A103", "A104", "A105", "A107", "A108"]

    # Heurísticas operativas
    MIN_TICKS_ESPERA = 2     # ticks con saturación antes de reaccionar (2 × 5 min = 10 min)
    UMBRAL_SATURACION = 35   # pasajeros mínimos para considerar una ruta en problemas
    RESERVA_CRITICA  = 2     # buses disponibles mínimos; por debajo solo se despacha en colapso
    COLAPSO_PAX      = 65    # umbral de colapso: ignorar reserva crítica si hay más pasajeros

    # Cooldowns (en ticks) tras aprobación o veto del supervisor
    COOLDOWN_APROBADO = 5    # 25 min simulados para observar el impacto del bus despachado
    COOLDOWN_VETADO   = 3    # 15 min de penalización para que la IA no insista de inmediato

    # Factor de alerta de demanda predicha:
    # Si la demanda proyectada por ruta < UMBRAL_SATURACION * DEMAND_ALERT_FACTOR,
    # no vale la pena despachar todavía.
    DEMAND_ALERT_FACTOR = 1.10   # 10 % de margen sobre el umbral

    def __init__(self, q_learning_agent: QLearningAgent, demand_agent, telemetry_agent, supervisor):
        self.name            = "Float_optimization_agent"
        self.brain           = q_learning_agent
        self.demand_agent    = demand_agent
        self.telemetry_agent = telemetry_agent
        self.supervisor      = supervisor

        # Persistencia temporal por ruta
        self.ticks_saturados = {r: 0 for r in self.RUTAS_CORREDOR}
        self.cooldowns       = {r: 0 for r in self.RUTAS_CORREDOR}

    # ------------------------------------------------------------------
    # Punto de entrada principal — llamado en cada tick de la simulación
    # ------------------------------------------------------------------
    def coordinate_and_decide(self, obs: np.ndarray) -> int:
        # 1. Decrementar cooldowns
        self._decrement_cooldowns()

        # 2. Consultar predicción de demanda vía FIPA
        #    (el resultado se usa para filtrar, no solo para decorar)
        msg_demand  = ACLMessage(self.name, self.demand_agent.name,
                                 FipaPerformative.REQUEST, "solicitar predicción")
        res_demand  = self.demand_agent.receive_message(msg_demand)
        # content es un dict {"por_ruta": ..., "total": float, "tendencia": ...}
        demanda_proyectada_total: float = res_demand.content["total"]

        # 3. Leer buses disponibles directamente del vector de observación
        #    obs[5] = buses_station / FLEET_SIZE  →  desnormalizar con FLEET_SIZE conocido
        #    Usamos telemetría solo para la ORDEN de ejecución (mantiene separación de roles)
        from envs.mb_env import FLEET_SIZE
        buses_disponibles = int(round(float(obs[5]) * FLEET_SIZE))

        # 4. (ticks_saturados reemplazado por wait_minutes del entorno — ver filtros abajo)

        # 5. Consultar al cerebro QL
        action = self.brain.select_action(obs)

        if action == 0:
            return 0

        nombre_ruta  = self.RUTAS_CORREDOR[action - 1]
        from envs.mb_env import MAX_QUEUE
        pax_en_fila  = float(obs[action - 1]) * MAX_QUEUE
        wait_norm    = float(obs[17 + (action - 1)]) if len(obs) >= 22 else 0.0
        wait_minutes = wait_norm * 60.0

        # 6. FILTRO DE COOLDOWN — no spamear al supervisor si ya se pidió hace poco
        if self.supervisor.interactive and self.cooldowns[nombre_ruta] > 0:
            return 0

        # 7. FILTRO DE RECURSOS — no operar sin buses
        if buses_disponibles == 0:
            return 0

        # 8. FILTRO DE RESERVA CRÍTICA — proteger flota salvo colapso
        if buses_disponibles <= self.RESERVA_CRITICA and pax_en_fila < self.COLAPSO_PAX:
            return 0

        # Los filtros de demanda y tiempo se eliminaron: el cerebro QL ya aprendió
        # cuándo despachar. Agregar filtros sobre su salida solo introduce bloqueos.
        # wait_minutes y demanda se registran en info para trazabilidad.

        # 11. Negociación FIPA con el supervisor humano
        proposal = ACLMessage(
            self.name, self.supervisor.name,
            FipaPerformative.PROPOSE,
            f"sugerir despliegue en ruta {nombre_ruta}"
        )
        decision = self.supervisor.receive_message(proposal)

        if decision.performative == FipaPerformative.ACCEPT_PROPOSAL:
            # Orden de ejecución a telemetría (separación de roles: telemetría ejecuta)
            order = ACLMessage(
                self.name, self.telemetry_agent.name,
                FipaPerformative.REQUEST, "ordena despliegue"
            )
            self.telemetry_agent.receive_message(order)

            if self.supervisor.interactive:
                self.cooldowns[nombre_ruta] = self.COOLDOWN_APROBADO
            return action

        else:
            # Veto del supervisor: penalizar temporalmente
            if self.supervisor.interactive:
                self.cooldowns[nombre_ruta] = self.COOLDOWN_VETADO
            return 0

    # ------------------------------------------------------------------
    def _decrement_cooldowns(self) -> None:
        for ruta in self.cooldowns:
            if self.cooldowns[ruta] > 0:
                self.cooldowns[ruta] -= 1