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

        EXPECTED_KEY_LEN = 6
        sample_keys = list(loaded.keys())[:5]
        if sample_keys and len(sample_keys[0]) != EXPECTED_KEY_LEN:
            print(f"[QLearningAgent] ADVERTENCIA: checkpoint incompatible. Descartando.")
            return

        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64),
            {k: np.array(v) for k, v in loaded.items()}
        )
        self.epsilon         = checkpoint.get("epsilon",        self.epsilon)
        self.episode_count   = checkpoint.get("episode_count",  0)
        self.total_steps     = checkpoint.get("total_steps",    0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        print(f"[QLearningAgent] Checkpoint cargado | estados={len(self.q_table):,}")

    def _discretize(self, obs: np.ndarray) -> tuple:
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

class FloatOptimizationAgentAdapter:

    RUTAS_CORREDOR   = ["A103", "A104", "A105", "A107", "A108"]

    # Heurísticas operativas
    MIN_TICKS_ESPERA = 2     
    UMBRAL_SATURACION = 35   
    RESERVA_CRITICA  = 2     
    COLAPSO_PAX      = 65    

    # Cooldowns tras respuesta del supervisor
    COOLDOWN_APROBADO = 5    
    COOLDOWN_VETADO   = 3    
    DEMAND_ALERT_FACTOR = 1.10   

    def __init__(self, q_learning_agent: QLearningAgent, demand_agent, telemetry_agent, visual_agent, supervisor):
        self.name            = "Float_optimization_agent"
        self.brain           = q_learning_agent
        self.demand_agent    = demand_agent
        self.telemetry_agent = telemetry_agent
        self.visual_agent    = visual_agent  # Inyectado para cumplir el flujo estricto FIPA
        self.supervisor      = supervisor

        self.ticks_saturados = {r: 0 for r in self.RUTAS_CORREDOR}
        self.cooldowns       = {r: 0 for r in self.RUTAS_CORREDOR}

    def coordinate_and_decide(self, obs: np.ndarray) -> int:
        # Decrementar cooldowns operacionales
        self._decrement_cooldowns()

        # CONSULTA FIPA: Predicción de demanda futura
        msg_demand  = ACLMessage(self.name, self.demand_agent.name,
                                 FipaPerformative.REQUEST, "solicitar predicción")
        res_demand  = self.demand_agent.receive_message(msg_demand)
        demanda_proyectada_total: float = res_demand.content["total"]

        # CONSULTA FIPA: Leer buses disponibles desde el Agente de Telemetría
        msg_telemetry = ACLMessage(self.name, self.telemetry_agent.name,
                                   FipaPerformative.REQUEST, "solicitar estado de los buses")
        res_telemetry = self.telemetry_agent.receive_message(msg_telemetry)
        buses_disponibles = len(res_telemetry.content)

        # CONSULTA FIPA: Leer estado de las filas desde el Agente de Percepción Visual
        msg_visual = ACLMessage(self.name, self.visual_agent.name,
                                FipaPerformative.REQUEST, "solicitar estado de las filas")
        res_visual = self.visual_agent.receive_message(msg_visual)
        colas_actuales = res_visual.content  # Lista de enteros directo del entorno [pax_A103, ...]

        # Consultar al cerebro matemático (QL) usando la observación nativa de Gym
        action = self.brain.select_action(obs)

        if action == 0:
            return 0

        nombre_ruta = self.RUTAS_CORREDOR[action - 1]
        
        # PROCESAMIENTO DE DATOS PROVENIENTES EXCLUSIVAMENTE DE LOS AGENTES SENSING
        pax_en_fila = float(colas_actuales[action - 1])
        
        # Consulta FIPA adicional: Tiempos de espera reales por ruta
        msg_wait = ACLMessage(self.name, self.visual_agent.name,
                              FipaPerformative.REQUEST, "solicitar tiempos de espera")
        res_wait = self.visual_agent.receive_message(msg_wait)
        wait_minutes = float(res_wait.content[action - 1])

        # FILTRO DE COOLDOWN
        if self.supervisor.interactive and self.cooldowns[nombre_ruta] > 0:
            return 0

        # FILTRO DE RECURSOS
        if buses_disponibles == 0:
            return 0

        # FILTRO DE RESERVA CRÍTICA
        if buses_disponibles <= self.RESERVA_CRITICA and pax_en_fila < self.COLAPSO_PAX:
            return 0

        # Negociación FIPA con el supervisor humano
        proposal = ACLMessage(
            self.name, self.supervisor.name,
            FipaPerformative.PROPOSE,
            f"sugerir despliegue en ruta {nombre_ruta}"
        )
        decision = self.supervisor.receive_message(proposal)

        if decision.performative == FipaPerformative.ACCEPT_PROPOSAL:
            # Enviar orden formal de ejecución al agente encargado de la flota física
            order = ACLMessage(
                self.name, self.telemetry_agent.name,
                FipaPerformative.REQUEST, "ordena despliegue"
            )
            self.telemetry_agent.receive_message(order)

            if self.supervisor.interactive:
                self.cooldowns[nombre_ruta] = self.COOLDOWN_APROBADO
            return action

        else:
            # Veto del supervisor
            if self.supervisor.interactive:
                self.cooldowns[nombre_ruta] = self.COOLDOWN_VETADO
            return 0

    def _decrement_cooldowns(self) -> None:
        for ruta in self.cooldowns:
            if self.cooldowns[ruta] > 0:
                self.cooldowns[ruta] -= 1