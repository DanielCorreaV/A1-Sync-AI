import os
import pickle
import random
import numpy as np
from collections import defaultdict
from typing import Any

# Ruta por defecto para guardar la tabla Q
DEFAULT_QTABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qtable_checkpoint.pkl"
)

QUEUE_BINS   = [0.0, 0.15, 0.40, 0.70, 1.01]   
AVAIL_BINS   = [0.0, 0.08, 0.20, 0.40, 1.01]    
N_ROUTES     = 5


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
        td_error  = td_target - q_current

        self.q_table[s][action] += self.alpha * td_error
        self.total_steps += 1

    def end_episode(self, episode_reward: float) -> None:
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        # Decaimiento de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self) -> None:
        checkpoint = {
            "q_table":        dict(self.q_table),   # defaultdict → dict para pickle
            "epsilon":        self.epsilon,
            "episode_count":  self.episode_count,
            "total_steps":    self.total_steps,
            "episode_rewards": self.episode_rewards,
        }
        os.makedirs(os.path.dirname(self.qtable_path) or ".", exist_ok=True)
        with open(self.qtable_path, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"[Agent] Checkpoint guardado → {self.qtable_path}")
        print(f"        Estados aprendidos: {len(self.q_table):,} | "
              f"Episodios totales: {self.episode_count} | "
              f"ε actual: {self.epsilon:.4f}")

    def load(self) -> None:
        if not os.path.exists(self.qtable_path):
            print(f"[Agent] No se encontró checkpoint en {self.qtable_path}. "
                  "Iniciando desde cero.")
            return

        with open(self.qtable_path, "rb") as f:
            checkpoint = pickle.load(f)

        loaded_table = checkpoint.get("q_table", {})
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64),
            {k: np.array(v) for k, v in loaded_table.items()}
        )
        self.epsilon        = checkpoint.get("epsilon",        self.epsilon)
        self.episode_count  = checkpoint.get("episode_count",  0)
        self.total_steps    = checkpoint.get("total_steps",    0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])

        print(f"[Agent] Checkpoint cargado desde {self.qtable_path}")
        print(f"        Estados aprendidos: {len(self.q_table):,} | "
              f"Episodios previos: {self.episode_count} | "
              f"ε restaurado: {self.epsilon:.4f}")


    def stats(self, last_n: int = 50) -> dict[str, Any]:
        recent = self.episode_rewards[-last_n:] if self.episode_rewards else [0.0]
        return {
            "episode_count":    self.episode_count,
            "total_steps":      self.total_steps,
            "epsilon":          round(self.epsilon, 4),
            "q_table_size":     len(self.q_table),
            "reward_mean":      round(float(np.mean(recent)), 4),
            "reward_std":       round(float(np.std(recent)),  4),
            "reward_max":       round(float(np.max(recent)),  4),
            "reward_min":       round(float(np.min(recent)),  4),
        }

    def _discretize(self, obs: np.ndarray) -> tuple:
        queues = obs[0:N_ROUTES]                     
        avail  = float(obs[5])                       
        rush   = int(round(float(obs[11])))          
        rain   = int(round(float(obs[13])))         
        block  = int(round(float(obs[14])))          

        max_q_route = int(np.argmax(queues))         

        max_q_val   = float(np.max(queues))
        max_q_bin   = int(np.digitize(max_q_val, QUEUE_BINS[1:])) 

        avail_bin   = int(np.digitize(avail, AVAIL_BINS[1:]))     

        return (max_q_bin, max_q_route, avail_bin, rush, rain, block)
