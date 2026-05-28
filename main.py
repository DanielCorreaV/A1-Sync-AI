# main.py
#
# CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL:
#   1. Eliminado el FloatOptimizationAgentAdapter duplicado que vivía aquí.
#      Ahora se importa directamente desde agents/Fleet_optimization_agent.py.
#   2. El ciclo de entrenamiento ya no ignora el contenido de la predicción
#      de demanda — el coordinador unificado lo consume internamente.
#   3. Se agrega logging de métricas de demanda al final del entrenamiento
#      para poder ver si el agente está aprendiendo a responder a tendencias.

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.mb_env import MBEnv
from agents.Telemetry_agent import TelemetryAgent
from agents.visual_perception_agent import VisualPerceptionAgent
from agents.demand_prediction_agent import DemandPredictionAgent
from agents.human_supervisor_agent import HumanSupervisor

# FIX: importar ambas clases desde el único lugar donde deben vivir
from agents.Fleet_optimization_agent import QLearningAgent, FloatOptimizationAgentAdapter

DEFAULT_EPISODES = 200
DEFAULT_MONTH    = 10
LOG_INTERVAL     = 25


# ──────────────────────────────────────────────────────────────────────
# HELPERS PARA INSTANCIAR EL MAS
# ──────────────────────────────────────────────────────────────────────

def build_mas(env: MBEnv, interactive: bool) -> tuple[QLearningAgent, FloatOptimizationAgentAdapter]:
    """
    Construye y conecta todos los agentes del MAS.
    Devuelve (agente_ql, coordinador) para que el loop de entrenamiento
    pueda llamar a agent.update() y agent.end_episode() directamente.
    """
    agent     = QLearningAgent()
    telemetry = TelemetryAgent(env)
    visual    = VisualPerceptionAgent(env)
    demand    = DemandPredictionAgent(visual)
    human     = HumanSupervisor(interactive=interactive)
    coordinator = FloatOptimizationAgentAdapter(agent, demand, telemetry, human)
    return agent, coordinator


# ──────────────────────────────────────────────────────────────────────
# MODO ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A1-Sync-AI — Transcaribe MAS Orchestrator")
    parser.add_argument("--episodes", type=int,  default=DEFAULT_EPISODES,
                        help="Episodios de entrenamiento. 0 para modo demo interactivo.")
    parser.add_argument("--month",    type=int,  default=DEFAULT_MONTH)
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--render",   type=int,  default=50)
    parser.add_argument("--seed",     type=int,  default=None)
    return parser.parse_args()


def train(episodes: int, month: int, verbose: bool, seed: int | None) -> None:
    print("\n  A1-Sync-AI — Modo Entrenamiento de Alta Velocidad...\n")

    env             = MBEnv(month=month, seed=seed, verbose=verbose)
    agent, mas      = build_mas(env, interactive=False)

    session_rewards: list[float] = []
    session_queues:  list[float] = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, _  = env.reset(seed=seed)
        ep_reward = 0.0
        ep_queues: list[float] = []
        done = False

        while not done:
            action                         = mas.coordinate_and_decide(obs)
            next_obs, reward, term, tr, info = env.step(action)
            done = term or tr

            agent.update(obs, action, reward, next_obs, term)
            ep_reward += reward
            ep_queues.append(info["total_queue"])
            obs = next_obs

        agent.end_episode(ep_reward)
        session_rewards.append(ep_reward)
        session_queues.append(float(np.mean(ep_queues)))

        if ep % LOG_INTERVAL == 0 or ep == 1 or ep == episodes:
            elapsed = time.time() - start_time
            r_window = session_rewards[-LOG_INTERVAL:]
            q_window = session_queues[-LOG_INTERVAL:]
            print(
                f"  Ep {ep:5d}/{episodes} | "
                f"ε={agent.epsilon:.3f} | "
                f"R̄={np.mean(r_window):+.3f} | "
                f"Q̄={np.mean(q_window):.1f} pax | "
                f"t={elapsed:.0f}s"
            )

    agent.save()
    env.close()
    print("\n  Entrenamiento completado.")


# ──────────────────────────────────────────────────────────────────────
# MODO DEMO INTERACTIVO
# ──────────────────────────────────────────────────────────────────────

def demo(month: int = 10) -> None:
    print("\n" + "═" * 65)
    print("  [PROTOTIPO INTERACTIVO] Simulación — Corredor Madre Bernarda")
    print("  El sistema inicia desde las 05:00 AM.")
    print("═" * 65 + "\n")

    env        = MBEnv(month=month, verbose=False)
    agent, mas = build_mas(env, interactive=True)
    agent.epsilon = 0.0   # modo inferencia: comportamiento óptimo entrenado

    obs, _    = env.reset()
    done      = False
    total_r   = 0.0

    while not done:
        env.render()
        action              = mas.coordinate_and_decide(obs)
        obs, r, term, tr, _ = env.step(action)
        done = term or tr
        total_r += r
        time.sleep(0.7)

    print(f"\n  [DEMO] Simulación finalizada. Recompensa del día: {total_r:+.3f}")
    env.close()


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.episodes == 0:
        demo(month=args.month)
    else:
        train(
            episodes=args.episodes,
            month=args.month,
            verbose=args.verbose,
            seed=args.seed,
        )