
import argparse
import sys
import os
import time
import numpy as np

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.mb_env import MBEnv
from agents.Fleet_optimization_agent import QLearningAgent


DEFAULT_EPISODES  = 200
DEFAULT_MONTH     = 10      # octubre: temporada alta de lluvia, mayor demanda
DEFAULT_RENDER_K  = 50      # renderizar cada 50 episodios
LOG_INTERVAL      = 25      # imprimir resumen cada N episodios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A1-Sync-AI — Entrenamiento autónomo del agente de flota Transcaribe"
    )
    parser.add_argument("--episodes", type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--month",    type=int,   default=DEFAULT_MONTH,
                        help="Mes a simular (1=enero ... 12=diciembre)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Imprimir estado en cada tick")
    parser.add_argument("--render",   type=int,   default=DEFAULT_RENDER_K,
                        help="Renderizar el entorno cada N episodios (0=nunca)")
    parser.add_argument("--seed",     type=int,   default=None,
                        help="Semilla de aleatoriedad")
    return parser.parse_args()


def train(
    episodes:  int,
    month:     int,
    verbose:   bool,
    render_k:  int,
    seed:      int | None,
) -> None:
    
    print("\n" + "═" * 65)
    print("  A1-Sync-AI — Sistema de optimización de flota Transcaribe")
    print("  Corredor Madre Bernarda — Zona Sur de Cartagena")
    print("═" * 65)
    print(f"  Episodios a entrenar : {episodes}")
    print(f"  Mes de simulación    : {month} (1=ene ... 12=dic)")
    print(f"  Semilla              : {seed if seed is not None else 'aleatoria'}")
    print("═" * 65 + "\n")

    env   = MBEnv(month=month, seed=seed, verbose=verbose)
    agent = QLearningAgent()

    session_rewards: list[float] = []
    session_queues:  list[float] = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed)
        ep_reward  = 0.0
        ep_queues: list[float] = []
        done = False

        while not done:
            action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, next_obs, terminated)

            ep_reward += reward
            ep_queues.append(info["total_queue"])

            if render_k > 0 and ep % render_k == 0:
                env.render()

            obs = next_obs

        agent.end_episode(ep_reward)
        session_rewards.append(ep_reward)
        session_queues.append(float(np.mean(ep_queues)))

        if ep % LOG_INTERVAL == 0 or ep == 1 or ep == episodes:
            elapsed  = time.time() - start_time
            ep_per_s = ep / elapsed if elapsed > 0 else 0
            recent_r = session_rewards[-LOG_INTERVAL:]
            recent_q = session_queues[-LOG_INTERVAL:]

            print(
                f"  Ep {ep:5d}/{episodes} | "
                f"ε={agent.epsilon:.3f} | "
                f"R̄={np.mean(recent_r):+.3f} | "
                f"Q̄={np.mean(recent_q):.1f} pax | "
                f"Estados={len(agent.q_table):,} | "
                f"{ep_per_s:.1f} ep/s"
            )

    elapsed = time.time() - start_time
    print(f"\n{'─'*65}")
    print(f"  Entrenamiento completado en {elapsed:.1f}s")

    stats = agent.stats(last_n=min(100, episodes))
    print(f"  Resumen últimos {min(100, episodes)} episodios:")
    print(f"    Recompensa media  : {stats['reward_mean']:+.4f}")
    print(f"    Recompensa máx    : {stats['reward_max']:+.4f}")
    print(f"    Recompensa mín    : {stats['reward_min']:+.4f}")
    print(f"    ε final           : {stats['epsilon']}")
    print(f"    Estados en tabla  : {stats['q_table_size']:,}")
    print(f"    Pasos totales     : {stats['total_steps']:,}")

    print(f"\n{'─'*65}")
    agent.save()

    env.close()
    print("\n  Listo. El agente continuará aprendiendo en la próxima sesión.\n")


def demo(month: int = 10) -> None:
   
    print("\n[DEMO] Corriendo 1 episodio con el agente entrenado...\n")
    env   = MBEnv(month=month, verbose=False)
    agent = QLearningAgent()
    agent.epsilon = 0.0    # sin exploración: solo explotación del conocimiento

    obs, _ = env.reset()
    done   = False
    total  = 0.0

    while not done:
        action            = agent.select_action(obs)
        obs, r, term, tr, info = env.step(action)
        done              = term or tr
        total            += r
        env.render()

    print(f"\n[DEMO] Recompensa total del episodio: {total:+.3f}")
    env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.episodes == 0:
        demo(month=args.month)
    else:
        train(
            episodes  = args.episodes,
            month     = args.month,
            verbose   = args.verbose,
            render_k  = args.render,
            seed      = args.seed,
        )
