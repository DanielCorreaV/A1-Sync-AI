import streamlit as st
import numpy as np
import time
import pandas as pd
import re
from core.fipa import ACLMessage, FipaPerformative
from envs.mb_env import MBEnv, FLEET_SIZE, MAX_QUEUE
from agents.Telemetry_agent import TelemetryAgent
from agents.visual_perception_agent import VisualPerceptionAgent
from agents.demand_prediction_agent import DemandPredictionAgent
from agents.Fleet_optimization_agent import QLearningAgent, FloatOptimizationAgentAdapter

# Configuración de la página web
st.set_page_config(page_title="A1-Sync-AI Control Panel", layout="wide")

# ──────────────────────────────────────────────────────────────────────
# SUB-CLASE ESPECIALIZADA PARA LA INTERFAZ WEB
# ──────────────────────────────────────────────────────────────────────
class WebHumanSupervisor:
    def __init__(self):
        self.name = "Human_supervisor"
        self.interactive = True

    def receive_message(self, message: ACLMessage) -> ACLMessage:
        if st.session_state.get("web_user_decision") == "ACCEPT":
            st.session_state.web_user_decision = None  # Resetear
            return ACLMessage(self.name, message.sender, FipaPerformative.ACCEPT_PROPOSAL, "Aprobado vía Panel Web")
        
        elif st.session_state.get("web_user_decision") == "VETO":
            st.session_state.web_user_decision = None  # Resetear
            return ACLMessage(self.name, message.sender, FipaPerformative.REJECT_PROPOSAL, "Vetado vía Panel Web")
        
        else:
            # Capturar la propuesta y lanzar la interrupción para pausar el bucle
            st.session_state.pending_proposal = message
            raise StopIteration("Esperando respuesta del supervisor humano en la UI")


# ──────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DEL ESTADO (SESSION STATE)
# ──────────────────────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    env = MBEnv(month=10, verbose=False)
    brain = QLearningAgent()
    brain.epsilon = 0.0  # Modo inferencia
    
    telemetry = TelemetryAgent(env)
    visual = VisualPerceptionAgent(env)
    demand = DemandPredictionAgent(visual)
    supervisor = WebHumanSupervisor()
    
    mas = FloatOptimizationAgentAdapter(brain, demand, telemetry, visual, supervisor)
    obs, _ = env.reset()
    
    st.session_state.initialized = True
    st.session_state.env = env
    st.session_state.visual_agent = visual
    st.session_state.telemetry_agent = telemetry
    st.session_state.mas = mas
    st.session_state.obs = obs
    st.session_state.done = False
    st.session_state.total_r = 0.0
    st.session_state.tick_count = 0
    
    st.session_state.pending_proposal = None
    st.session_state.web_user_decision = None
    st.session_state.auto_run = False
    st.session_state.history = {
        "Tick": [], "Total_Pasajeros": [],
        "A103": [], "A104": [], "A105": [], "A107": [], "A108": []
    }
    st.session_state.current_fipa_sequence = []

# Helper para extraer la ruta de manera segura (ej: "A105") desde el texto del mensaje
def extraer_ruta(texto: str) -> str:
    match = re.search(r"A10[34578]", texto)
    return match.group(0) if match else "Desconocida"


# ──────────────────────────────────────────────────────────────────────
# LÓGICA DEL TICK DE SIMULACIÓN
# ──────────────────────────────────────────────────────────────────────
def step_simulation():
    if st.session_state.done:
        return

    # Secuencia base inicial de consultas FIPA
    st.session_state.current_fipa_sequence = [
        {"icon": "🤖 ➔ 🧠", "perf": "REQUEST", "desc": "solicitar predicción", "from": "Float_optimization_agent", "to": "Demand_prediction_agent"},
        {"icon": "🧠 ➔ 🤖", "perf": "INFORM", "desc": "predicción enviada", "from": "Demand_prediction_agent", "to": "Float_optimization_agent"},
        {"icon": "🤖 ➔ 👁️", "perf": "REQUEST", "desc": "solicitar estado de las filas", "from": "Float_optimization_agent", "to": "Visual_perception_agent"},
        {"icon": "👁️ ➔ 🤖", "perf": "INFORM", "desc": "lista de estado de filas", "from": "Visual_perception_agent", "to": "Float_optimization_agent"},
        {"icon": "🤖 ➔ ⚡", "perf": "REQUEST", "desc": "solicitar estado de los buses", "from": "Float_optimization_agent", "to": "Telemetry_agent"},
        {"icon": "⚡ ➔ 🤖", "perf": "INFORM", "desc": "lista de estado de la flota", "from": "Telemetry_agent", "to": "Float_optimization_agent"},
    ]

    try:
        action = st.session_state.mas.coordinate_and_decide(st.session_state.obs)
        next_obs, reward, term, tr, info = st.session_state.env.step(action)
        
        if action > 0:
            rutas = ["A103", "A104", "A105", "A107", "A108"]
            ruta_act = rutas[action-1]
            st.session_state.current_fipa_sequence.append(
                {"icon": "🤖 ➔ ⚡", "perf": "REQUEST", "desc": f"ordena despliegue ruta {ruta_act}", "from": "Float_optimization_agent", "to": "Telemetry_agent"}
            )
            st.session_state.current_fipa_sequence.append(
                {"icon": "⚡ ➔ 🌍", "perf": "REQUEST", "desc": f"despliega el Transcaribe en ruta {ruta_act}", "from": "Telemetry_agent", "to": "Environment"}
            )

        st.session_state.obs = next_obs
        st.session_state.done = term or tr
        st.session_state.total_r += reward
        st.session_state.tick_count += 1
        st.session_state.pending_proposal = None
        
        msg_filas = ACLMessage("UI", st.session_state.visual_agent.name, FipaPerformative.REQUEST, "solicitar estado de las filas")
        colas = st.session_state.visual_agent.receive_message(msg_filas).content
        st.session_state.history["Tick"].append(st.session_state.tick_count)
        st.session_state.history["Total_Pasajeros"].append(sum(colas))
        for idx, r in enumerate(["A103", "A104", "A105", "A107", "A108"]):
            st.session_state.history[r].append(colas[idx])

    except StopIteration:
        st.session_state.auto_run = False
        ruta_sugerida = extraer_ruta(st.session_state.pending_proposal.content)
        
        st.session_state.current_fipa_sequence.append(
            {
                "icon": "🤖 ➔ 👤", 
                "perf": "PROPOSE", 
                "desc": f"sugerir despliegue ruta {ruta_sugerida}", 
                "from": "Float_optimization_agent", 
                "to": "Human_supervisor"
            }
        )


# ──────────────────────────────────────────────────────────────────────
# RENDERIZADO DE LA INTERFAZ GRÁFICA (UI)
# ──────────────────────────────────────────────────────────────────────
st.title("🎛️ A1-Sync-AI — Centro de Control Transcaribe")
st.subheader("Monitoreo de Performativas FIPA ACL en Tiempo Real")
st.markdown("---")

col_ctrl, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])

with col_ctrl:
    st.write("**Controles de Simulación**")
    btn_step = st.button("👣 Siguiente Tick (5 min)", disabled=st.session_state.done or st.session_state.pending_proposal is not None)
    st.session_state.auto_run = st.toggle("🔄 Modo Automático Continuous", value=st.session_state.auto_run, disabled=st.session_state.done)

msg_buses = ACLMessage("UI", st.session_state.telemetry_agent.name, FipaPerformative.REQUEST, "solicitar estado de los buses")
buses_disponibles = len(st.session_state.telemetry_agent.receive_message(msg_buses).content)
msg_filas_now = ACLMessage("UI", st.session_state.visual_agent.name, FipaPerformative.REQUEST, "solicitar estado de las filas")
pax_totales = sum(st.session_state.visual_agent.receive_message(msg_filas_now).content)

with col_m1:
    st.metric("Pasajeros en Estación", f"{pax_totales} pax")
with col_m2:
    st.metric("Flota Disponible (Patio)", f"🚌 {buses_disponibles} / {FLEET_SIZE}")
with col_m3:
    st.metric("Eficiencia de Operación", f"{st.session_state.total_r:+.2f}")

st.markdown("---")

col_left, col_right = st.columns([3, 2.5])

with col_left:
    st.subheader("📊 Estado de Saturación por Rutas (Percepción Visual)")
    rutas_list = ["A103", "A104", "A105", "A107", "A108"]
    msg_f = ACLMessage("UI", st.session_state.visual_agent.name, FipaPerformative.REQUEST, "solicitar estado de las filas")
    colas_actuales = st.session_state.visual_agent.receive_message(msg_f).content
    
    for i, r in enumerate(rutas_list):
        pax = colas_actuales[i]
        porcentaje = min(1.0, pax / MAX_QUEUE)
        if pax > 65:
            st.error(f"🚨 **Ruta {r}:** {pax} pasajeros esperando (Colapso Crítico)")
        elif pax > 35:
            st.warning(f"⚠️ **Ruta {r}:** {pax} pasajeros esperando (Saturado)")
        else:
            st.success(f"🟢 **Ruta {r}:** {pax} pasajeros esperando (Estable)")
        st.progress(porcentaje)
    
    if len(st.session_state.history["Tick"]) > 0:
        st.subheader("📈 Historial de Acumulación de Pasajeros")
        df_history = pd.DataFrame(st.session_state.history).set_index("Tick")
        st.line_chart(df_history[rutas_list])

# ──────────────────────────────────────────────────────────────────────
# PANEL DERECHO REORGANIZADO
# ──────────────────────────────────────────────────────────────────────
with col_right:
    st.subheader("🛡️ Panel de Decisiones del Supervisor")
    
    # El bloque de sugerencias y botones ahora se renderiza en primer lugar
    if st.session_state.pending_proposal is not None:
        ruta_sug = extraer_ruta(st.session_state.pending_proposal.content)
        st.warning(f"🤖 **Acción Requerida:** La IA sugiere desplegar un bus en la **Ruta {ruta_sug}**.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"👍 ACEPTAR PROPUESTA ({ruta_sug})", use_container_width=True, type="primary"):
                st.session_state.current_fipa_sequence.append(
                    {"icon": "👤 ➔ 🤖", "perf": "ACCEPT_PROPOSAL", "desc": f"acepta sugerencia ruta {ruta_sug}", "from": "Human_supervisor", "to": "Float_optimization_agent"}
                )
                st.session_state.web_user_decision = "ACCEPT"
                step_simulation()
                st.rerun()
        with c2:
            if st.button(f"👎 VETAR PROPUESTA ({ruta_sug})", use_container_width=True):
                st.session_state.current_fipa_sequence.append(
                    {"icon": "👤 ➔ 🤖", "perf": "REJECT_PROPOSAL", "desc": f"rechaza sugerencia ruta {ruta_sug}", "from": "Human_supervisor", "to": "Float_optimization_agent"}
                )
                st.session_state.current_fipa_sequence.append(
                    {"icon": "🤖 ➔ 👤", "perf": "PROPOSE", "desc": "sugiere nuevo despliegue (recalculando)", "from": "Float_optimization_agent", "to": "Human_supervisor"}
                )
                st.session_state.web_user_decision = "VETO"
                step_simulation()
                st.rerun()
    else:
        st.success("✨ **Estado:** El sistema opera de forma óptima. Sin alertas del MAS.")
        
    st.markdown("---")
    
    # El trazador histórico del ciclo queda en la parte inferior
    st.subheader("⏳ Trazador de Secuencia FIPA (Ciclo del Tick)")
    
    if not st.session_state.current_fipa_sequence:
        st.caption("Presiona 'Siguiente Tick' para iniciar el intercambio de mensajes entre agentes.")
    else:
        for msg in st.session_state.current_fipa_sequence:
            if msg["perf"] == "REQUEST": color = "blue"
            elif msg["perf"] == "INFORM": color = "green"
            elif msg["perf"] == "PROPOSE": color = "orange"
            elif "ACCEPT" in msg["perf"]: color = "#1E88E5"
            elif "REJECT" in msg["perf"]: color = "#D32F2F"
            else: color = "grey"
                
            st.markdown(f"""
            **{msg['icon']}** &nbsp; <span style='color:{color}; font-weight:bold;'>{msg['perf']}</span> : *{msg['desc']}* <small style='color:gray;'>De: `{msg['from']}` ➔ Para: `{msg['to']}`</small>
            """, unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0px; border-style:dashed;' />", unsafe_allow_html=True)

# Manejo del Automatismo
if btn_step:
    step_simulation()
    st.rerun()

if st.session_state.auto_run and st.session_state.pending_proposal is None and not st.session_state.done:
    time.sleep(0.4)
    step_simulation()
    st.rerun()