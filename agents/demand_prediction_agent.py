
 
from collections import deque
from agents.visual_perception_agent import VisualPerceptionAgent
from core.fipa import ACLMessage, FipaPerformative
 
RUTAS_CORREDOR = ["A103", "A104", "A105", "A107", "A108"]
N_ROUTES       = len(RUTAS_CORREDOR)
WINDOW_SIZE    = 6   # últimos 6 ticks = 30 minutos de historial (5 mins por tick)
 
 
class DemandPredictionAgent:
 
    def __init__(self, visual_agent: VisualPerceptionAgent):
        self.name         = "Demand_prediction_agent"
        self.visual_agent = visual_agent
 
        # Historial circular por ruta
        self._history: dict[str, deque] = {
            r: deque(maxlen=WINDOW_SIZE) for r in RUTAS_CORREDOR
        }
 
    def receive_message(self, message: ACLMessage) -> ACLMessage:
        if message.performative == FipaPerformative.REQUEST \
                and message.content == "solicitar predicción":
 
            # Obtener estado actual de las filas vía Visual Perception
            req  = ACLMessage(
                self.name, self.visual_agent.name,
                FipaPerformative.REQUEST, "solicitar estado de las filas"
            )
            resp = self.visual_agent.receive_message(req)
            filas_actuales: list[int] = resp.content  
 
            # Actualizar historial
            for i, ruta in enumerate(RUTAS_CORREDOR):
                self._history[ruta].append(filas_actuales[i])
 
            # Calcular proyección por ruta
            proyeccion_por_ruta: dict[str, float] = {}
            tendencia_por_ruta:  dict[str, float] = {}
 
            for i, ruta in enumerate(RUTAS_CORREDOR):
                hist = list(self._history[ruta])
                cola_actual = filas_actuales[i]
 
                if len(hist) >= 2:
                    # Tasa de cambio promedio entre ticks consecutivos
                    deltas = [hist[j] - hist[j - 1] for j in range(1, len(hist))]
                    tasa   = sum(deltas) / len(deltas)
                else:
                    # Sin historial suficiente: asumir crecimiento moderado
                    tasa = 2.0
 
                tendencia_por_ruta[ruta]  = round(tasa, 2)
                proyeccion_por_ruta[ruta] = max(0.0, cola_actual + tasa)
 
            prediccion = {
                "por_ruta":  proyeccion_por_ruta,
                "total":     sum(proyeccion_por_ruta.values()),
                "tendencia": tendencia_por_ruta,
            }
 
            return ACLMessage(
                self.name, message.sender,
                FipaPerformative.INFORM, prediccion
            )
 
        return ACLMessage(
            self.name, message.sender,
            FipaPerformative.REJECT_PROPOSAL, "Mensaje no reconocido"
        )