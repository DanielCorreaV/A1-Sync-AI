# agents/demand_prediction_agent.py
#
# CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL:
#   1. La predicción ya no es un escalar fijo de x1.15.
#      Usa una ventana deslizante de los últimos N ticks para calcular
#      la tendencia real de demanda (tasa de cambio promedio).
#   2. Expone también la predicción por ruta individual, no solo el total.
#   3. El contenido del mensaje INFORM ahora es un dict estructurado que
#      el coordinador puede consumir directamente.
 
from collections import deque
from agents.visual_perception_agent import VisualPerceptionAgent
from core.fipa import ACLMessage, FipaPerformative
 
RUTAS_CORREDOR = ["A103", "A104", "A105", "A107", "A108"]
N_ROUTES       = len(RUTAS_CORREDOR)
WINDOW_SIZE    = 6   # últimos 6 ticks = 30 minutos de historial
 
 
class DemandPredictionAgent:
    """
    Predice la demanda futura usando una ventana deslizante.
 
    Por cada ruta mantiene el historial de colas de los últimos WINDOW_SIZE
    ticks. Con esos valores calcula:
      • tasa de cambio promedio (pasajeros / tick)
      • proyección a 1 tick adelante = cola_actual + tasa_promedio
 
    El resultado que devuelve en el INFORM es un dict:
        {
            "por_ruta":  {"A103": float, "A104": float, ...},   # proyección individual
            "total":     float,                                   # suma de proyecciones
            "tendencia": {"A103": float, ...}                    # tasa de cambio promedio
        }
    """
 
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
 
            # 1. Obtener estado actual de las filas vía Visual Perception
            req  = ACLMessage(
                self.name, self.visual_agent.name,
                FipaPerformative.REQUEST, "solicitar estado de las filas"
            )
            resp = self.visual_agent.receive_message(req)
            filas_actuales: list[int] = resp.content  # [pax_A103, pax_A104, ...]
 
            # 2. Actualizar historial
            for i, ruta in enumerate(RUTAS_CORREDOR):
                self._history[ruta].append(filas_actuales[i])
 
            # 3. Calcular proyección por ruta
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