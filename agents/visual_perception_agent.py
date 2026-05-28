# agents/visual_perception_agent.py

from core.fipa import FipaPerformative, ACLMessage

# Orden oficial de las rutas en el corredor Madre Bernarda
RUTAS_CORREDOR = ["A103", "A104", "A105", "A107", "A108"]

class VisualPerceptionAgent:
    def __init__(self, env):
        self.name = "Visual_perception_agent"
        self.env = env

    def receive_message(self, message: ACLMessage) -> ACLMessage:
        if message.performative == FipaPerformative.REQUEST and message.content == "solicitar estado de las filas":
            # Extrae directamente los pasajeros de la estructura interna del entorno
            colas_actuales = [self.env._routes[r]["fila"] for r in RUTAS_CORREDOR]
            return ACLMessage(self.name, message.sender, FipaPerformative.INFORM, colas_actuales)
            
        return ACLMessage(self.name, message.sender, FipaPerformative.REJECT_PROPOSAL, "Fallo de lectura perceptual")