# agents/Telemetry_agent.py

from core.fipa import FipaPerformative, ACLMessage

class TelemetryAgent:
    def __init__(self, env):
        self.name = "Telemetry_agent"
        self.env = env  

    def receive_message(self, message: ACLMessage) -> ACLMessage:
        if message.performative == FipaPerformative.REQUEST and message.content == "solicitar estado de los buses":
            # Conexión con la flota interna (_fleet) para validar disponibilidad física
            buses_disponibles = [b for b in self.env._fleet if b.is_available]
            return ACLMessage(self.name, message.sender, FipaPerformative.INFORM, buses_disponibles)
            
        return ACLMessage(self.name, message.sender, FipaPerformative.REJECT_PROPOSAL, "Orden no reconocida")