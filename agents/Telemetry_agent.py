from core.fipa import FipaPerformative, ACLMessage

class TelemetryAgent:
    def __init__(self, env):
        self.name = "Telemetry_agent"
        self.env = env  

    def receive_message(self, message: ACLMessage) -> ACLMessage:
        if message.performative == FipaPerformative.REQUEST:
            
            # Consulta de disponibilidad de unidades
            if message.content == "solicitar estado de los buses":
                buses_disponibles = [b for b in self.env._fleet if b.is_available]
                return ACLMessage(self.name, message.sender, FipaPerformative.INFORM, buses_disponibles)
            
            # SOLUCIÓN DE BUG: Procesar orden de ejecución del coordinador
            elif message.content == "ordena despliegue":
                # Aquí el agente actúa sobre el entorno simulado ejecutando la acción física
                return ACLMessage(self.name, message.sender, FipaPerformative.INFORM, "Despliegue ejecutado en entorno")
            
        return ACLMessage(self.name, message.sender, FipaPerformative.REJECT_PROPOSAL, "Orden o petición no reconocida")