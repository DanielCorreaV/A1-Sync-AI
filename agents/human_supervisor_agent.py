
from core.fipa import FipaPerformative, ACLMessage

class HumanSupervisor:

    def __init__(self, interactive: bool = False):
        self.name = "Human_supervisor"
        self.interactive = interactive  

    def receive_message(self, message: ACLMessage) -> ACLMessage:
        if message.performative == FipaPerformative.PROPOSE and "sugerir despliegue" in message.content:
            if not self.interactive:
                return ACLMessage(self.name, message.sender, FipaPerformative.ACCEPT_PROPOSAL, "auto_aprobado")
            
            print("\n" + "═"*60)
            print(f" [FIPA: PROPOSE] Mensaje recibido de: {message.sender}")
            print(f" PROPUESTA MAS: {message.content.upper()}")
            print("═"*60)
            
            ans = ""
            while ans not in ["s", "n"]:
                ans = input("❓ ¿Autoriza el despacho de esta unidad de Transcaribe? (s/n): ").strip().lower()
            
            if ans == "s":
                print("\n[FIPA: ACCEPT_PROPOSAL] Enviando confirmación de despacho...")
                return ACLMessage(self.name, message.sender, FipaPerformative.ACCEPT_PROPOSAL, "aprobado")
            else:
                print("\n[FIPA: REJECT_PROPOSAL] Despacho cancelado por veto del supervisor.")
                return ACLMessage(self.name, message.sender, FipaPerformative.REJECT_PROPOSAL, "rechazado")
                
        return ACLMessage(self.name, message.sender, FipaPerformative.REJECT_PROPOSAL, "Sin comentarios")