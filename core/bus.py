
 
import random
 
 
class Bus:
 
    BREAKDOWN_PROB     = 0.02   # probabilidad de avería al finalizar un viaje
    BREAKDOWN_RECOVERY = 120    # minutos en taller antes de volver al portal
 
    REST_MIN = 20               # descanso mínimo en minutos
    REST_MAX = 30               # descanso máximo en minutos
 
    TRIP_DURATION = 45          # duración de un viaje completo en minutos
 
    def __init__(self, bus_id: int):
        self.id        = bus_id
        self.state     = "station"
        self.wait_time = 0.0
        self._transitioned_this_tick = False   
 
    def update(self, delta: float, traffic_multiplier: float) -> None:

        self._transitioned_this_tick = False
 
        if self.wait_time > 0:
            effective_delta = (
                delta / traffic_multiplier if self.state == "onGoing" else delta
            )
            self.wait_time -= effective_delta
 
        if self.wait_time <= 0 and self.state != "station":
            self._transition()
            self._transitioned_this_tick = True
 
    def _transition(self) -> None:
        """
        Máquina de estados interna.
 
        onGoing → (avería?) stranded : portal
                            normal   → rest
        rest    → station
        stranded → portal
        portal  → (gestionado externamente por _refill_platform en el entorno)
        """
        if self.state == "onGoing":
            if random.random() < self.BREAKDOWN_PROB:
                self.state     = "stranded"
                self.wait_time = self.BREAKDOWN_RECOVERY
            else:
                self.state     = "rest"
                self.wait_time = random.randint(self.REST_MIN, self.REST_MAX)
 
        elif self.state == "rest":

            self.state     = "station"
            self.wait_time = 0.0
 
        elif self.state == "stranded":
            self.state     = "portal"
            self.wait_time = 0.0
 
    def dispatch(self) -> None:

        assert self.is_available, (
            f"Bus {self.id}: intento de despacho en estado '{self.state}' "
            f"(wait={self.wait_time:.1f} min, recién llegado={self._transitioned_this_tick})"
        )
        self.state     = "onGoing"
        self.wait_time = self.TRIP_DURATION
 
 
    @property
    def is_available(self) -> bool:
        """
        True solo si el bus está en plataforma Y no acaba de transicionar
        a "station" en este mismo tick (evita despacho inmediato tras descanso).
        """
        return self.state == "station" and not self._transitioned_this_tick
 
    @property
    def is_in_portal(self) -> bool:
        return self.state == "portal"
 
    @property
    def just_completed_trip(self) -> bool:
        """True si el bus terminó su viaje en este tick (útil para métricas)."""
        return self._transitioned_this_tick and self.state in ("rest", "stranded")
 
    def __repr__(self) -> str:
        return (
            f"Bus(id={self.id}, state={self.state}, "
            f"wait={self.wait_time:.1f}min)"
        )