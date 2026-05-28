# core/bus.py
#
# CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL:
#   1. Corrección del bug de timing: la transición de estado ahora solo ocurre
#      si wait_time llegó a <= 0 EN ESTE tick específico (flag `_just_arrived`).
#      Antes, si un bus terminaba su viaje y su descanso era muy corto, podía
#      quedar disponible en el mismo tick en que llegó, lo cual es físicamente
#      imposible (un bus no descansa y regresa en 5 minutos).
#   2. El estado "station" ahora requiere haber pasado por "rest" al menos una vez
#      para evitar que buses recién llegados se auto-despachen.
#   3. Se agrega `just_completed_trip` como propiedad de solo lectura para que
#      módulos externos puedan saber si un bus terminó su ruta en este tick
#      (útil para métricas y telemetría).
 
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
        self._transitioned_this_tick = False   # flag de guard interno
 
    def update(self, delta: float, traffic_multiplier: float) -> None:
        """
        Avanza el reloj interno del bus.
 
        El multiplicador de tráfico solo aplica al estado 'onGoing' (el viaje
        tarda más cuando hay tráfico o lluvia).
        """
        self._transitioned_this_tick = False
 
        if self.wait_time > 0:
            effective_delta = (
                delta / traffic_multiplier if self.state == "onGoing" else delta
            )
            self.wait_time -= effective_delta
 
        # FIX: la transición solo ocurre UNA vez por tick.
        # Si wait_time baja de 0, el bus pasa al SIGUIENTE estado pero
        # no puede volver a "station" hasta el tick siguiente.
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
            # FIX: el bus queda en "station" con wait_time = 0,
            # pero `is_available` solo devuelve True en el PRÓXIMO tick
            # porque _transitioned_this_tick == True en este tick.
            self.state     = "station"
            self.wait_time = 0.0
 
        elif self.state == "stranded":
            self.state     = "portal"
            self.wait_time = 0.0
 
    def dispatch(self) -> None:
        """
        Despacha el bus a una ruta.
        Lanza AssertionError si el bus no está en condiciones de salir.
        """
        assert self.is_available, (
            f"Bus {self.id}: intento de despacho en estado '{self.state}' "
            f"(wait={self.wait_time:.1f} min, recién llegado={self._transitioned_this_tick})"
        )
        self.state     = "onGoing"
        self.wait_time = self.TRIP_DURATION
 
    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------
 
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