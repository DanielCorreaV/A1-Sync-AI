import random


class Bus:
    BREAKDOWN_PROB = 0.02

    BREAKDOWN_RECOVERY = 120

    REST_MIN = 20
    REST_MAX = 30

    TRIP_DURATION = 45

    def __init__(self, bus_id: int):
        self.id = bus_id
        self.state = "station"   
        self.wait_time = 0.0     

    def update(self, delta: float, traffic_multiplier: float) -> None:
        if self.wait_time > 0:
            effective_delta = delta / traffic_multiplier if self.state == "onGoing" else delta
            self.wait_time -= effective_delta

        if self.wait_time <= 0:
            self._transition()

    def _transition(self) -> None:
        if self.state == "onGoing":
            if random.random() < self.BREAKDOWN_PROB:
                self.state = "stranded"
                self.wait_time = self.BREAKDOWN_RECOVERY
            else:
                self.state = "rest"
                self.wait_time = random.randint(self.REST_MIN, self.REST_MAX)

        elif self.state == "rest":
            self.state = "station"
            self.wait_time = 0

        elif self.state == "stranded":
            self.state = "portal"
            self.wait_time = 0

    def dispatch(self) -> None:
        assert self.state == "station", (
            f"Bus {self.id}: intento de despacho en estado '{self.state}'"
        )
        self.state = "onGoing"
        self.wait_time = self.TRIP_DURATION

    @property
    def is_available(self) -> bool:
        return self.state == "station"

    @property
    def is_in_portal(self) -> bool:
        return self.state == "portal"

    def __repr__(self) -> str:
        return f"Bus(id={self.id}, state={self.state}, wait={self.wait_time:.1f}min)"
