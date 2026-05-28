BLOCK_PROBABILITY = 0.05

BLOCK_ADDEND = 1.2

HOURLY_MULTIPLIERS: list[tuple[float, float, float]] = [
    (6.5,  8.5,  1.6),   
    (11.0, 13.0, 1.4),   
    (16.0, 19.5, 2.0),   
]

DEFAULT_MULTIPLIER = 1.0   


class TrafficModule:


    def __init__(self, weather_rain_factor: float = 1.0, seed: int | None = None):
        import random
        if seed is not None:
            random.seed(seed)

        self._random = random
        self.weather_factor = weather_rain_factor
        self.is_blocked: bool = self._random.random() < BLOCK_PROBABILITY

    def get_multiplier(self, hour: float) -> float:
        base = self._get_hourly_base(hour)
        multiplier = base * self.weather_factor
        if self.is_blocked:
            multiplier += BLOCK_ADDEND
        return multiplier

    def is_rush_hour(self, hour: float) -> bool:
        return any(h_start <= hour <= h_end for h_start, h_end, _ in HOURLY_MULTIPLIERS)

    @staticmethod
    def _get_hourly_base(hour: float) -> float:
        for h_start, h_end, factor in HOURLY_MULTIPLIERS:
            if h_start <= hour <= h_end:
                return factor
        return DEFAULT_MULTIPLIER

    def describe(self, hour: float) -> str:
        mult = self.get_multiplier(hour)
        parts = [f"mult={mult:.2f}"]
        if self.is_rush_hour(hour):
            parts.append("hora-pico")
        if self.weather_factor > 1.0:
            parts.append(f"lluvia(x{self.weather_factor})")
        if self.is_blocked:
            parts.append(f"bloqueo(+{BLOCK_ADDEND})")
        return f"TrafficModule({', '.join(parts)})"

    def __repr__(self) -> str:
        return (
            f"TrafficModule(weather_factor={self.weather_factor}, "
            f"blocked={self.is_blocked})"
        )
