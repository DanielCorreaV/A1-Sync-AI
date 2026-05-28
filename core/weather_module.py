import random
from typing import Literal

WeatherType = Literal["clear", "rain"]

RAIN_PROBABILITY_BY_MONTH: dict[int, float] = {
    1:  0.05,   # enero    — temporada seca
    2:  0.05,   # febrero  — temporada seca
    3:  0.08,   # marzo    — inicio de transición
    4:  0.35,   # abril    — inicio 1ª temporada de lluvias
    5:  0.45,   # mayo     — 1ª temporada, pico moderado
    6:  0.40,   # junio    — 1ª temporada
    7:  0.25,   # julio    — veranillo de San Juan
    8:  0.40,   # agosto   — inicio 2ª temporada
    9:  0.50,   # sept.    — 2ª temporada, alta intensidad
    10: 0.55,   # octubre  — mes más lluvioso del año
    11: 0.45,   # noviembre — final 2ª temporada
    12: 0.15,   # diciembre — regreso a temporada seca
}


class WeatherModule:

    RAIN_TRAFFIC_FACTOR = 1.4

    RAIN_DEMAND_FACTOR = 0.70

    def __init__(self, month: int, seed: int | None = None):
        
        if not 1 <= month <= 12:
            raise ValueError(f"Mes inválido: {month}. Debe estar entre 1 y 12.")

        if seed is not None:
            random.seed(seed)

        self.month = month
        self.rain_probability = RAIN_PROBABILITY_BY_MONTH[month]
        self.weather: WeatherType = self._generate_synthetic()

    def _generate_synthetic(self) -> WeatherType:
        return "rain" if random.random() < self.rain_probability else "clear"

    @property
    def is_raining(self) -> bool:
        return self.weather == "rain"

    @property
    def rain_traffic_factor(self) -> float:
        return self.RAIN_TRAFFIC_FACTOR if self.is_raining else 1.0

    @property
    def demand_factor(self) -> float:
        return self.RAIN_DEMAND_FACTOR if self.is_raining else 1.0

    def __repr__(self) -> str:
        return (
            f"WeatherModule(month={self.month}, weather={self.weather}, "
            f"p_rain={self.rain_probability:.0%})"
        )
