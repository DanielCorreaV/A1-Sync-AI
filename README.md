# A1-Sync-AI

Sistema de optimización de despacho de flota para el corredor **Madre Bernarda — Zona Sur de Cartagena de Indias**.

Combina un simulador de operación de buses con un agente Q-Learning que aprende a tomar mejores decisiones de despacho sesión a sesión.

---

## Estructura

```
A1-Sync-AI/
├── core/
│   ├── bus.py              # Ciclo de vida de cada unidad de flota
│   ├── weather_module.py   # Clima sintético por mes (probabilidades IDEAM/WeatherSpark)
│   └── traffic_module.py   # Multiplicador de tráfico por franja horaria (TomTom 2025)
├── envs/
│   └── mb_env.py           # Entorno Gymnasium — Madre Bernarda
├── agents/
│   ├── Fleet_optimization_agent.py  # Agente Q-Learning tabular
│   └── qtable_checkpoint.pkl        # Tabla Q guardada (se crea tras el primer entrenamiento)
├── main.py                 # Punto de entrada
└── requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

### Primer entrenamiento (desde cero)
```bash
python main.py
```

### Continuar entrenando (carga el checkpoint automáticamente)
```bash
python main.py --episodes 500
```

### Simular mes específico (ej. octubre, temporada alta)
```bash
python main.py --episodes 200 --month 10
```

### Ver el agente ya entrenado en acción (sin entrenamiento)
```bash
python main.py --episodes 0
```

### Todos los parámetros
```
--episodes N   Episodios a entrenar (default: 200)
--month M      Mes 1-12 (default: 10 = octubre)
--verbose      Imprime estado en cada tick
--render K     Renderiza el entorno cada K episodios (default: 50)
--seed S       Semilla de aleatoriedad
```

---

## Cómo funciona el aprendizaje acumulativo

Al finalizar cada sesión, el agente guarda `agents/qtable_checkpoint.pkl` con:
- La tabla Q completa
- El valor actual de ε (exploración)
- El contador de episodios y pasos totales
- El historial de recompensas

La próxima sesión carga ese checkpoint y continúa desde donde se quedó. El agente nunca olvida lo que aprendió.

---

## Espacio de acciones

| Acción | Significado |
|--------|-------------|
| 0 | No despachar ningún bus este tick |
| 1 | Despachar → ruta A103 |
| 2 | Despachar → ruta A104 |
| 3 | Despachar → ruta A105 |
| 4 | Despachar → ruta A107 |
| 5 | Despachar → ruta A108 |

---

## Fuentes de calibración

- **Tráfico**: TomTom Traffic Index 2025 — Cartagena (congestión por franja horaria)
- **Clima**: WeatherSpark 2024-2025 + IDEAM (probabilidades de lluvia por mes)
- **Demanda**: Encuesta de Percepción Ciudadana 2025, Cartagena Cómo Vamos
