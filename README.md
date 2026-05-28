# Sistema de Predicción de Loterías (38 Animales)

Sistema multiplotería para análisis, predicción y scraping de resultados de loterías de animales. Combina métodos estadísticos clásicos (Markov, frecuencia histórica) con Machine Learning (Random Forest, XGBoost) para generar predicciones.

## Loterías Soportadas

| Lotería | Sorteos/día | Horario |
|---|---|---|
| Lotto Activo | 12 | 08:00 - 19:00 (cada hora) |
| La Granjita | 12 | 08:00 - 19:00 (cada hora) |
| Selva Plus | 12 | 08:00 - 19:00 (cada hora) |
| Lotto Activo Rd Int | 12 | 08:30 - 19:30 (cada 30 min) |

## Métodos de Predicción

- **Markov**: Probabilidad de transición entre animales consecutivos
- **Hist. Hora**: Frecuencia histórica por hora del día
- **M+H**: Combinación Markov + Hist. Hora (aproximadamente 67% precisión Top-20)
- **Random Forest**: Modelo de ML con features de frecuencia, distancia y transiciones
- **XGBoost**: Modelo gradient-boosted con las mismas features
- **RF+XGB**: Consenso entre ambos modelos ML

## Requisitos

- Python 3.8+
- pandas, numpy
- scikit-learn, xgboost
- requests, beautifulsoup4
- tkinter (incluido con Python)

Instalación rápida:

```bash
pip install pandas numpy scikit-learn xgboost requests beautifulsoup4 openpyxl
```

## Instalación y Ejecución

```bash
git clone https://github.com/GabozAlex/PrediccionNumeros.git
cd PrediccionNumeros
pip install -r requirements.txt
python ui.py
```

## Estructura del Proyecto

```
├── ui.py                          # Interfaz gráfica (Tkinter)
├── loteria_base.py                # Clase base: análisis, ML, predicciones
├── utils.py                       # Animales, grupos, mapeo de horas
├── prediccionNumero.py            # Wrapper para Lotto Activo
├── lotto_activo.py                # Wrapper Lotto Activo
├── la_granjita.py                 # Wrapper La Granjita
├── selva_plus.py                  # Wrapper Selva Plus
├── lotto_rd_int.py                # Wrapper Lotto Activo Rd Int
├── scraper_lotto.py               # Scraper Lotto Activo
├── scraper_la_granjita.py         # Scraper La Granjita
├── scraper_selva_plus.py          # Scraper Selva Plus
├── scraper_lotto_rd_int.py        # Scraper Lotto Activo Rd Int
├── scrape_all.py                  # Scrapea todas las loterías
├── train_all.py                   # Entrena modelos RF + XGB para todas
├── requirements.txt               # Dependencias
└── models/                        # Modelos entrenados (por lotería)
```

## Interfaz de Usuario

### Pestaña Predicción
- **Predecir Siguiente (M+H)**: Predice el próximo sorteo usando Markov + Hora (solicita animal y hora)
- **Predecir Siguiente (Completo)**: Predicción completa para la siguiente hora con todos los modelos
- **Top-10 Frecuencia por Hora**: Muestra los animales más frecuentes históricamente por cada hora
- **Matriz de Transición Markov**: Tabla de probabilidades de transición entre animales
- **Validar Modelo Markov**: Validación cruzada del modelo de Markov
- **Predicción por Hora Específica**: Consulta interactiva por hora

### Pestaña Dashboard
- **Acertados por Día de Semana**: Precisión de cada modelo desglosada por día
- **Acertados por Hora**: Precisión de cada modelo desglosada por hora
- **Patrones del Sorteo**: Análisis de grupos, transiciones, animales fríos/calientes

### Pestaña Modelos ML
- **Entrenar RF / XGB / Patrones**: Entrena modelos (comparten panel de salida)
- **Evaluación IA**: Predicción combinada (ensemble) + matrices RF/XGB por hora
- **Auto-Evaluación**: Evalúa los últimos 46 sorteos contra todos los modelos
- **Predecir con IA**: Genera matriz de predicción Top-20 usando modelos cargados

### Pestaña Scraper
- **Scrapers individuales** por lotería (Lotto Activo, La Granjita, Selva Plus, Rd Int)
- **Scrapear Todo**: Ejecuta los 4 scrapers en secuencia

## Scraping Automático

Los scrapers obtienen resultados desde la web de cada lotería y los guardan en archivos Excel. Ejecutar:

```bash
python scrape_all.py
```

O desde la UI en la pestaña Scraper.

## Entrenamiento de Modelos ML

```bash
python train_all.py
```

O desde la UI en la pestaña Modelos ML (botones Entrenar RF y Entrenar XGB).

## Notas Legales

Este software es **solo para fines educativos y de análisis**. No garantiza ganancias. Consulte los términos de servicio de cada lotería antes de usar. El scraping automatizado y el uso de IA para apuestas pueden estar prohibidos por los términos de algunas plataformas.
