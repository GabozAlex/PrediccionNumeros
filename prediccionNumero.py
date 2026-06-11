"""
Light wrapper module kept for backward compatibility.

Delegates all functionality to the Lotto Activo analyzer.
See lotto_activo, la_granjita, selva_plus for specific lottery wrappers.
"""

import warnings
import sys
import pandas as pd

try:
    from lotto_activo import analizador as _analizador
    analizador = _analizador
except Exception as e:
    warnings.warn(f"Fallo delegacion en prediccionNumero: {e}")
    analizador = None


def _not_available(*a, **k):
    raise RuntimeError("Funcion no disponible: el wrapper de loteria no pudo cargarse.")


_names = [
    'verificar_diccionario_animales', 'validar_animal', 'validar_numero',
    'calcular_diferencia_ciclica', 'agregar_caracteristicas_avanzadas',
    'preparar_datos_ml_completo', 'crear_pipeline_ml', 'calcular_precision_top_k',
    'entrenar_modelo_ml', 'predecir_top_k_por_hora', 'optimizar_hiperparametros_rf',
    'optimizar_hiperparametros_xgb', 'entrenar_modelo_con_optimizacion',
    'guardar_modelo', 'cargar_modelo', 'cargar_ultimo_modelo',
    'random_forest_optimizado', 'xgboost_optimizado', 'evaluacion_estrategia_frecuencia',
    'evaluacion_estrategia_ia', 'simular_estrategia', 'mostrar_matriz_prediccion',
    'prediccion_hoy_ensemble', 'prediccion_completa_hoy', 'evaluar_predicciones_historicas',
    'analizar_aciertos_por_dia_semana', 'analizar_aciertos_por_hora', 'analizar_patrones_sorteo',
    'generar_matriz_probabilidad', 'matriz_probabilidad_transicion', 'mejor_prediccion_siguiente',
    'probabilidad_maxima_por_hora', 'prediccion_markov_hora', 'validar_modelo_markov',
    'prediccion_por_hora_especifica', 'agregar_datos_al_excel', 'evaluacion_estrategia_solo_manana',
    'evaluacion_estrategia_filtrada', 'analisis_estadistico_avanzado', 'patrones_dias_rentables',
    'predictor_dia_actual', 'ver_ultimos_registros_y_faltantes', 'ver_estado_actual_dia',
    'analizar_rachas_tempranas', 'probar_umbrales_rachas', 'main_menu'
]

for _n in _names:
    if analizador is not None:
        globals()[_n] = getattr(analizador, _n, _not_available)
    else:
        globals()[_n] = _not_available


if __name__ == "__main__":
    print(f"Python version: {sys.version}")

    if not verificar_diccionario_animales():
        print("ERROR CRITICO: El diccionario de animales tiene problemas.")
        sys.exit(1)

    from utils import auto_scrape_missing_dates, load_and_prepare_data
    from lotto_activo import CONFIG

    excel_file = CONFIG['excel_file']

    try:
        datos = pd.read_excel(excel_file)
        print(f"Archivo cargado: {len(datos)} registros")

        try:
            from scraper_lotto import scrape_date, save_to_excel
            datos = auto_scrape_missing_dates(datos, scrape_date, save_to_excel, excel_file)
        except Exception as e:
            print(f"Auto-scraper: error ({e})")

        datos = load_and_prepare_data(excel_file, _analizador)
        main_menu(datos)

    except FileNotFoundError:
        print(f"Archivo no encontrado. Creando '{excel_file}'...")
        datos = pd.DataFrame(columns=["Fecha", "Hora", "Animal", "Numero"])
        datos.to_excel(excel_file, index=False)
        print(f"Creado '{excel_file}'. Agrega datos y ejecuta nuevamente.")
    except Exception as e:
        print(f"Error critico: {e}")
        import traceback
        traceback.print_exc()
