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
    verificar_diccionario_animales = _analizador.verificar_diccionario_animales
    validar_animal = _analizador.validar_animal
    validar_numero = _analizador.validar_numero
    calcular_diferencia_ciclica = _analizador.calcular_diferencia_ciclica
    agregar_caracteristicas_avanzadas = _analizador.agregar_caracteristicas_avanzadas
    preparar_datos_ml_completo = _analizador.preparar_datos_ml_completo
    crear_pipeline_ml = _analizador.crear_pipeline_ml
    calcular_precision_top_k = _analizador.calcular_precision_top_k
    entrenar_modelo_ml = _analizador.entrenar_modelo_ml
    predecir_top_k_por_hora = _analizador.predecir_top_k_por_hora
    optimizar_hiperparametros_rf = _analizador.optimizar_hiperparametros_rf
    optimizar_hiperparametros_xgb = _analizador.optimizar_hiperparametros_xgb
    entrenar_modelo_con_optimizacion = _analizador.entrenar_modelo_con_optimizacion
    guardar_modelo = _analizador.guardar_modelo
    cargar_modelo = _analizador.cargar_modelo
    cargar_ultimo_modelo = _analizador.cargar_ultimo_modelo
    random_forest_optimizado = _analizador.random_forest_optimizado
    xgboost_optimizado = _analizador.xgboost_optimizado
    evaluacion_estrategia_frecuencia = _analizador.evaluacion_estrategia_frecuencia
    evaluacion_estrategia_ia = _analizador.evaluacion_estrategia_ia
    simular_estrategia = _analizador.simular_estrategia
    mostrar_matriz_prediccion = _analizador.mostrar_matriz_prediccion
    prediccion_hoy_ensemble = _analizador.prediccion_hoy_ensemble
    prediccion_completa_hoy = _analizador.prediccion_completa_hoy
    evaluar_predicciones_historicas = _analizador.evaluar_predicciones_historicas
    analizar_aciertos_por_dia_semana = _analizador.analizar_aciertos_por_dia_semana
    analizar_aciertos_por_hora = _analizador.analizar_aciertos_por_hora
    analizar_patrones_sorteo = _analizador.analizar_patrones_sorteo
    generar_matriz_probabilidad = _analizador.generar_matriz_probabilidad
    matriz_probabilidad_transicion = _analizador.matriz_probabilidad_transicion
    mejor_prediccion_siguiente = _analizador.mejor_prediccion_siguiente
    probabilidad_maxima_por_hora = _analizador.probabilidad_maxima_por_hora
    prediccion_markov_hora = _analizador.prediccion_markov_hora
    validar_modelo_markov = _analizador.validar_modelo_markov
    prediccion_por_hora_especifica = _analizador.prediccion_por_hora_especifica
    agregar_datos_al_excel = _analizador.agregar_datos_al_excel
    evaluacion_estrategia_solo_manana = _analizador.evaluacion_estrategia_solo_manana
    evaluacion_estrategia_filtrada = _analizador.evaluacion_estrategia_filtrada
    analisis_estadistico_avanzado = _analizador.analisis_estadistico_avanzado
    patrones_dias_rentables = _analizador.patrones_dias_rentables
    predictor_dia_actual = _analizador.predictor_dia_actual
    ver_ultimos_registros_y_faltantes = _analizador.ver_ultimos_registros_y_faltantes
    ver_estado_actual_dia = _analizador.ver_estado_actual_dia
    analizar_rachas_tempranas = _analizador.analizar_rachas_tempranas
    probar_umbrales_rachas = _analizador.probar_umbrales_rachas
    main_menu = _analizador.main_menu
except Exception as e:
    warnings.warn(f"Fallo delegacion en prediccionNumero: {e}")

    def _not_available(*a, **k):
        raise RuntimeError("Funcion no disponible: el wrapper de loteria no pudo cargarse.")

    analizador = None
    verificar_diccionario_animales = _not_available
    validar_animal = _not_available
    validar_numero = _not_available
    calcular_diferencia_ciclica = _not_available
    agregar_caracteristicas_avanzadas = _not_available
    preparar_datos_ml_completo = _not_available
    crear_pipeline_ml = _not_available
    calcular_precision_top_k = _not_available
    entrenar_modelo_ml = _not_available
    predecir_top_k_por_hora = _not_available
    optimizar_hiperparametros_rf = _not_available
    optimizar_hiperparametros_xgb = _not_available
    entrenar_modelo_con_optimizacion = _not_available
    guardar_modelo = _not_available
    cargar_modelo = _not_available
    cargar_ultimo_modelo = _not_available
    random_forest_optimizado = _not_available
    xgboost_optimizado = _not_available
    evaluacion_estrategia_frecuencia = _not_available
    evaluacion_estrategia_ia = _not_available
    simular_estrategia = _not_available
    mostrar_matriz_prediccion = _not_available
    prediccion_hoy_ensemble = _not_available
    prediccion_completa_hoy = _not_available
    evaluar_predicciones_historicas = _not_available
    analizar_aciertos_por_dia_semana = _not_available
    analizar_aciertos_por_hora = _not_available
    analizar_patrones_sorteo = _not_available
    generar_matriz_probabilidad = _not_available
    matriz_probabilidad_transicion = _not_available
    mejor_prediccion_siguiente = _not_available
    probabilidad_maxima_por_hora = _not_available
    prediccion_markov_hora = _not_available
    validar_modelo_markov = _not_available
    prediccion_por_hora_especifica = _not_available
    agregar_datos_al_excel = _not_available
    evaluacion_estrategia_solo_manana = _not_available
    evaluacion_estrategia_filtrada = _not_available
    analisis_estadistico_avanzado = _not_available
    patrones_dias_rentables = _not_available
    predictor_dia_actual = _not_available
    ver_ultimos_registros_y_faltantes = _not_available
    ver_estado_actual_dia = _not_available
    analizar_rachas_tempranas = _not_available
    probar_umbrales_rachas = _not_available
    main_menu = _not_available


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
