import sys
import pandas as pd
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES

CONFIG = {
    'nombre': 'Selva Plus',
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'max_numero': 37,
    'excel_file': 'SelvaPlus.xlsx',
    'modelos_dir': 'modelos/selva_plus',
    'logger_name': 'selva_plus',
}

analizador = Loteria(CONFIG)

verificar_diccionario_animales = analizador.verificar_diccionario_animales
validar_animal = analizador.validar_animal
validar_numero = analizador.validar_numero
calcular_diferencia_ciclica = analizador.calcular_diferencia_ciclica
agregar_caracteristicas_avanzadas = analizador.agregar_caracteristicas_avanzadas
preparar_datos_ml_completo = analizador.preparar_datos_ml_completo
crear_pipeline_ml = analizador.crear_pipeline_ml
calcular_precision_top_k = analizador.calcular_precision_top_k
entrenar_modelo_ml = analizador.entrenar_modelo_ml
predecir_top_k_por_hora = analizador.predecir_top_k_por_hora
optimizar_hiperparametros_rf = analizador.optimizar_hiperparametros_rf
optimizar_hiperparametros_xgb = analizador.optimizar_hiperparametros_xgb
entrenar_modelo_con_optimizacion = analizador.entrenar_modelo_con_optimizacion
guardar_modelo = analizador.guardar_modelo
cargar_modelo = analizador.cargar_modelo
cargar_ultimo_modelo = analizador.cargar_ultimo_modelo
random_forest_optimizado = analizador.random_forest_optimizado
xgboost_optimizado = analizador.xgboost_optimizado
evaluacion_estrategia_frecuencia = analizador.evaluacion_estrategia_frecuencia
evaluacion_estrategia_ia = analizador.evaluacion_estrategia_ia
simular_estrategia = analizador.simular_estrategia
mostrar_matriz_prediccion = analizador.mostrar_matriz_prediccion
prediccion_hoy_ensemble = analizador.prediccion_hoy_ensemble
prediccion_completa_hoy = analizador.prediccion_completa_hoy
evaluar_predicciones_historicas = analizador.evaluar_predicciones_historicas
analizar_aciertos_por_dia_semana = analizador.analizar_aciertos_por_dia_semana
analizar_aciertos_por_hora = analizador.analizar_aciertos_por_hora
analizar_patrones_sorteo = analizador.analizar_patrones_sorteo
generar_matriz_probabilidad = analizador.generar_matriz_probabilidad
matriz_probabilidad_transicion = analizador.matriz_probabilidad_transicion
mejor_prediccion_siguiente = analizador.mejor_prediccion_siguiente
probabilidad_maxima_por_hora = analizador.probabilidad_maxima_por_hora
prediccion_markov_hora = analizador.prediccion_markov_hora
validar_modelo_markov = analizador.validar_modelo_markov
prediccion_por_hora_especifica = analizador.prediccion_por_hora_especifica
agregar_datos_al_excel = analizador.agregar_datos_al_excel
evaluacion_estrategia_solo_manana = analizador.evaluacion_estrategia_solo_manana
evaluacion_estrategia_filtrada = analizador.evaluacion_estrategia_filtrada
analisis_estadistico_avanzado = analizador.analisis_estadistico_avanzado
patrones_dias_rentables = analizador.patrones_dias_rentables
predictor_dia_actual = analizador.predictor_dia_actual
ver_ultimos_registros_y_faltantes = analizador.ver_ultimos_registros_y_faltantes
ver_estado_actual_dia = analizador.ver_estado_actual_dia
analizar_rachas_tempranas = analizador.analizar_rachas_tempranas
probar_umbrales_rachas = analizador.probar_umbrales_rachas
prediccion_dia_completo = analizador.prediccion_dia_completo
top_25_general = analizador.top_25_general
get_parejas_horarias = analizador.get_parejas_horarias
get_matriz_global_por_animal = analizador.get_matriz_global_por_animal
get_matriz_hora_por_animal = analizador.get_matriz_hora_por_animal
get_matriz_segundo_orden = analizador.get_matriz_segundo_orden
get_matriz_global = analizador.get_matriz_global
get_matriz_hora = analizador.get_matriz_hora
get_prediccion_combinada = analizador.get_prediccion_combinada
num_int_a_animal = analizador.num_int_a_animal
animal_a_num_int = analizador.animal_a_num_int
main_menu = analizador.main_menu

if __name__ == "__main__":
    from utils import auto_scrape_missing_dates, load_and_prepare_data

    excel_file = CONFIG['excel_file']

    print(f"Python version: {sys.version}")

    if not analizador.verificar_diccionario_animales():
        print("ERROR CRITICO: El diccionario de animales tiene problemas.")
        sys.exit(1)

    try:
        datos = pd.read_excel(excel_file)
        print(f"Archivo cargado: {len(datos)} registros")

        try:
            from scraper_selva_plus import scrape_date, save_to_excel
            datos = auto_scrape_missing_dates(datos, scrape_date, save_to_excel, excel_file)
        except Exception as e:
            print(f"Auto-scraper: error ({e})")

        datos = load_and_prepare_data(excel_file, analizador)
        analizador.main_menu(datos)

    except FileNotFoundError:
        print(f"Archivo no encontrado. Creando '{excel_file}'...")
        datos = pd.DataFrame(columns=['Fecha', 'Hora', 'Animal', 'Numero'])
        datos.to_excel(excel_file, index=False)
        print(f"Creado '{excel_file}'. Agrega datos y ejecuta nuevamente.")
    except Exception as e:
        print(f"Error critico: {e}")
        import traceback
        traceback.print_exc()
