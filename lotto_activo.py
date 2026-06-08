import sys
import pandas as pd
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES

CONFIG = {
    'nombre': 'Lotto Activo',
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'max_numero': 37,
    'excel_file': 'LottoActivoCompleto.xlsx',
    'modelos_dir': 'modelos/lotto_activo',
    'logger_name': 'lotto_activo',
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
get_matriz_combinada_por_animal = analizador.get_matriz_combinada_por_animal
get_matriz_segundo_orden = analizador.get_matriz_segundo_orden
main_menu = analizador.main_menu

if __name__ == "__main__":
    from utils import mostrar_menu
    import datetime as _dt

    datosLotto = CONFIG['excel_file']
    datos = None

    print(f"Python version: {sys.version}")

    if not analizador.verificar_diccionario_animales():
        print("ERROR CRITICO: El diccionario de animales tiene problemas.")
        sys.exit(1)

    try:
        datos = pd.read_excel(datosLotto)
        print(f"Archivo cargado: {len(datos)} registros")

        from scraper_lotto import scrape_date, save_to_excel

        fechas_existentes = set(pd.to_datetime(datos['Fecha']).dt.strftime("%Y-%m-%d").unique())
        hoy = _dt.date.today()
        ultima = _dt.datetime.strptime(max(fechas_existentes), "%Y-%m-%d").date()
        if ultima < hoy:
            desde = ultima + _dt.timedelta(days=1)
            faltantes = []
            d = desde
            while d <= hoy:
                ds = d.strftime("%Y-%m-%d")
                if ds not in fechas_existentes:
                    faltantes.append(ds)
                d += _dt.timedelta(days=1)
            if faltantes:
                print(f"Auto-scraper: {len(faltantes)} fechas faltantes ({desde} -> {hoy})")
                todos = []
                for i, fs in enumerate(faltantes):
                    r = scrape_date(fs)
                    todos.extend(r)
                    if (i+1) % 10 == 0:
                        print(f"  Progreso: {i+1}/{len(faltantes)}")
                    import time
                    time.sleep(1.5)
                if todos:
                    df_nuevo = pd.DataFrame(todos)
                    save_to_excel(df_nuevo, datosLotto)
                    datos = pd.read_excel(datosLotto)
                    print(f"Auto-scraper: {len(df_nuevo)} nuevos registros")
                else:
                    print("Auto-scraper: sin nuevos registros")
    except Exception as e:
        print(f"Auto-scraper: error ({e})")

    try:
        datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
        datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce')
        numeros_invalidos = datos['Numero'].isna().sum()
        if numeros_invalidos > 0:
            print(f"  {numeros_invalidos} registros con numeros invalidos")

        datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
        datos['Hora'] = datos['Hora'].astype(str).str.strip().str.zfill(8)
        datos['Timestamp'] = pd.to_datetime(datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce')
        tiempos_invalidos = datos['Timestamp'].isna().sum()
        if tiempos_invalidos > 0:
            print(f"  {tiempos_invalidos} registros con timestamp invalido")

        datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
        datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip()
        datos = datos.sort_values(by='Timestamp').reset_index(drop=True)
        datos = analizador.agregar_caracteristicas_avanzadas(datos)

        print("\nRESUMEN DE DATOS:")
        print(f"  Total registros: {len(datos)}")
        print(f"  Rango fechas: {datos['Timestamp'].min()} a {datos['Timestamp'].max()}")
        print(f"  Animales unicos: {datos['Animal'].nunique()}")
        print(f"  Horas unicas: {datos['Solo_hora'].nunique()}")

        analizador.main_menu(datos)

    except FileNotFoundError:
        print("Archivo no encontrado. Creando archivo de ejemplo...")
        datos = pd.DataFrame(columns=['Fecha', 'Hora', 'Animal', 'Numero'])
        datos.to_excel(datosLotto, index=False)
        print(f"Se creo '{datosLotto}'. Agrega datos y ejecuta nuevamente.")
    except Exception as e:
        print(f"Error critico: {e}")
        import traceback
        traceback.print_exc()
