import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES, ANIMAL_A_NUM_INT

CONFIG = {
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'logger_name': 'sim',
    'max_numero': 37,
    'modelos_dir': '/tmp/sim_modelos',
}

EXCEL_FILES = [
    'data/LottoActivoINT.xlsx',
    'data/LottoActivoRDInt.xlsx',
    'data/SelvaPlus.xlsx',
    'data/LaGranjita.xlsx',
]

RESULTADOS = {}

def probar_funcion(loteria, datos, nombre, func, *args, **kwargs):
    print(f"  {nombre}...", end=" ")
    try:
        if nombre in ('get_matriz_global', 'get_matriz_hora', 'get_matriz_segundo_orden',
                      'get_prediccion_combinada', 'analizar_coocurrencias',
                      'analizar_coocurrencias_por_rango', 'analizar_frecuencia_por_dia_semana',
                      'analizar_secuencias_aciertos_fallos', 'comparar_estrategias',
                      'construir_matrices_markov', 'preparar_datos_markov'):
            func(datos, *args, **kwargs)
        elif nombre == 'get_parejas_horarias':
            func()
        else:
            func(*args, **kwargs)
        print("OK")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

for fname in EXCEL_FILES:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    print(f"\n{'='*70}")
    print(f"  ARCHIVO: {fname}")
    print(f"{'='*70}")

    if not os.path.exists(path):
        print(f"  NO ENCONTRADO, saltando")
        RESULTADOS[fname] = {'error': 'archivo no encontrado'}
        continue

    loteria = Loteria(CONFIG)

    try:
        datos = pd.read_excel(path)
        datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
        datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce')
        datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
        datos['Hora'] = datos['Hora'].astype(str).str.strip().str.zfill(8)
        datos['Timestamp'] = pd.to_datetime(
            datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce'
        )
        datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
        datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip()
        datos = datos.sort_values(by='Timestamp').reset_index(drop=True)
        datos['Num_Int'] = datos['Animal'].map(ANIMAL_A_NUM_INT)
        datos = datos.dropna(subset=['Num_Int']).reset_index(drop=True)
        datos['Num_Int'] = datos['Num_Int'].astype(int)

        print(f"  Registros: {len(datos)}")
        print(f"  Fechas: {datos['Fecha'].min()} a {datos['Fecha'].max()}")
        print(f"  Num_Int: {datos['Num_Int'].min()}-{datos['Num_Int'].max()}")

        resultados = {}
        pruebas = [
            ('analizar_coocurrencias', [], {'top_k': 25}),
            ('analizar_coocurrencias_por_rango', [], {'top_k': 25}),
            ('analizar_frecuencia_por_dia_semana', [], {'top_k': 15}),
            ('analizar_secuencias_aciertos_fallos', [], {'top_k': 25}),
            ('comparar_estrategias', [], {'top_k': 25}),
            ('get_parejas_horarias', [], {}),
        ]

        for nombre, args, kwargs in pruebas:
            resultados[nombre] = probar_funcion(loteria, datos, nombre, getattr(loteria, nombre), *args, **kwargs)

        # get_prediccion_combinada: needs a sample origin
        parejas = loteria.get_parejas_horarias()
        if parejas:
            h_o, h_d = parejas[0]
            resultados['get_prediccion_combinada'] = probar_funcion(
                loteria, datos, 'get_prediccion_combinada',
                loteria.get_prediccion_combinada, 5, h_o, h_d, top_k=10)

        # get_matriz_segundo_orden
        resultados['get_matriz_segundo_orden'] = probar_funcion(
            loteria, datos, 'get_matriz_segundo_orden',
            loteria.get_matriz_segundo_orden, 5, 10, top_k=38)

        # preparar_datos_markov + construir_matrices_markov
        resultados['construir_matrices_markov'] = probar_funcion(
            loteria, datos, 'construir_matrices_markov',
            loteria.construir_matrices_markov,
            loteria.preparar_datos_markov(datos))

        # get_matriz_global
        resultados['get_matriz_global'] = probar_funcion(
            loteria, datos, 'get_matriz_global',
            loteria.get_matriz_global, top_k=38)

        # get_matriz_hora with a sample pair
        if parejas:
            h_o, h_d = parejas[0]
            resultados['get_matriz_hora'] = probar_funcion(
                loteria, datos, 'get_matriz_hora',
                loteria.get_matriz_hora, h_o, h_d, top_k=38)

        aciertos = sum(1 for v in resultados.values() if v)
        total = len(resultados)
        print(f"\n  -> Resultados: {aciertos}/{total} pruebas OK")
        RESULTADOS[fname] = resultados

    except Exception as e:
        print(f"  ERROR CRITICO: {e}")
        traceback.print_exc()
        RESULTADOS[fname] = {'error': str(e)}

print(f"\n\n{'='*70}")
print(f"  RESUMEN GLOBAL")
print(f"{'='*70}")
total_ok = 0
total_all = 0
for fname, res in RESULTADOS.items():
    if 'error' in res:
        print(f"  {fname}: ERROR -> {res['error']}")
    else:
        ok = sum(1 for v in res.values() if v)
        all_ = len(res)
        total_ok += ok
        total_all += all_
        print(f"  {fname}: {ok}/{all_}")
print(f"\n  TOTAL: {total_ok}/{total_all}")
