import pandas as pd
import sys
import os

LOTTERIES = [
    {
        'name': 'Lotto Activo',
        'module': 'lotto_activo',
        'analizador': None,
    },
    {
        'name': 'La Granjita',
        'module': 'la_granjita',
        'analizador': None,
    },
    {
        'name': 'Selva Plus',
        'module': 'selva_plus',
        'analizador': None,
    },
]


def import_modules():
    for lot in LOTTERIES:
        mod = __import__(lot['module'])
        lot['analizador'] = mod.analizador
        lot['config'] = mod.CONFIG


def load_data(config):
    excel_file = config['excel_file']
    if not os.path.exists(excel_file):
        print(f"  Archivo no encontrado: {excel_file}")
        return None

    datos = pd.read_excel(excel_file)
    datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
    datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce')
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
    datos['Hora'] = datos['Hora'].astype(str).str.strip()
    datos['Timestamp'] = pd.to_datetime(
        datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce'
    )
    datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
    datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip()
    datos = datos.sort_values(by='Timestamp').reset_index(drop=True)
    datos = config['analizador'].agregar_caracteristicas_avanzadas(datos)
    return datos


def train_lottery(lot, tipo):
    name = lot['name']
    analizador = lot['analizador']
    config = lot['config']

    print(f"\n{'='*60}")
    print(f"  ENTRENANDO {tipo.upper()} - {name}")
    print(f"{'='*60}")

    datos = load_data(config)
    if datos is None or len(datos) < 50:
        print(f"  Datos insuficientes: {len(datos) if datos is not None else 0} registros")
        return

    if tipo == 'rf':
        analizador.random_forest_optimizado(datos)
    elif tipo == 'xgb':
        analizador.xgboost_optimizado(datos)


def main():
    import_modules()

    print("TRAIN ALL - ENTRENAR MODELOS EN TODAS LAS LOTERIAS")
    print()
    print("1. Entrenar Random Forest en todas")
    print("2. Entrenar XGBoost en todas")
    print("3. Entrenar ambos en todas")

    option = input("Selecciona opcion: ").strip()

    tipos = []
    if option == '1':
        tipos = ['rf']
    elif option == '2':
        tipos = ['xgb']
    elif option == '3':
        tipos = ['rf', 'xgb']
    else:
        print("Opcion no valida")
        return

    for tipo in tipos:
        for lot in LOTTERIES:
            train_lottery(lot, tipo)
            import time
            time.sleep(1)

    print(f"\nEntrenamiento completado para todas las loterias")


if __name__ == "__main__":
    main()
