import sys
import os
import datetime
import pandas as pd
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES

CACHE_FILE = 'data/LottoActivoUnificado.xlsx'
SOURCE_FILES = {
    'INT': 'data/LottoActivoINT.xlsx',
    'RD': 'data/LottoActivoRD.xlsx',
    'RDInt': 'data/LottoActivoRDInt.xlsx',
}

CONFIG = {
    'nombre': 'Lotto Activo Unificado',
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'max_numero': 37,
    'excel_file': CACHE_FILE,
    'modelos_dir': 'modelos/lotto_activo_unificado',
    'logger_name': 'lotto_activo_unificado',
}

analizador = Loteria(CONFIG)


def _cache_obsoleto():
    if not os.path.exists(CACHE_FILE):
        return True
    cache_mtime = os.path.getmtime(CACHE_FILE)
    for src in SOURCE_FILES.values():
        if not os.path.exists(src):
            continue
        if os.path.getmtime(src) > cache_mtime:
            return True
    return False


def _generar_cache():
    print("Generando cache unificado...")
    partes = []
    for origen, path in SOURCE_FILES.items():
        if not os.path.exists(path):
            print(f"  AVISO: {path} no encontrado, se omite")
            continue
        df = pd.read_excel(path)
        df['Origen'] = origen
        partes.append(df)
        print(f"  {origen}: {len(df)} registros")

    if not partes:
        print("ERROR: No hay datos de origen para unificar")
        return None

    combined = pd.concat(partes, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Fecha", "Hora", "Origen"], keep="last")
    combined = combined.sort_values(["Fecha", "Hora", "Origen"]).reset_index(drop=True)
    combined.to_excel(CACHE_FILE, index=False)
    print(f"Cache guardado: {len(combined)} registros en {CACHE_FILE}")
    return combined


def cargar_datos_unificados():
    if _cache_obsoleto():
        print("Cache desactualizado o inexistente. Regenerando...")
        df = _generar_cache()
        if df is None:
            return None
        return df
    else:
        df = pd.read_excel(CACHE_FILE)
        print(f"Cache cargado: {len(df)} registros")
        return df

def __getattr__(name):
    return getattr(analizador, name)


def regenerar_cache():
    return _generar_cache()


if __name__ == "__main__":
    from utils import load_and_prepare_data

    excel_file = CONFIG['excel_file']

    print(f"Python version: {sys.version}")

    if not analizador.verificar_diccionario_animales():
        print("ERROR CRITICO: El diccionario de animales tiene problemas.")
        sys.exit(1)

    try:
        datos = cargar_datos_unificados()
        if datos is None:
            print("No hay datos para trabajar.")
            sys.exit(1)

        datos = load_and_prepare_data(excel_file, analizador)
        analizador.main_menu(datos)

    except Exception as e:
        print(f"Error critico: {e}")
        import traceback
        traceback.print_exc()
