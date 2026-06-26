import sys
import pandas as pd
from loteria_base import Loteria
from utils import ANIMALES_38, GRUPOS_ANIMALES

CONFIG = {
    'nombre': 'Lotto Activo',
    'animales': ANIMALES_38,
    'grupos_animales': GRUPOS_ANIMALES,
    'max_numero': 37,
    'excel_file': 'data/LottoActivoINT.xlsx',
    'modelos_dir': 'modelos/lotto_activo',
    'logger_name': 'lotto_activo',
}

analizador = Loteria(CONFIG)

def __getattr__(name):
    return getattr(analizador, name)

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
            from scraper_lotto import scrape_date, save_to_excel
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
