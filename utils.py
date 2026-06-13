import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_requests_session(retries: int = 3, backoff_factor: float = 0.3, status_forcelist=(500, 502, 504)):
    """Create a requests Session with retry/backoff configured.

    This centralizes HTTP robustness for all scrapers.
    """
    import requests
    from requests.adapters import HTTPAdapter
    try:
        # urllib3 Retry is available via requests.packages.urllib3 or urllib3
        from urllib3.util.retry import Retry
    except Exception:
        from requests.packages.urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update(HEADERS)
    return session

ANIMALES_38 = [
    "DELFIN", "BALLENA", "CARNERO", "TORO", "CIEMPIES", "ALACRAN",
    "LEON", "RANA", "PERICO", "RATON", "AGUILA", "TIGRE", "GATO",
    "CABALLO", "MONO", "PALOMA", "ZORRO", "OSO", "PAVO", "BURRO",
    "CHIVO", "COCHINO", "GALLO", "CAMELLO", "CEBRA", "IGUANA",
    "GALLINA", "VACA", "PERRO", "ZAMURO", "ELEFANTE", "CAIMAN",
    "LAPA", "ARDILLA", "PESCADO", "VENADO", "JIRAFA", "CULEBRA"
]

ANIMAL_A_NUM_INT = {"DELFIN": 0}
_idx = 1
for _a in ANIMALES_38:
    if _a in ("DELFIN", "BALLENA"):
        continue
    ANIMAL_A_NUM_INT[_a] = _idx
    _idx += 1
ANIMAL_A_NUM_INT["BALLENA"] = 37

NUM_INT_A_ANIMAL = {v: k for k, v in ANIMAL_A_NUM_INT.items()}

GRUPOS_ANIMALES = {
    "MAMIFERO": {"TORO","LEON","TIGRE","GATO","CABALLO","MONO","ZORRO","OSO","BURRO","CHIVO","COCHINO","CAMELLO","CEBRA","VACA","PERRO","ELEFANTE","VENADO","JIRAFA","RATON","CARNERO","LAPA","ARDILLA"},
    "AVE":      {"PERICO","AGUILA","PALOMA","PAVO","GALLO","GALLINA","ZAMURO"},
    "ACUATICO": {"DELFIN","BALLENA","PESCADO"},
    "REPTIL":   {"RANA","IGUANA","CAIMAN","CULEBRA"},
    "INSECTO":  {"CIEMPIES","ALACRAN"},
}

HORA_MAP_12_TO_24 = {
    "08:00 AM": "08:00:00",
    "09:00 AM": "09:00:00",
    "10:00 AM": "10:00:00",
    "11:00 AM": "11:00:00",
    "12:00 PM": "12:00:00",
    "01:00 PM": "13:00:00",
    "02:00 PM": "14:00:00",
    "03:00 PM": "15:00:00",
    "04:00 PM": "16:00:00",
    "05:00 PM": "17:00:00",
    "06:00 PM": "18:00:00",
    "07:00 PM": "19:00:00",
}

# Also include common :30 times used by some scrapers (e.g. RD Int)
HORA_MAP_12_TO_24.update({
    "08:30 AM": "08:30:00",
    "09:30 AM": "09:30:00",
    "10:30 AM": "10:30:00",
    "11:30 AM": "11:30:00",
    "12:30 PM": "12:30:00",
    "01:30 PM": "13:30:00",
    "02:30 PM": "14:30:00",
    "03:30 PM": "15:30:00",
    "04:30 PM": "16:30:00",
    "05:30 PM": "17:30:00",
    "06:30 PM": "18:30:00",
    "07:30 PM": "19:30:00",
})

GUACHARITO_ANIMALES = [
    "BALLENA","DELFIN","CARNERO","TORO","CIEMPIES","ALACRAN","LEON","RANA",
    "PERICO","RATON","AGUILA","TIGRE","GATO","CABALLO","MONO","PALOMA",
    "ZORRO","OSO","PAVO","BURRO","CHIVO","COCHINO","GALLO","CAMELLO",
    "CEBRA","IGUANA","GALLINA","VACA","PERRO","ZAMURO","ELEFANTE","CAIMAN",
    "LAPA","ARDILLA","PESCADO","VENADO","JIRAFA","CULEBRA","TORTUGA","BUFFALO",
    "LECHUZA","AVISPA","CANGURO","TUCAN","MARIPOSA","CHIGUIRE","GARZA","PUMA",
    "PAVO REAL","PUERCOESPIN","PEREZA","CANARIO","PELICANO","PULPO","CARACOL",
    "GRILLO","OSO HORMIGUERO","TIBURON","PATO","HORMIGA","PANTERA","CAMALEON",
    "PANDA","CACHICAMO","CANGREJO","GAVILAN","ARANA","LOBO","AVESTRUZ",
    "JAGUAR","CONEJO","BISONTE","GUACAMAYA","GORILA","HIPOPOTAMO","TURPIAL",
    "GUACHARO","RINOCERONTE","PINGUINO","ANTILOPE","CALAMAR","MURCIELAGO",
    "CUERVO","CUCARACHA","BUHO","CAMARON","HAMSTER","BUEY","CABRA",
    "ERIZO DE MAR","ANGUILLA","HURON","MORROCOY","CISNE","GAVIOTA","PAUJIL",
    "ESCARABAJO","CABALLITO DE MAR","LORO","COCODRILO","GUACHARITO"
]

GUACHARITO_NUMERO_A_ANIMAL = {
    0:"DELFIN", "00":"BALLENA", 1:"CARNERO", 2:"TORO", 3:"CIEMPIES", 4:"ALACRAN",
    5:"LEON", 6:"RANA", 7:"PERICO", 8:"RATON", 9:"AGUILA", 10:"TIGRE",
    11:"GATO", 12:"CABALLO", 13:"MONO", 14:"PALOMA", 15:"ZORRO", 16:"OSO",
    17:"PAVO", 18:"BURRO", 19:"CHIVO", 20:"COCHINO", 21:"GALLO", 22:"CAMELLO",
    23:"CEBRA", 24:"IGUANA", 25:"GALLINA", 26:"VACA", 27:"PERRO", 28:"ZAMURO",
    29:"ELEFANTE", 30:"CAIMAN", 31:"LAPA", 32:"ARDILLA", 33:"PESCADO",
    34:"VENADO", 35:"JIRAFA", 36:"CULEBRA", 37:"TORTUGA", 38:"BUFFALO",
    39:"LECHUZA", 40:"AVISPA", 41:"CANGURO", 42:"TUCAN", 43:"MARIPOSA",
    44:"CHIGUIRE", 45:"GARZA", 46:"PUMA", 47:"PAVO REAL", 48:"PUERCOESPIN",
    49:"PEREZA", 50:"CANARIO", 51:"PELICANO", 52:"PULPO", 53:"CARACOL",
    54:"GRILLO", 55:"OSO HORMIGUERO", 56:"TIBURON", 57:"PATO", 58:"HORMIGA",
    59:"PANTERA", 60:"CAMALEON", 61:"PANDA", 62:"CACHICAMO", 63:"CANGREJO",
    64:"GAVILAN", 65:"ARANA", 66:"LOBO", 67:"AVESTRUZ", 68:"JAGUAR",
    69:"CONEJO", 70:"BISONTE", 71:"GUACAMAYA", 72:"GORILA", 73:"HIPOPOTAMO",
    74:"TURPIAL", 75:"GUACHARO", 76:"RINOCERONTE", 77:"PINGUINO", 78:"ANTILOPE",
    79:"CALAMAR", 80:"MURCIELAGO", 81:"CUERVO", 82:"CUCARACHA", 83:"BUHO",
    84:"CAMARON", 85:"HAMSTER", 86:"BUEY", 87:"CABRA", 88:"ERIZO DE MAR",
    89:"ANGUILLA", 90:"HURON", 91:"MORROCOY", 92:"CISNE", 93:"GAVIOTA",
    94:"PAUJIL", 95:"ESCARABAJO", 96:"CABALLITO DE MAR", 97:"LORO",
    98:"COCODRILO", 99:"GUACHARITO"
}


def setup_logging(nombre="lotto_predictor"):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger(nombre)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = RotatingFileHandler(
        f'logs/{nombre}.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def mostrar_menu(titulo, opciones):
    print(f"\n--- **{titulo}** ---")
    for i, opcion in enumerate(opciones, 1):
        print(f"{i}. {opcion}")
    print("-------------------")
    while True:
        try:
            seleccion = input("Selecciona una opcion (numero): ")
            numero_seleccionado = int(seleccion)
            if 1 <= numero_seleccionado <= len(opciones):
                return numero_seleccionado
            else:
                print(f"Error: El numero debe estar entre 1 y {len(opciones)}")
        except ValueError:
            print("Error: Por favor, ingresa un numero valido.")


def auto_scrape_missing_dates(datos, scrape_func, save_func, excel_file, delay=1.5):
    """Scrapea fechas faltantes desde la ultima fecha en datos hasta hoy."""
    import datetime as _dt
    import time

    if datos is None or datos.empty:
        print("Auto-scraper: no hay datos existentes")
        return datos

    fechas_existentes = set(pd.to_datetime(datos['Fecha']).dt.strftime("%Y-%m-%d").unique())
    hoy = _dt.date.today()
    ultima = _dt.datetime.strptime(max(fechas_existentes), "%Y-%m-%d").date()

    if ultima >= hoy:
        print("Auto-scraper: datos al dia")
        return datos

    desde = ultima + _dt.timedelta(days=1)
    faltantes = []
    d = desde
    while d <= hoy:
        ds = d.strftime("%Y-%m-%d")
        if ds not in fechas_existentes:
            faltantes.append(ds)
        d += _dt.timedelta(days=1)

    if not faltantes:
        print("Auto-scraper: sin fechas faltantes")
        return datos

    print(f"Auto-scraper: {len(faltantes)} fechas faltantes ({desde} -> {hoy})")
    todos = []
    for i, fs in enumerate(faltantes):
        r = scrape_func(fs)
        todos.extend(r)
        if (i + 1) % 10 == 0:
            print(f"  Progreso: {i+1}/{len(faltantes)}")
        time.sleep(delay)

    if todos:
        df_nuevo = pd.DataFrame(todos)
        save_func(df_nuevo, excel_file)
        datos = pd.read_excel(excel_file)
        print(f"Auto-scraper: {len(df_nuevo)} nuevos registros")
    else:
        print("Auto-scraper: sin nuevos registros")

    return datos


def load_and_prepare_data(excel_file, analizador):
    """Carga datos desde excel y aplica limpieza + feature engineering comun."""
    datos = pd.read_excel(excel_file)
    datos['Animal'] = datos['Animal'].astype(str).str.strip().str.upper()
    datos['Numero'] = pd.to_numeric(datos['Numero'], errors='coerce')
    if datos['Numero'].isna().sum() > 0:
        # Use the analyzer's logger when available, fallback to printing
        try:
            analizador.logger.warning(f"{datos['Numero'].isna().sum()} registros con numeros invalidos")
        except Exception:
            print(f"  {datos['Numero'].isna().sum()} registros con numeros invalidos")

    datos['Fecha'] = pd.to_datetime(datos['Fecha'], errors='coerce').dt.date
    datos['Hora'] = datos['Hora'].astype(str).str.strip().str.zfill(8)
    datos['Timestamp'] = pd.to_datetime(
        datos['Fecha'].astype(str) + ' ' + datos['Hora'], errors='coerce'
    )
    datos = datos.dropna(subset=['Timestamp']).reset_index(drop=True)
    datos['Solo_hora'] = datos['Timestamp'].dt.strftime('%I:%M %p').str.strip()
    datos = datos.sort_values(by='Timestamp').reset_index(drop=True)
    datos['Num_Int'] = datos['Animal'].map(ANIMAL_A_NUM_INT)
    if datos['Num_Int'].isna().sum() > 0:
        animales_sin_mapeo = datos.loc[datos['Num_Int'].isna(), 'Animal'].unique()
        print(f"  {len(animales_sin_mapeo)} animales sin mapeo interno: {animales_sin_mapeo}")
        datos = datos.dropna(subset=['Num_Int']).reset_index(drop=True)
    datos['Num_Int'] = datos['Num_Int'].astype(int)
    datos = analizador.agregar_caracteristicas_avanzadas(datos)

    # Summary via logger if available
    try:
        analizador.logger.info("\nRESUMEN DE DATOS:")
        analizador.logger.info(f"  Total registros: {len(datos)}")
        analizador.logger.info(f"  Rango fechas: {datos['Timestamp'].min()} a {datos['Timestamp'].max()}")
        analizador.logger.info(f"  Animales unicos: {datos['Animal'].nunique()}")
    except Exception:
        print(f"\nRESUMEN DE DATOS:")
        print(f"  Total registros: {len(datos)}")
        print(f"  Rango fechas: {datos['Timestamp'].min()} a {datos['Timestamp'].max()}")
        print(f"  Animales unicos: {datos['Animal'].nunique()}")

    return datos
