import sys
import os
import logging
from logging.handlers import RotatingFileHandler

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

ANIMALES_38 = [
    "DELFIN", "BALLENA", "CARNERO", "TORO", "CIEMPIES", "ALACRAN",
    "LEON", "RANA", "PERICO", "RATON", "AGUILA", "TIGRE", "GATO",
    "CABALLO", "MONO", "PALOMA", "ZORRO", "OSO", "PAVO", "BURRO",
    "CHIVO", "COCHINO", "GALLO", "CAMELLO", "CEBRA", "IGUANA",
    "GALLINA", "VACA", "PERRO", "ZAMURO", "ELEFANTE", "CAIMAN",
    "LAPA", "ARDILLA", "PESCADO", "VENADO", "JIRAFA", "CULEBRA"
]

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
