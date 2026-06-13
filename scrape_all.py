import datetime
import time
import pandas as pd
import sys

LOTTERIES = [
    {
        'name': 'Lotto Activo',
        'file': 'data/LottoActivoINT.xlsx',
        'module': 'scraper_lotto',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'La Granjita',
        'file': 'data/LaGranjita.xlsx',
        'module': 'scraper_la_granjita',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'Selva Plus',
        'file': 'data/SelvaPlus.xlsx',
        'module': 'scraper_selva_plus',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'Lotto Activo Rd Int',
        'file': 'data/LottoActivoRDInt.xlsx',
        'module': 'scraper_lotto_rd_int',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'Lotto Activo RD',
        'file': 'data/LottoActivoRD.xlsx',
        'module': 'scraper_lotto_activo_rd',
        'scrape_date': None,
        'save_to_excel': None,
    },
]


def import_scrapers():
    for lot in LOTTERIES:
        try:
            mod = __import__(lot['module'])
            lot['scrape_date'] = mod.scrape_date
            lot['save_to_excel'] = mod.save_to_excel
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Warning: Could not load {lot['name']}: {e}")
            lot['scrape_date'] = None
            lot['save_to_excel'] = None


def find_missing_dates(excel_file, start_date, end_date):
    try:
        existing = pd.read_excel(excel_file)
        if existing.empty:
            existing_dates = set()
        else:
            existing['Fecha'] = pd.to_datetime(existing['Fecha']).dt.strftime("%Y-%m-%d")
            existing_dates = set(existing['Fecha'].unique())
    except FileNotFoundError:
        existing_dates = set()
    except Exception as e:
        print(f"Error leyendo {excel_file}: {e}")
        existing_dates = set()

    all_dates = set()
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end_dt:
        all_dates.add(current.strftime("%Y-%m-%d"))
        current += datetime.timedelta(days=1)

    missing = sorted(all_dates - existing_dates)
    return missing


def scrape_lottery(lot, start_date, end_date):
    print(f"\n{'='*60}")
    print(f"  SCRAPING: {lot['name']}")
    print(f"{'='*60}")

    if lot['scrape_date'] is None:
        print(f"  Scraper no disponible para {lot['name']}, saltando.")
        return 0

    if start_date == end_date:
        # Single day: always scrape fresh
        print(f"  Scrapeando {start_date}...")
        records = lot['scrape_date'](start_date)
        if records:
            df = pd.DataFrame(records)
            lot['save_to_excel'](df, lot['file'])
            print(f"  Registros encontrados: {len(records)}")
        else:
            print(f"  No se encontraron datos para {start_date}.")
        return len(records) if records else 0

    # Range: only missing dates
    missing = find_missing_dates(lot['file'], start_date, end_date)
    print(f"  Fechas faltantes: {len(missing)}")

    if not missing:
        print("  No hay fechas faltantes.")
        return 0

    total = 0
    all_new_records = []
    for i, date_str in enumerate(missing):
        records = lot['scrape_date'](date_str)
        if records:
            all_new_records.extend(records)
            total += len(records)
        if (i + 1) % 10 == 0:
            print(f"  Progreso: {i+1}/{len(missing)} dias")
        time.sleep(1.5)

    if all_new_records:
        df = pd.DataFrame(all_new_records)
        lot['save_to_excel'](df, lot['file'])

    print(f"  Total registros agregados: {total}")
    return total


def main():
    import_scrapers()

    print("SCRAPE ALL - ACTUALIZAR TODAS LAS LOTERIAS")
    print()

    hoy = datetime.date.today()
    default_start = f"{hoy.year}-01-01"

    start = input(f"Fecha inicio (default: {default_start}): ").strip() or default_start
    end = input(f"Fecha fin (default: {hoy}): ").strip() or hoy.strftime("%Y-%m-%d")

    total_records = 0
    for lot in LOTTERIES:
        total_records += scrape_lottery(lot, start, end)
        time.sleep(2)

    print(f"\nResumen: {total_records} registros agregados en total")


if __name__ == "__main__":
    main()
