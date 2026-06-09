import datetime
import time
import pandas as pd
import sys

LOTTERIES = [
    {
        'name': 'Lotto Activo',
        'file': 'LottoActivoINT.xlsx',
        'module': 'scraper_lotto',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'La Granjita',
        'file': 'LaGranjita.xlsx',
        'module': 'scraper_la_granjita',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'Selva Plus',
        'file': 'SelvaPlus.xlsx',
        'module': 'scraper_selva_plus',
        'scrape_date': None,
        'save_to_excel': None,
    },
    {
        'name': 'Lotto Activo Rd Int',
        'file': 'LottoActivoRDInt.xlsx',
        'module': 'scraper_lotto_rd_int',
        'scrape_date': None,
        'save_to_excel': None,
    },
]


def import_scrapers():
    for lot in LOTTERIES:
        mod = __import__(lot['module'])
        lot['scrape_date'] = mod.scrape_date
        lot['save_to_excel'] = mod.save_to_excel


def find_missing_dates(excel_file, start_date, end_date):
    try:
        existing = pd.read_excel(excel_file)
        existing['Fecha'] = pd.to_datetime(existing['Fecha']).dt.strftime("%Y-%m-%d")
        existing_dates = set(existing['Fecha'].unique())
    except (FileNotFoundError, Exception):
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

    missing = find_missing_dates(lot['file'], start_date, end_date)
    print(f"  Fechas faltantes: {len(missing)}")

    if not missing:
        print("  No hay fechas faltantes.")
        return 0

    total = 0
    for i, date_str in enumerate(missing):
        records = lot['scrape_date'](date_str)
        if records:
            df = pd.DataFrame(records)
            lot['save_to_excel'](df, lot['file'])
            total += len(records)
        if (i + 1) % 10 == 0:
            print(f"  Progreso: {i+1}/{len(missing)} dias")
        time.sleep(1.5)

    print(f"  Total registros agregados: {total}")
    return total


def main():
    import_scrapers()

    print("SCRAPE ALL - ACTUALIZAR TODAS LAS LOTERIAS")
    print()

    hoy = datetime.date.today()
    default_start = "2024-01-01"

    start = input(f"Fecha inicio (default: {default_start}): ").strip() or default_start
    end = input(f"Fecha fin (default: {hoy}): ").strip() or hoy.strftime("%Y-%m-%d")

    total_records = 0
    for lot in LOTTERIES:
        total_records += scrape_lottery(lot, start, end)
        time.sleep(2)

    print(f"\nResumen: {total_records} registros agregados en total")


if __name__ == "__main__":
    main()
