import sys
import time
import datetime
import logging
from logging.handlers import RotatingFileHandler
from utils import setup_logging
from utils import get_requests_session, HORA_MAP_12_TO_24
import requests as _requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL_BASE = "https://www.loteriadehoy.com/animalito/lottoactivordint/resultados/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

logger = setup_logging('lotto_rd_int_scraper')

H5_PREFIXES = ["Lotto Activo Rd Int ", "Lotto Activo Int ", "Lotto Activo Int ( Lotto Internacional ) ", "Lotto Activo Int ("]

def scrape_date(date_str, session=None, timeout=30):
    records = []
    try:
        session = session or get_requests_session()
        try:
            session.headers.update(HEADERS)
        except Exception:
            pass
        resp = session.post(URL_BASE, data={"fecha": date_str}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error fetching {date_str}: {e}")
        return records

    soup = BeautifulSoup(resp.text, 'lxml')
    items = soup.select('div.row.text-center.js-con > div')
    if not items:
        logger.warning(f"No results found for {date_str}")
        return records

    for item in items:
        h4 = item.select_one('h4')
        h5 = item.select_one('h5')
        if not h4 or not h5:
            continue
        text = h4.get_text(strip=True)
        time_text = h5.get_text(strip=True)
        for prefix in H5_PREFIXES:
            if time_text.startswith(prefix):
                time_text = time_text[len(prefix):]
                break
        parts = text.split(' ', 1)
        if len(parts) != 2:
            continue
        num_str, animal = parts
        try:
            numero = int(num_str)
        except ValueError:
            logger.warning(f"Could not parse number '{num_str}' on {date_str}")
            continue

        # Use shared mapping from utils
        hour_24 = HORA_MAP_12_TO_24.get(time_text)
        if not hour_24:
            logger.warning(f"Unknown time format '{time_text}' on {date_str}")
            continue

        records.append({
            "Fecha": date_str,
            "Hora": hour_24,
            "Animal": animal.upper(),
            "Numero": numero,
        })

    logger.info(f"{date_str}: {len(records)} records scraped")
    return records


def scrape_range(start_date, end_date, delay=1.5):
    all_records = []
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - current).days + 1
    day_count = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        records = scrape_date(date_str)
        all_records.extend(records)
        day_count += 1
        if day_count % 10 == 0:
            logger.info(f"Progress: {day_count}/{total_days} days")
        time.sleep(delay)
        current += datetime.timedelta(days=1)

    df = pd.DataFrame(all_records)
    logger.info(f"Total: {len(df)} records from {start_date} to {end_date}")
    return df


def save_to_excel(df, filename="data/LottoActivoRDInt.xlsx"):
    existing = None
    if os.path.exists(filename):
        try:
            existing = pd.read_excel(filename)
            logger.info(f"Existing file: {len(existing)} records")
        except Exception as e:
            logger.warning(f"Could not read existing file: {e}")

    if existing is not None and not existing.empty:
        existing['Fecha'] = pd.to_datetime(existing['Fecha']).dt.strftime("%Y-%m-%d")
        existing['Hora'] = existing['Hora'].astype(str).str.strip().str.zfill(8)
        df['Fecha'] = pd.to_datetime(df['Fecha']).dt.strftime("%Y-%m-%d")
        df['Hora'] = df['Hora'].astype(str).str.strip().str.zfill(8)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Fecha", "Hora"], keep="last")
        combined = combined.sort_values(["Fecha", "Hora"]).reset_index(drop=True)
    else:
        combined = df.copy()
        combined = combined.sort_values(["Fecha", "Hora"]).reset_index(drop=True)

    combined.to_excel(filename, index=False)
    logger.info(f"Saved {len(combined)} records to {filename}")
    return combined


if __name__ == "__main__":
    print("=== LOTTO ACTIVO RD INT WEB SCRAPER ===")
    print("1. Scrape single date")
    print("2. Scrape date range")
    print("3. Find missing dates and scrape")
    option = input("Select option: ").strip()

    if option == "1":
        date = input("Date (YYYY-MM-DD): ").strip()
        records = scrape_date(date)
        df = pd.DataFrame(records)
        if not df.empty:
            print(df.to_string(index=False))
            save = input("Save to Excel? (y/n): ").strip().lower()
            if save == 'y':
                save_to_excel(df)
        else:
            print("No records found.")

    elif option == "2":
        start = input("Start date (YYYY-MM-DD): ").strip()
        end = input("End date (YYYY-MM-DD): ").strip()
        df = scrape_range(start, end)
        if not df.empty:
            save_to_excel(df)
        else:
            print("No records found.")

    elif option == "3":
        filename = input("Excel filename (default: data/LottoActivoRDInt.xlsx): ").strip() or "data/LottoActivoRDInt.xlsx"
        if os.path.exists(filename):
            existing = pd.read_excel(filename)
            existing['Fecha'] = pd.to_datetime(existing['Fecha']).dt.strftime("%Y-%m-%d")
            existing_dates = set(existing['Fecha'].unique())
        else:
            existing_dates = set()

        all_dates = set()
        start = input("Start date (YYYY-MM-DD): ").strip()
        end = input("End date (YYYY-MM-DD): ").strip()
        current = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
        while current <= end_dt:
            all_dates.add(current.strftime("%Y-%m-%d"))
            current += datetime.timedelta(days=1)

        missing = sorted(all_dates - existing_dates)
        print(f"Missing {len(missing)} dates out of {len(all_dates)}")
        if not missing:
            print("No missing dates!")
            sys.exit(0)

        show = input("Show missing dates? (y/n): ").strip().lower()
        if show == 'y':
            for d in missing:
                print(f"  {d}")

        all_records = []
        for i, date_str in enumerate(missing):
            records = scrape_date(date_str)
            all_records.extend(records)
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(missing)} days")
            time.sleep(1.5)

        if all_records:
            df = pd.DataFrame(all_records)
            save_to_excel(df, filename)
        else:
            print("No new records.")
