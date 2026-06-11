import types
from pathlib import Path

from scraper_selva_plus import scrape_date as selva_scrape
from scraper_lotto_rd_int import scrape_date as rd_scrape
from scraper_la_granjita import scrape_date as granjita_scrape


class DummyResp:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        return None


class DummySession:
    def __init__(self, text):
        self._text = text
        self.headers = {}

    def post(self, *args, **kwargs):
        return DummyResp(self._text)

    def get(self, *args, **kwargs):
        return DummyResp(self._text)


def load_fixture(name):
    p = Path(__file__).parent / 'fixtures' / name
    return p.read_text(encoding='utf-8')


def test_selva_plus_fixture():
    html = load_fixture('selva_plus_sample.html')
    session = DummySession(html)
    recs = selva_scrape('2023-01-01', session=session)
    assert len(recs) == 1
    assert recs[0]['Numero'] == 12


def test_lotto_rd_int_fixture():
    html = load_fixture('lotto_rd_int_sample.html')
    session = DummySession(html)
    recs = rd_scrape('2023-01-01', session=session)
    assert len(recs) == 1
    assert recs[0]['Numero'] == 34


def test_la_granjita_fixture():
    html = load_fixture('la_granjita_sample.html')
    session = DummySession(html)
    recs = granjita_scrape('2023-01-01', session=session)
    assert len(recs) == 1
    assert recs[0]['Numero'] == 7
