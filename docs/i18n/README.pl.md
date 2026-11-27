# Semantic Signal — agent tradingowy oparty na LLM, który analizuje rynek w formie tekstowej i pamięta zamknięte transakcje

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![Testy](https://img.shields.io/badge/tests-3%2C043-brightgreen)](../../tests)
[![Licencja: MIT](https://img.shields.io/badge/license-MIT-green)](../../LICENSE.md)
[![testy](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml)
[![CodeQL](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml)

**Wersje językowe:** [English](../../README.md) · **Polski** · [中文](README.zh.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Deutsch](README.de.md) · [Português](README.pt.md) · [Русский](README.ru.md)

![Panel na żywo: wynik rachunku od pierwszej transakcji, bieżąca decyzja i koszty działania](../assets/dashboard-overview.webp)

**Semantic Signal** to eksperymentalny agent tradingowy. Po zamknięciu każdej świecy wysyła do modelu tekstowego wskaźniki rynkowe, wiadomości i dane z pamięci zamkniętych transakcji. Następnie sprawdza odpowiedź według deterministycznych reguł ryzyka. Wykresy trafiają do panelu i na Discorda, a nie do modelu. Poniższe wyniki pochodzą z symulacji. Opcjonalne przekazywanie zleceń wymaga osobnej usługi wykonawczej.

[Panel na żywo](https://semanticsignal.qrak.org) · [Rejestr transakcji](https://qrak.org/results/) · [Jak działa cykl](https://qrak.org/story/) · [Dokumentacja](../README.md)

## Wyniki

Poniżej migawka z dnia 2026-11-25. Najnowsze opublikowane zestawienie jest w [angielskiej wersji README](../../README.md).

**Okres:** 2026-09-20 → 2026-11-25 (67 dni), BTC/USDC, świece 4h, 10 000 USDC kapitału symulowanego.

**Źródło:** rejestr transakcji bota, zapisany w [`results/2026-11-25/`](../../results/2026-11-25). Polecenie `python scripts/results_report.py --data-dir results/2026-11-25` odtwarza wszystkie poniższe dane.

| Miara | Bot | Kup i trzymaj |
|---|---|---|
| Stopa zwrotu (brutto) | **+0.71%** (+$70.78) | **-27.91%** |
| Zamknięte transakcje | 23 uwzględnione + 8 wykluczonych (realizacji nie zweryfikowano lub do niej nie doszło) | — |
| Odsetek zyskownych transakcji | 52.2% (12 zyskownych / 11 stratnych) | — |
| Współczynnik zysku (profit factor) | 1.91 | — |
| Sharpe / Sortino (na transakcję, bez przeliczenia na rok) | 0.31 / 0.74 | — |
| Maksymalne obsunięcie kapitału | -0.25% | — |
| Najlepsza / najgorsza transakcja | +6.29% / -2.67% | — |
| Koszt zapytań do modelu | $7.64 (11.8 mln tokenów wejściowych / 5.2 mln wyjściowych) | — |
| Wynik po odjęciu kosztu zapytań do modelu | +$63.14 | — |
| Analizy | 224 cykle, mediana 259 s, $0.0093 za cykl | — |

![Wynik bota w porównaniu ze strategią „kup i trzymaj” w tym samym okresie](../assets/results-vs-btc-2026-11-25.png)

W ciągu tych 67 dni bot osiągnął +0.71%, a instrument bazowy spadł o 27.91%. Różnica 28.6 pkt proc. przy 23 uwzględnionych transakcjach nie dowodzi przewagi strategii. Wynik nie uwzględnia prowizji transakcyjnych. W statystykach pominięto osiem par otwarcie–zamknięcie, dla których nie dało się potwierdzić realizacji na podstawie faktycznych danych rynkowych albo do otwarcia nigdy nie doszło. Dane pochodzą z zapisanej migawki, nie z bieżącego rejestru, więc nie zmieniają się po każdym cyklu.

Metodę, tabelę poszczególnych transakcji i instrukcję odtworzenia wyników wyświetla `python scripts/results_report.py --data-dir results/2026-11-25`.

## Szybki start

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
cp config/config.ini.example config/config.ini  # szablon z repozytorium; w razie potrzeby zmień dostawcę
cp keys.env.example keys.env  # albo ustaw klucz dostawcy w środowisku; lokalny serwer z `provider = custom` nie wymaga klucza
python start.py               # panel pod http://localhost:8000
```

Wymagany jest Python **3.13**, ok. 3 GB miejsca na dysku i jeden klucz API (lub lokalny serwer zgodny z API OpenAI albo Anthropic). Skrypty uruchomieniowe, wskazówki dla systemu Windows i pełna tabela sekretów: [instrukcja instalacji](../SETUP.md).

## Więcej informacji

- [Funkcje i implementacja](../FEATURES.md) · [Konfiguracja](../CONFIGURATION.md) · [Lokalny panel ustawień](../SETTINGS_PANEL.md)
- [Współtworzenie projektu](../../CONTRIBUTING.md) · [Instrukcje dla agentów](../../AGENTS.md) · [Bezpieczeństwo](../../SECURITY.md) · [Historia zmian](../../CHANGELOG.md)

```bash
python -m pytest tests -q  # 3043 testy, ok. 65 s
```

## Licencja

[MIT](../../LICENSE.md). To eksperymentalne oprogramowanie, nie porada finansowa. Wyniki symulacji nie dowodzą rentowności.
