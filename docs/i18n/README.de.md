# Semantic Signal — ein LLM-Handelsagent, der den Markt anhand von Text analysiert und sich abgeschlossene Trades merkt

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-3%2C043-brightgreen)](../../tests)
[![Lizenz: MIT](https://img.shields.io/badge/license-MIT-green)](../../LICENSE.md)
[![Tests](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml)
[![CodeQL](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml)

**Sprachversionen:** [English](../../README.md) · [Polski](README.pl.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Deutsch** · [Português](README.pt.md) · [Русский](README.ru.md)

![Live-Dashboard: Kontogewinn seit dem ersten Trade, aktuelle Entscheidung und Betriebskosten](../assets/dashboard-overview.webp)

**Semantic Signal** ist ein experimenteller Handelsagent. Nach Abschluss jeder Kerze übermittelt er Marktindikatoren, Nachrichten und Informationen aus seinem Gedächtnis für abgeschlossene Trades an ein Textmodell. Anschließend prüft er die Antwort anhand deterministischer Risikoregeln. Die Charts sind für das Dashboard und Discord bestimmt, nicht für das Modell. Die folgenden öffentlichen Ergebnisse stammen aus einer Simulation. Wer Aufträge weiterleiten möchte, benötigt dafür einen separaten Ausführungsdienst.

[Live-Dashboard](https://semanticsignal.qrak.org) · [Handelsjournal](https://qrak.org/results/) · [Ablauf eines Zyklus](https://qrak.org/story/) · [Dokumentation](../README.md)

## Ergebnisse

Die folgende Auswertung basiert auf dem gespeicherten Stand vom 25. November 2026. Die neueste veröffentlichte Auswertung steht in der [englischen README](../../README.md).

**Zeitraum:** 2026-09-20 → 2026-11-25 (67 Tage), BTC/USDC, 4-Stunden-Kerzen, 10.000 USDC simuliertes Kapital.

**Quelle:** das Handelsjournal des Bots, festgehalten in [`results/2026-11-25/`](../../results/2026-11-25). `python scripts/results_report.py --data-dir results/2026-11-25` gibt alle folgenden Zahlen erneut aus.

| Kennzahl | Bot | Kaufen und halten |
|---|---|---|
| Rendite (brutto) | **+0,71 %** (+70,78 USD) | **−27,91 %** |
| Abgeschlossene Trades | 23 berücksichtigt + 8 ausgeschlossen (Ausführung nicht verifiziert oder nicht erfolgt) | — |
| Gewinnquote | 52,2 % (12 Gewinne / 11 Verluste) | — |
| Profitfaktor | 1,91 | — |
| Sharpe / Sortino (pro Trade, nicht annualisiert) | 0,31 / 0,74 | — |
| Maximaler Drawdown | −0,25 % | — |
| Bester / schlechtester Trade | +6,29 % / −2,67 % | — |
| Inferenzkosten | 7,64 USD (11,8 Mio. Eingabe- / 5,2 Mio. Ausgabetokens) | — |
| Ergebnis nach Inferenzkosten | +63,14 USD | — |
| Analysen | 224 Zyklen, Median 259 s, je 0,0093 USD | — |

![Bot im Vergleich zu Kaufen und Halten im selben Zeitraum](../assets/results-vs-btc-2026-11-25.png)

In diesen 67 Tagen erzielte der Bot +0,71 %, während der Basiswert um 27,91 % fiel. Das sind 28,6 Prozentpunkte bei 23 berücksichtigten Trades – kein Beleg für einen Handelsvorteil. Handelsgebühren wurden von der Rendite nicht abgezogen. Die Statistik schließt acht Paare aus Einstiegs- und Ausstiegsorder aus, bei denen sich die Ausführung anhand des tatsächlichen Markts nicht verifizieren ließ oder die Einstiegsorder nie ausgeführt wurde. Die Zahlen stammen aus dem gespeicherten Stand, nicht aus dem laufenden Handelsjournal, und ändern sich daher nicht mit jedem Zyklus.

Methode, Tabelle der einzelnen Trades und Anleitung zur Reproduktion: `python scripts/results_report.py --data-dir results/2026-11-25`.

## Schnellstart

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
cp config/config.ini.example config/config.ini  # Vorlage aus dem Repository; Anbieter bei Bedarf ändern
cp keys.env.example keys.env  # oder den API-Schlüssel als Umgebungsvariable setzen; ein lokaler Server mit `provider = custom` braucht keinen Schlüssel
python start.py               # Dashboard unter http://localhost:8000
```

Erforderlich sind Python **3.13**, ca. 3 GB Speicherplatz und ein API-Schlüssel (oder ein lokaler Server mit OpenAI- oder Anthropic-kompatibler API). Startskripte, Hinweise zu Windows und die vollständige Tabelle der Zugangsdaten stehen in der [Einrichtungsanleitung](../SETUP.md).

## Weitere Informationen

- [Funktionen und Implementierung](../FEATURES.md) · [Konfiguration](../CONFIGURATION.md) · [Lokales Einstellungsmenü](../SETTINGS_PANEL.md)
- [Mitwirken](../../CONTRIBUTING.md) · [Anweisungen für Agenten](../../AGENTS.md) · [Sicherheit](../../SECURITY.md) · [Änderungsprotokoll](../../CHANGELOG.md)

```bash
python -m pytest tests -q  # 3.043 Tests, ca. 65 s
```

## Lizenz

[MIT](../../LICENSE.md). Experimentelle Software, keine Finanzberatung. Simulationsergebnisse sind kein Nachweis für Rentabilität.
