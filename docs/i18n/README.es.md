# Semantic Signal — un agente de trading con LLM que lee el mercado como texto y guarda memoria de sus operaciones cerradas

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![Pruebas](https://img.shields.io/badge/tests-3%2C043-brightgreen)](../../tests)
[![Licencia: MIT](https://img.shields.io/badge/license-MIT-green)](../../LICENSE.md)
[![tests](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml)
[![CodeQL](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml)

**Léelo en:** [English](../../README.md) · [Polski](README.pl.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Español** · [Deutsch](README.de.md) · [Português](README.pt.md) · [Русский](README.ru.md)

![El panel en vivo: la ganancia de la cuenta desde la primera operación, la decisión actual y el coste de la ejecución](../assets/dashboard-overview.webp)

**Semantic Signal** es un agente de trading experimental. En cada vela cerrada envía indicadores de mercado, noticias y la memoria de operaciones cerradas a un modelo de texto, y después valida la respuesta con reglas de riesgo deterministas. Los gráficos son para el panel y Discord, no para la entrada del modelo. Los resultados públicos de abajo son simulados; el reenvío opcional de órdenes requiere un ejecutor aparte.

[Panel en vivo](https://semanticsignal.qrak.org) · [Registro de operaciones](https://qrak.org/results/) · [Cómo funciona un ciclo](https://qrak.org/story/) · [Documentación](../README.md)

## Resultados

*Nota: esta copia localizada está congelada. Las cifras oficiales más recientes están en el [README en inglés](../../README.md).*

**Ventana:** 2026-09-20 → 2026-11-25 (67 días), BTC/USDC, velas de 4h, 10 000 USDC de capital simulado.
**Fuente:** el propio registro del bot, congelado en [`results/2026-11-25/`](../../results/2026-11-25); `python scripts/results_report.py --data-dir results/2026-11-25` reimprime todas las cifras de abajo.

| métrica | bot | comprar y mantener |
|---|---|---|
| Rentabilidad (bruta) | **+0.71%** (+$70.78) | **-27.91%** |
| Operaciones cerradas | 23 (23 contabilizadas) + 8 excluidas (ejecución no verificada o nunca ejecutada) | — |
| Tasa de acierto | 52.2% (12G / 11P) | — |
| Factor de beneficio | 1.91 | — |
| Sharpe / Sortino (por operación, sin anualizar) | 0.31 / 0.74 | — |
| Drawdown máximo | -0.25% | — |
| Mejor / peor operación | +6.29% / -2.67% | — |
| Coste de inferencia | $7.64 (11.8M de tokens de entrada / 5.2M de salida) | — |
| Neto tras la inferencia | +$63.14 | — |
| Análisis | 224 ciclos, mediana 259 s, $0.0093 cada uno | — |

![El bot frente a comprar y mantener en la misma ventana](../assets/results-vs-btc-2026-11-25.png)

En esos 67 días el bot rindió +0.71% mientras el activo subyacente cayó 27.91%. Son
28.6 puntos porcentuales en 23 operaciones contabilizadas: no es prueba de una ventaja.
La rentabilidad no descuenta las comisiones de trading. Las estadísticas excluyen ocho pares cuya
ejecución no se pudo verificar contra el mercado real o cuya entrada nunca se ejecutó. Las cifras de
arriba proceden de esa instantánea congelada, no del registro en vivo, así que no se mueven con cada
ciclo que ejecuta el bot. Método, tabla por operación y el comando de reproducción:
`python scripts/results_report.py --data-dir results/2026-11-25`.

## Inicio rápido

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
cp config/config.ini.example config/config.ini  # plantilla versionada; cambia el proveedor si hace falta
cp keys.env.example keys.env  # o exporta la clave de un proveedor; `provider = custom` en un servidor local no necesita ninguna
python start.py               # panel en http://localhost:8000
```

Solo Python **3.13**, ~3 GB de disco, una clave de API (o cualquier servidor local compatible con OpenAI o Anthropic). Lanzadores por plataforma,
particularidades de Windows y la tabla completa de secretos: [docs/SETUP.md](../SETUP.md).

## Más información

- [Funciones e implementación](../FEATURES.md) · [Configuración](../CONFIGURATION.md) · [Panel de ajustes local](../SETTINGS_PANEL.md)
- [Contribuir](../../CONTRIBUTING.md) · [Instrucciones para agentes](../../AGENTS.md) · [Seguridad](../../SECURITY.md) · [Changelog](../../CHANGELOG.md)
```bash
python -m pytest tests -q  # 3,043 pruebas, ~65 s
```

## Licencia

[MIT](../../LICENSE.md). Software experimental, no es asesoramiento financiero; los resultados simulados no demuestran rentabilidad.
