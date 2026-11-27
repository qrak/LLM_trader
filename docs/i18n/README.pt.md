# Semantic Signal — um agente de trading com LLM que lê o mercado como texto e mantém memória de suas operações encerradas

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![Testes](https://img.shields.io/badge/tests-3%2C043-brightgreen)](../../tests)
[![Licença: MIT](https://img.shields.io/badge/license-MIT-green)](../../LICENSE.md)
[![tests](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml)
[![CodeQL](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml)

**Leia em:** [English](../../README.md) · [Polski](README.pl.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Deutsch](README.de.md) · **Português** · [Русский](README.ru.md)

![O painel ao vivo: o ganho da conta desde a primeira operação, a decisão atual e o custo da execução](../assets/dashboard-overview.webp)

**Semantic Signal** é um agente de trading experimental. A cada candle fechado, ele envia indicadores de mercado, notícias e a memória de operações encerradas a um modelo de texto e, em seguida, valida a resposta com regras de risco determinísticas. Os gráficos são para o painel e o Discord, não para a entrada do modelo. Os resultados públicos abaixo são simulados; o encaminhamento opcional de ordens exige um executor separado.

[Painel ao vivo](https://semanticsignal.qrak.org) · [Livro de operações](https://qrak.org/results/) · [Como um ciclo funciona](https://qrak.org/story/) · [Documentação](../README.md)

## Resultados

*Nota: esta cópia localizada está congelada. Os números oficiais mais recentes estão no [README em inglês](../../README.md).*

**Janela:** 2026-09-20 → 2026-11-25 (67 dias), BTC/USDC, candles de 4h, 10 000 USDC de capital simulado.
**Fonte:** o próprio livro do bot, congelado em [`results/2026-11-25/`](../../results/2026-11-25); `python scripts/results_report.py --data-dir results/2026-11-25` reimprime todos os números abaixo.

| métrica | bot | comprar e manter |
|---|---|---|
| Retorno (bruto) | **+0.71%** (+$70.78) | **-27.91%** |
| Operações encerradas | 23 (23 contabilizadas) + 8 excluídas (execução não verificada ou nunca executada) | — |
| Taxa de acerto | 52.2% (12G / 11P) | — |
| Fator de lucro | 1.91 | — |
| Sharpe / Sortino (por operação, não anualizado) | 0.31 / 0.74 | — |
| Drawdown máximo | -0.25% | — |
| Melhor / pior operação | +6.29% / -2.67% | — |
| Custo de inferência | $7.64 (11.8M de tokens de entrada / 5.2M de saída) | — |
| Líquido após a inferência | +$63.14 | — |
| Análises | 224 ciclos, mediana de 259 s, $0.0093 cada | — |

![O bot versus comprar e manter na mesma janela](../assets/results-vs-btc-2026-11-25.png)

Nesses 67 dias o bot rendeu +0.71% enquanto o ativo subjacente caiu 27.91%. São
28.6 pontos percentuais em 23 operações contabilizadas — não é prova de vantagem.
O retorno não desconta as taxas de corretagem. As estatísticas excluem oito pares cuja
execução não pôde ser verificada contra o mercado real ou cuja entrada nunca foi executada. Os números
acima vêm dessa captura congelada, não do livro ao vivo, portanto não mudam a cada ciclo que o bot
executa. Método, tabela por operação e o comando de reprodução:
`python scripts/results_report.py --data-dir results/2026-11-25`.

## Início rápido

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
cp config/config.ini.example config/config.ini  # modelo versionado; edite o provedor se precisar
cp keys.env.example keys.env  # ou exporte a chave de um provedor; `provider = custom` em um servidor local não precisa de nenhuma
python start.py               # painel em http://localhost:8000
```

Apenas Python **3.13**, ~3 GB de disco, uma chave de API (ou qualquer servidor local compatível com OpenAI ou Anthropic). Lançadores por plataforma,
peculiaridades do Windows e a tabela completa de segredos: [docs/SETUP.md](../SETUP.md).

## Saiba mais

- [Recursos e implementação](../FEATURES.md) · [Configuração](../CONFIGURATION.md) · [Painel de ajustes local](../SETTINGS_PANEL.md)
- [Contribuir](../../CONTRIBUTING.md) · [Instruções para agentes](../../AGENTS.md) · [Segurança](../../SECURITY.md) · [Changelog](../../CHANGELOG.md)
```bash
python -m pytest tests -q  # 3,043 testes, ~65 s
```

## Licença

[MIT](../../LICENSE.md). Software experimental, não é aconselhamento financeiro; resultados simulados não comprovam lucratividade.
