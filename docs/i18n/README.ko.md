# Semantic Signal — 시장을 텍스트로 읽고 청산된 거래를 기억하는 LLM 트레이딩 에이전트

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![테스트](https://img.shields.io/badge/tests-3%2C043-brightgreen)](../../tests)
[![라이선스: MIT](https://img.shields.io/badge/license-MIT-green)](../../LICENSE.md)
[![tests](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/tests.yml)
[![CodeQL](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/qrak/LLM_trader/actions/workflows/codeql.yml)

**다른 언어로 보기:** [English](../../README.md) · [Polski](README.pl.md) · [中文](README.zh.md) · [日本語](README.ja.md) · **한국어** · [Español](README.es.md) · [Deutsch](README.de.md) · [Português](README.pt.md) · [Русский](README.ru.md)

![실시간 대시보드: 첫 거래 이후 계정 수익, 현재 판단, 이번 실행 비용](../assets/dashboard-overview.webp)

**Semantic Signal** 은 실험적인 트레이딩 에이전트입니다. 종가 캔들이 마감될 때마다 시장 지표, 뉴스, 청산된 거래의 기억을 텍스트 모델로 보내고, 그 답변을 결정론적 위험 규칙으로 검증합니다. 차트는 대시보드와 Discord 를 위한 것이며 모델 입력이 아닙니다. 아래 공개 결과는 모두 시뮬레이션이며, 선택적으로 켜는 주문 전달에는 별도의 실행기가 필요합니다.

[실시간 대시보드](https://semanticsignal.qrak.org) · [거래 원장](https://qrak.org/results/) · [한 사이클의 동작 방식](https://qrak.org/story/) · [문서](../README.md)

## 결과

**기간:** 2026-09-20 → 2026-11-25(67일), BTC/USDC, 4시간 캔들, 시뮬레이션 자본 10 000 USDC.
**출처:** 봇 자체 원장을 [`results/2026-11-25/`](../../results/2026-11-25) 에 동결한 것. `python scripts/results_report.py --data-dir results/2026-11-25` 로 아래 모든 수치를 다시 출력합니다.

| 지표 | 봇 | 매수 후 보유 |
|---|---|---|
| 수익률(총액 기준) | **+0.71%** (+$70.78) | **-27.91%** |
| 청산된 거래 | 23(집계 23건) + 8건 제외(체결 미검증 또는 미실행) | — |
| 승률 | 52.2%(12 승 / 11 패) | — |
| 수익 팩터 | 1.91 | — |
| 샤프 / 소르티노(거래당, 연율화하지 않음) | 0.31 / 0.74 | — |
| 최대 낙폭 | -0.25% | — |
| 최고 / 최악 거래 | +6.29% / -2.67% | — |
| 추론 비용 | $7.64(입력 11.8M / 출력 5.2M 토큰) | — |
| 추론 비용 차감 후 순액 | +$63.14 | — |
| 분석 횟수 | 224 사이클, 중앙값 259초, 회당 $0.0093 | — |

![같은 기간의 봇과 매수 후 보유 비교](../assets/results-vs-btc-2026-11-25.png)

그 67일 동안 봇은 +0.71%를 기록한 반면 기초 자산은 27.91% 하락했습니다. 이는
집계된 23건의 거래에서 28.6 퍼센트포인트이며, 우위의 증거가 아닙니다.
수익률은 거래 수수료를 차감하지 않았습니다. 통계에서는 체결을 실제 시장과 대조해 검증할 수 없었거나
진입이 실행되지 않은 8쌍을 제외했습니다. 위 수치는 실시간 원장이 아니라
그 동결 스냅샷에서 나온 것이므로, 봇이 사이클을 돌릴 때마다 바뀌지 않습니다.
방법, 거래별 표, 재현 명령:
`python scripts/results_report.py --data-dir results/2026-11-25`.

> 이 번역본은 동결된 사본입니다. 최신 공식 수치는 [English README](../../README.md) 를 참조하세요.

## 빠른 시작

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --no-compile -r requirements.txt -c constraints.txt
cp config/config.ini.example config/config.ini  # 추적되는 템플릿, 필요하면 provider 수정
cp keys.env.example keys.env  # 또는 provider 키 하나를 export; 로컬 서버라면 `provider = custom` 은 키가 필요 없음
python start.py               # 대시보드: http://localhost:8000
```

Python **3.13** 전용, 디스크 약 3 GB, API 키 하나(또는 OpenAI 나 Anthropic 과 호환되는 로컬 서버 아무거나). 플랫폼별 실행 스크립트,
Windows 주의 사항, 전체 비밀 키 표: [docs/SETUP.md](../SETUP.md).

## 더 알아보기

- [기능과 구현](../FEATURES.md) · [설정](../CONFIGURATION.md) · [로컬 설정 패널](../SETTINGS_PANEL.md)
- [기여하기](../../CONTRIBUTING.md) · [에이전트 지침](../../AGENTS.md) · [보안](../../SECURITY.md) · [변경 기록](../../CHANGELOG.md)
```bash
python -m pytest tests -q  # 3,043개 테스트, 약 65초
```

## 라이선스

[MIT](../../LICENSE.md). 실험적 소프트웨어이며 투자 조언이 아닙니다. 페이퍼 결과가 수익성을 입증하지는 않습니다.
