export const site = {
    name: "Semantic Signal LLM",
    dashboardUrl: "https://semanticsignal.qrak.org/",
    repoUrl: "https://github.com/qrak/LLM_trader",
    headline: "Autonomous, Memory-Augmented LLM Trading Framework.",
    subheadline:
        "A high-performance Python-based quantitative engine synthesizing statistical pattern recognition with human-grade semantic evaluation."
};

export const snippets = {
    analyzer: `class AnalysisEngine:\n    async def run_cycle(self, symbol: str, timeframe: str) -> dict:\n        context = await self.market_data_collector.collect(symbol, timeframe)\n        technical = self.technical_calculator.calculate(context.ohlcv_candles)\n        patterns = self.pattern_analyzer.analyze(context.ohlcv_candles, technical)\n        prompt = self.prompt_builder.build(context, technical, patterns)\n        return await self.model_manager.generate(prompt)`,
    platforms: `async def _provision_platforms(self, infra: dict, utils: dict) -> dict:\n    coingecko = CoinGeckoAPI(logger=self.logger, ...)\n    await coingecko.initialize()\n    alternative_me = AlternativeMeAPI(logger=self.logger)\n    await alternative_me.initialize()\n    return {\n        \"coingecko\": coingecko,\n        \"market\": CCXTMarketAPI(logger=self.logger, exchange_manager=infra[\"exchange_manager\"]),\n        \"defillama\": DefiLlamaClient(logger=self.logger, session=infra[\"session\"], cache_dir=\"cache\"),\n        \"alternative_me\": alternative_me\n    }`,
    trading: `class TradingBrainService:\n    def __init__(self, logger, persistence, vector_memory, ...):\n        self.exit_profiles = ExitProfileResolver(self._default_exit_execution_context)\n        self.pattern_analyzer = TradePatternAnalyzer(self.exit_profiles)\n        self.reflection_engine = BrainReflectionEngine(...)\n        self.experience_recorder = BrainExperienceRecorder(...)\n        self.context_provider = BrainContextProvider(...)\n\n    def update_from_closed_trade(self, position, close_price, close_reason, ...):\n        self.experience_recorder.record_closed_trade(...)\n        if self._trade_count % self._reflection_interval == 0:\n            self._trigger_reflection()`
};