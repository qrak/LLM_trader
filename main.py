import asyncio
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Any

if sys.version_info < (3, 13):
    print("Error: TransformerBot requires Python 3.13 or higher")
    sys.exit(1)

from ccxt import NotSupported

from core.trading_strategy import TradingStrategy
from logger.logger import Logger
from utils.retry_decorator import retry_async

async def shutdown(loop, analyzer: TradingStrategy, logger: Logger) -> None:
    logger.info("Shutting down gracefully...")
    await analyzer.close()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

@retry_async()
async def _wait_for_next_timeframe_step(analyzer, delay: Optional[int] = None, add_delay: int = 0) -> None:
    try:
        current_time_ms = await analyzer.exchange.fetch_time()
    except NotSupported:
        analyzer.logger.debug(f"{analyzer.exchange.id} does not support fetch_time(). Using local time instead.")
        current_time_ms = int(time.time() * 1000)
    except Exception as e:
        analyzer.logger.exception(f"Error fetching time from {analyzer.exchange.id}, using local time: {str(e)}")
        current_time_ms = int(time.time() * 1000)

    interval_ms = analyzer.interval * 1000
    next_timeframe_start_ms = (current_time_ms // interval_ms + 1) * interval_ms

    if delay is None:
        delay_ms = next_timeframe_start_ms - current_time_ms + add_delay * 1000
        
        delay_seconds = delay_ms / 1000
        next_check_time = datetime.fromtimestamp(next_timeframe_start_ms / 1000)
        
        wait_time = str(timedelta(seconds=int(delay_seconds)))
        analyzer.logger.info(f"Next check in {wait_time} at {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(delay_seconds)
        return
    else:
        delay_ms = delay * 1000
        wait_time = str(timedelta(seconds=delay))
        analyzer.logger.info(f"Using fixed delay of {wait_time}")

    await asyncio.sleep(delay_ms / 1000)

async def periodic_check(analyzer) -> None:
    check_count = 0

    while True:
        try:
            await _wait_for_next_timeframe_step(analyzer)
            current_time = datetime.now()
            check_count += 1

            analyzer.logger.info("=" * 50)
            analyzer.logger.info(f"Periodic Check #{check_count} at {current_time}")

            market_data = await analyzer.fetch_ohlcv()
            current_price = analyzer.periods['3D'].data[-1].close
            if analyzer.current_position:
                await analyzer.check_position(current_price)

            analyzer.logger.info("Performing market analysis...")
            analysis = await analyzer.analyze_trend(market_data)
            await analyzer.process_analysis(analysis)
        except Exception as e:
            analyzer.logger.exception(f"Error during periodic check: {e}")
            await asyncio.sleep(60)

async def run(analyzer) -> None:
    tasks: list[asyncio.Task[Any]] = []
    try:
        analyzer.logger.info(f"Starting {analyzer.symbol} analyzer...")
        analyzer.current_position = analyzer.data_persistence.load_position()
        check_task = asyncio.create_task(periodic_check(analyzer))
        tasks.append(check_task)

        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        analyzer.logger.info("Analyzer received cancellation request...")
        for task in tasks:
            if not task.done():
                task.cancel("Application shutdown requested")
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        analyzer.logger.error(f"Error in analyzer: {e}")
        for task in tasks:
            if not task.done():
                task.cancel(f"Error occurred: {str(e)}")
        await asyncio.gather(*tasks, return_exceptions=True)

def main() -> None:
    logger = Logger(logger_name="Bot", logger_debug=False)
    analyzer = TradingStrategy(logger=logger)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run(analyzer))
    except KeyboardInterrupt:
        loop.run_until_complete(shutdown(loop, analyzer, logger))
    finally:
        if loop.is_running():
            loop.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()