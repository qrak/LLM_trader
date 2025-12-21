"""
Crypto Trading Bot - Entry Point
Automated trading with AI-powered decisions.
"""
import asyncio
import sys
import argparse

from src.utils.loader import config
from src.app import CryptoTradingBot
from src.logger.logger import Logger
from src.utils.graceful_shutdown_manager import GracefulShutdownManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot - AI-powered automated trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                    # Trade default symbol from config
  python start.py ETH/USDT           # Trade ETH/USDT
  python start.py BTC/USDT -t 4h     # Trade BTC/USDT on 4h timeframe
  python start.py SOL/USDT -t 1h     # Trade SOL/USDT on 1h timeframe
        """
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Trading symbol (e.g., BTC/USDT). Default: from config"
    )
    parser.add_argument(
        "-t", "--timeframe",
        default=None,
        help="Timeframe for trading (e.g., 1h, 4h, 1d). Default: from config"
    )
    return parser.parse_args()


async def main_async():
    """Async entry point for the application"""
    args = parse_args()
    
    logger = Logger(logger_name="Bot", logger_debug=config.LOGGER_DEBUG)
    bot = CryptoTradingBot(logger)
    
    try:
        await bot.initialize()
        
        symbol = args.symbol or config.CRYPTO_PAIR
        timeframe = args.timeframe or config.TIMEFRAME
        
        logger.info(f"\n{'='*60}")
        logger.info("CRYPTO TRADING BOT")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info("Press Ctrl+C to stop")
        logger.info(f"{'='*60}\n")
        
        await bot.run(symbol, timeframe)
        
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await asyncio.wait_for(bot.shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Shutdown timed out! Forcing exit...")
            sys.exit(1)


def main() -> None:
    """Main entry point with simplified cleanup"""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shutdown_manager = GracefulShutdownManager(loop)
    shutdown_manager.setup_signal_handlers()
    try:
        loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received in main loop")
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
    finally:
        try:
            loop.run_until_complete(asyncio.wait_for(loop.shutdown_asyncgens(), timeout=5.0))
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Error shutting down async generators: {e}")
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, asyncio.Task) and not obj.done():
                obj.cancel()
        loop.close()


if __name__ == "__main__":
    main()
