import asyncio
import sys

from src.utils.loader import config
from src.app import DiscordCryptoBot
from src.logger.logger import Logger
from src.utils.graceful_shutdown_manager import GracefulShutdownManager


async def main_async():
    """Async entry point for the application"""
    logger = Logger(logger_name="Bot", logger_debug=config.LOGGER_DEBUG)
    bot = DiscordCryptoBot(logger)
    try:
        await bot.initialize()
        await bot.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
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
        print("Event loop closed")
        sys.exit(0)

if __name__ == "__main__":
    main()