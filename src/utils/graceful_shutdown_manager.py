import asyncio
import signal
import sys
from typing import Any, Optional, Callable

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

class GracefulShutdownManager:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        logger=None,
        confirmation_callback: Optional[Callable[[], bool]] = None
    ):
        self.loop = loop
        self.logger = logger
        self.confirmation_callback = confirmation_callback
        self._callbacks = []
        self._shutting_down = False

    def setup_signal_handlers(self):
        # Only set up signal handlers on Unix systems
        # On Windows, let KeyboardInterrupt propagate naturally
        if sys.platform != 'win32':
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.loop.add_signal_handler(sig, lambda s=sig, *args: self.handle_signal(s))

    def register_shutdown_callback(self, callback):
        """Register a callback to be executed during graceful shutdown."""
        if asyncio.iscoroutinefunction(callback):
            self._callbacks.append(callback)
        else:
            # Wrap synchronous callbacks if necessary, or just append
            # For now, assume coroutines or handle execution accordingly
            self._callbacks.append(callback)

    def handle_signal(self, sig: int):
        if self.logger:
            self.logger.info(f"Signal {sig} received. Asking for confirmation...")
        else:
            print(f"Received signal {sig}, asking for confirmation...")
        
        if self.confirmation_callback:
            if self.confirmation_callback():
                if self.logger:
                    self.logger.info("User confirmed shutdown. Initiating graceful shutdown...")
                else:
                    print("User confirmed shutdown, initiating...")
                
                if self.loop.is_running() and not self.loop.is_closed():
                    self.loop.create_task(self.shutdown_gracefully())
            else:
                if self.logger:
                    self.logger.info("User cancelled shutdown. Continuing operation...")
                else:
                    print("User cancelled shutdown. Continuing operation...")
        else:
            print(f"Received signal {sig}, initiating shutdown...")
            if self.loop.is_running() and not self.loop.is_closed():
                self.loop.create_task(self.shutdown_gracefully())

    async def shutdown_gracefully(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        
        print("Performing graceful shutdown...")
        
        # Execute registered callbacks first
        if self._callbacks:
            print(f"Executing {len(self._callbacks)} shutdown callbacks...")
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    print(f"Error in shutdown callback {callback}: {e}")

        pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if pending_tasks:
            print(f"Cancelling {len(pending_tasks)} tasks...")
            for task in pending_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.wait(pending_tasks), timeout=10.0)
            except asyncio.TimeoutError:
                task_details = []
                for t in pending_tasks:
                    if not t.done():
                        if hasattr(t, 'get_name') and callable(t.get_name):
                            name = t.get_name()
                        elif hasattr(t, 'name'):
                            name = t.name
                        else:
                            name = str(t)
                        task_details.append(name)
                print(f"Some tasks didn't complete in time: {task_details}")
        try:
            await asyncio.wait_for(self.loop.shutdown_asyncgens(), timeout=2.0)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Error shutting down async generators: {e}")
    async def cleanup_bot_resources(self, bot: Any):
        """Clean up bot resources safely."""
        logger = bot.logger
        
        # Cancel and wait for active tasks
        pending_tasks = list(bot._active_tasks)
        if pending_tasks:
            logger.info(f"Cancelling {len(pending_tasks)} active tasks...")
            for task in pending_tasks:
                if not task.done():
                    task.cancel()

            try:
                await asyncio.wait(pending_tasks, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Some tasks didn't complete in time")

        # Close keyboard handler
        if bot.keyboard_handler:
            try:
                logger.info("Closing keyboard handler...")
                await bot.keyboard_handler.stop_listening()
                bot.keyboard_handler = None
            except Exception as e:
                logger.warning(f"Error closing keyboard handler: {e}")

        # Close Discord notifier
        if bot._discord_task and not bot._discord_task.done():
            bot._discord_task.cancel()
            try:
                await asyncio.wait_for(bot._discord_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        if bot.discord_notifier:
            try:
                logger.info("Closing Discord notifier...")
                async with bot.discord_notifier:
                    pass  # __aexit__ handles cleanup
                bot.discord_notifier = None
            except Exception as e:
                logger.warning(f"Error closing Discord notifier: {e}")

        # Close components in reverse initialization order
        
        # Market Analyzer & Model Manager
        if hasattr(bot, 'model_manager') and bot.model_manager:
            try:
                logger.info("Closing ModelManager...")
                await asyncio.wait_for(bot.model_manager.close(), timeout=3.0)
                bot.model_manager = None
            except asyncio.TimeoutError:
                logger.warning("ModelManager shutdown timed out")
            except Exception as e:
                logger.warning(f"Error closing ModelManager: {e}")
        
        if bot.market_analyzer:
            try:
                logger.info("Closing market analyzer...")
                await asyncio.wait_for(bot.market_analyzer.close(), timeout=3.0)
                bot.market_analyzer = None
            except asyncio.TimeoutError:
                logger.warning("Market analyzer shutdown timed out")
            except Exception as e:
                logger.warning(f"Error closing market analyzer: {e}")

        # RAG Engine
        if hasattr(bot, 'rag_engine') and bot.rag_engine:
            try:
                logger.info("Closing RAG engine...")
                await asyncio.wait_for(bot.rag_engine.close(), timeout=3.0)
                bot.rag_engine = None
            except asyncio.TimeoutError:
                logger.warning("RAG engine shutdown timed out")
            except Exception as e:
                logger.warning(f"Error closing RAG engine: {e}")

        # API Clients
        api_clients = [
            ("AlternativeMeAPI", bot.alternative_me_api),
            ("CoinGeckoAPI", bot.coingecko_api)
        ]

        for client_name, client in api_clients:
            if client and hasattr(client, 'close'):
                try:
                    logger.info(f"Closing {client_name}...")
                    await asyncio.wait_for(client.close(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning(f"{client_name} shutdown timed out")
                except Exception as e:
                    logger.warning(f"Error closing {client_name}: {e}")
        
        # Close shared CryptoCompare session
        if bot.cryptocompare_session:
            try:
                logger.info("Closing CryptoCompare session...")
                await bot.cryptocompare_session.close()
                bot.cryptocompare_session = None
            except Exception as e:
                logger.warning(f"Error closing CryptoCompare session: {e}")

        # Exchange Manager
        if bot.exchange_manager:
            try:
                logger.info("Closing ExchangeManager...")
                await asyncio.wait_for(bot.exchange_manager.shutdown(), timeout=3.0)
                bot.exchange_manager = None
            except asyncio.TimeoutError:
                logger.warning("ExchangeManager shutdown timed out")
            except Exception as e:
                logger.warning(f"Error closing ExchangeManager: {e}")
        
        # Cleanup references
        bot.alternative_me_api = None
        bot.news_api = None
        bot.market_api = None
        bot.categories_api = None
        bot.cryptocompare_session = None
        bot.coingecko_api = None
        bot.discord_notifier = None
        bot.keyboard_handler = None
    
    @staticmethod
    def show_exit_confirmation() -> bool:
        """
        Show a confirmation dialog before closing the application.
        
        Returns:
            True if user confirmed exit, False if they cancelled.
        """
        if not PYQT_AVAILABLE:
            try:
                response = input("\nAre you sure you want to exit? (y/n): ").strip().lower()
                return response in ['y', 'yes']
            except (EOFError, KeyboardInterrupt):
                return True
        
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                QApplication.setHighDpiScaleFactorRoundingPolicy(
                    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
                )
            
            result = QMessageBox.question(
                None,
                "Exit Confirmation",
                "Are you sure you want to close the Crypto Trading Bot application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            return result == QMessageBox.StandardButton.Yes
        except Exception as e:
            print(f"Warning: Could not show confirmation dialog: {e}. Proceeding with shutdown.")
            return True
