"""
Cleanup scheduler for periodic message cleanup.
Handles background tasks and scheduling for message cleanup operations.
"""
import asyncio
from typing import Set, Optional


class CleanupScheduler:
    """Manages periodic cleanup scheduling and background tasks."""
    
    def __init__(self, cleanup_interval: int, logger):
        self.cleanup_interval = cleanup_interval
        self.logger = logger
        self.cleanup_task: Optional[asyncio.Task] = None
        self.deletion_tasks: Set[asyncio.Task] = set()
        self.is_running = False
    
    def start_cleanup_task(self, bot, cleanup_callback):
        """Start the background cleanup task."""
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            
        self.cleanup_task = bot.loop.create_task(
            self._periodic_cleanup_loop(cleanup_callback),
            name="MessageCleanupTask"
        )
        self.is_running = True
        self.logger.info(f"Started background message cleanup task (interval: {self.cleanup_interval}s)")
    
    async def _periodic_cleanup_loop(self, cleanup_callback):
        """Main periodic cleanup loop."""
        try:
            await asyncio.sleep(10)  # Initial delay
            
            while self.is_running:
                try:
                    await self._run_cleanup_cycle(cleanup_callback)
                except asyncio.CancelledError:
                    self.logger.info("Message cleanup task cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Error in scheduled message cleanup: {e}")
                
                await self._sleep_with_cancellation_check()
                
        except asyncio.CancelledError:
            self.logger.info("Periodic message cleanup task cancelled")
        except Exception as e:
            self.logger.error(f"Unexpected error in periodic message cleanup task: {e}")
    
    async def _run_cleanup_cycle(self, cleanup_callback):
        """Run a single cleanup cycle."""
        deleted_count = await cleanup_callback()
        if deleted_count > 0:
            self.logger.debug(f"Cleaned up {deleted_count} expired messages")
    
    async def _sleep_with_cancellation_check(self):
        """Sleep with proper cancellation handling."""
        try:
            await asyncio.sleep(self.cleanup_interval)
        except asyncio.CancelledError:
            self.logger.info("Message cleanup sleep cancelled")
            raise
    
    async def shutdown(self):
        """Shutdown all cleanup tasks."""
        self.is_running = False
        cancelled_tasks = 0
        
        # Cancel the main cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            try:
                self.cleanup_task.cancel()
                cancelled_tasks += 1
            except Exception as e:
                self.logger.warning(f"Error cancelling cleanup task: {e}")
            self.cleanup_task = None
        
        # Cancel all deletion tasks
        deletion_tasks = list(self.deletion_tasks)
        if deletion_tasks:
            for task in deletion_tasks:
                if not task.done():
                    task.cancel()
                    cancelled_tasks += 1
            
            if cancelled_tasks > 0:
                await asyncio.gather(
                    *[self._wait_task_cancelled(task) for task in deletion_tasks], 
                    return_exceptions=True
                )
        
        if cancelled_tasks > 0:
            self.logger.info(f"Cancelled {cancelled_tasks} cleanup tasks during shutdown")
    
    async def _wait_task_cancelled(self, task: asyncio.Task):
        """Wait for a task to be cancelled gracefully."""
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.warning(f"Task raised exception during cancellation: {e}")
    
    def get_task_count(self) -> int:
        """Get the number of active deletion tasks."""
        return len(self.deletion_tasks)
