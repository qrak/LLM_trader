"""
Simplified Discord File Handler using specialized components.
Orchestrates message tracking, cleanup scheduling, and deletion operations.
"""
import asyncio
from typing import Optional, TYPE_CHECKING

from .filehandler_components.tracking_persistence import TrackingPersistence
from .filehandler_components.message_tracker import MessageTracker
from .filehandler_components.cleanup_scheduler import CleanupScheduler
from .filehandler_components.message_deleter import MessageDeleter

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol


class DiscordFileHandler:
    """Simplified handler for message tracking and automatic deletion with specialized components."""
    
    def __init__(self, bot, logger, config: "ConfigProtocol", tracking_file="data/tracked_messages.json", cleanup_interval=7200):
        """Initialize DiscordFileHandler with bot, logger, and config.
        
        Args:
            bot: Discord bot instance
            logger: Logger instance
            config: ConfigProtocol instance for message expiry settings
            tracking_file: Path to tracking persistence file
            cleanup_interval: Cleanup interval in seconds
        """
        self.bot = bot
        self.logger = logger
        self.config = config
        self.is_initialized = False
        
        # Initialize specialized components
        self.persistence = TrackingPersistence(tracking_file, logger)
        self.tracker = MessageTracker(self.persistence, logger, config)
        self.scheduler = CleanupScheduler(cleanup_interval, logger)
        self.deleter = MessageDeleter(bot, logger)
    
    def initialize(self):
        """Initialize the file handler and start background tasks."""
        self.is_initialized = True
        self.scheduler.start_cleanup_task(self.bot, self.check_and_delete_expired_messages)
        self.logger.info("DiscordFileHandler initialized with specialized components")
    
    async def track_message(self, message_id: int, channel_id: int, user_id: int, 
                          message_type: str = "general", expire_after: Optional[int] = None) -> bool:
        """Track a message for automatic deletion."""
        if not self.is_initialized:
            self.logger.debug("FileHandler not yet initialized, waiting for ready event...")
            # Wait for the bot to be ready and FileHandler to be initialized
            if hasattr(self.bot, 'discord_notifier') and hasattr(self.bot.discord_notifier, 'wait_until_ready'):
                try:
                    await asyncio.wait_for(self.bot.discord_notifier.wait_until_ready(), timeout=10.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for FileHandler initialization, cannot track message")
                    return False
            else:
                self.logger.warning("FileHandler not initialized and no ready event available, cannot track message")
                return False
        
        return await self.tracker.track_message(message_id, channel_id, user_id, message_type, expire_after)
    
    async def check_and_delete_expired_messages(self) -> int:
        """Check for and delete all expired messages."""
        try:
            expired_messages = await self.tracker.get_expired_messages()
            if not expired_messages:
                return 0
            
            deleted_count = await self._delete_expired_messages(expired_messages)
            
            if deleted_count > 0:
                self.logger.info(f"Successfully deleted {deleted_count} expired messages during cleanup")
            
            return deleted_count
        except Exception as e:
            self.logger.error(f"Error in check_and_delete_expired_messages: {e}")
            return 0
    
    async def _delete_expired_messages(self, expired_messages) -> int:
        """Delete a list of expired messages."""
        deleted_count = 0
        
        for message_id, channel_id in expired_messages:
            success = await self._process_single_message_deletion(message_id, channel_id)
            if success:
                deleted_count += 1
        
        return deleted_count
    
    async def _process_single_message_deletion(self, message_id: int, channel_id: int) -> bool:
        """Process deletion of a single message."""
        try:
            success = await self.deleter.try_delete_message(message_id, channel_id)
            if success:
                await self.tracker.remove_message_tracking(message_id)
                return True
        except Exception as e:
            self.logger.error(f"Error processing deletion for message {message_id}: {e}")
        
        return False
    
    async def get_tracking_stats(self):
        """Get statistics about tracked messages."""
        stats = await self.tracker.get_tracking_stats()
        stats["active_deletion_tasks"] = self.scheduler.get_task_count()
        return stats
    
    async def shutdown(self):
        """Clean up resources and cancel all background tasks."""
        try:
            await self.scheduler.shutdown()
            self.is_initialized = False
            self.logger.info("DiscordFileHandler shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during DiscordFileHandler shutdown: {e}")
