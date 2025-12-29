import discord
from src.utils.decorators import retry_async


class MessageDeleter:
    """Handles Discord message deletion with retry logic and error handling."""
    
    def __init__(self, bot, logger):
        self.bot = bot
        self.logger = logger
    
    async def try_delete_message(self, message_id: int, channel_id: int) -> bool:
        """Try to delete a message, handling all error cases."""
        try:
            result = await self._delete_message(message_id, channel_id)
            if result:
                self.logger.debug(f"Successfully deleted message {message_id}")
                return True
            return False
        except discord.NotFound:
            self.logger.debug(f"Message {message_id} already deleted (NotFound)")
            return True  # Message doesn't exist, which is our goal
        except Exception as e:
            self.logger.error(f"Error deleting message {message_id}: {e}")
            return False
    
    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def _delete_message(self, message_id: int, channel_id: int) -> bool:
        """Core message deletion logic with retry."""
        if not self.bot or self.bot.is_closed():
            self.logger.warning(f"Bot closed, cannot delete message {message_id}")
            return False
            
        channel = await self._get_channel(channel_id, message_id)
        if not channel:
            self.logger.warning(f"Channel {channel_id} not found for message {message_id}; removing tracking entry")
            return True
            
        return await self._delete_from_channel(message_id, channel)
    
    async def _get_channel(self, channel_id: int, message_id: int):
        """Get channel for message deletion."""
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return None
        return channel
    
    async def _delete_from_channel(self, message_id: int, channel) -> bool:
        """Delete message from specific channel."""
        try:
            message = await channel.fetch_message(message_id)
            await message.delete()
            return True
        except discord.NotFound:
            # Message already deleted - this is actually success
            return True
        except discord.Forbidden:
            self.logger.warning(f"Missing permissions to delete message {message_id}")
            return False
        except discord.HTTPException as e:
            self.logger.error(f"Failed to delete message {message_id} due to HTTP error: {e}")
            return False
