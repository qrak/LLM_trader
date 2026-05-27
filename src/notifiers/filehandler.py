import asyncio
import json
import os
from datetime import datetime
from typing import Any, TYPE_CHECKING

import discord

from src.utils.decorators import retry_async

if TYPE_CHECKING:
    from src.config.loader import Config


class DiscordFileHandler:
    """Tracks Discord messages and deletes them after expiry."""

    def __init__(
        self,
        bot,
        logger,
        config: "Config",
        tracking_file: str = "data/tracked_messages.json",
        cleanup_interval: int = 7200,
    ):
        self.bot = bot
        self.logger = logger
        self.config = config
        self.tracking_file = tracking_file
        self.cleanup_interval = cleanup_interval
        self.cleanup_task: asyncio.Task | None = None
        self._tracking_lock = asyncio.Lock()
        self.is_initialized = False
        tracking_dir = os.path.dirname(self.tracking_file)
        if tracking_dir:
            os.makedirs(tracking_dir, exist_ok=True)

    def initialize(self):
        """Initialize the file handler and start background tasks."""
        self.is_initialized = True
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
        self.cleanup_task = self.bot.loop.create_task(self._periodic_cleanup_loop(), name="MessageCleanupTask")
        self.logger.debug("DiscordFileHandler initialized")

    async def track_message(
        self,
        message_id: int,
        channel_id: int,
        user_id: int,
        message_type: str = "general",
        expire_after: int | None = None,
    ) -> bool:
        """Track a message for automatic deletion."""
        if not self.is_initialized:
            self.logger.debug("FileHandler not yet initialized, waiting for ready event...")
            try:
                ready_coro = self.bot.discord_notifier.wait_until_ready()
                try:
                    await asyncio.wait_for(ready_coro, timeout=10.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for FileHandler initialization, cannot track message")
                    return False
            except AttributeError:
                self.logger.warning("FileHandler not initialized and no ready event available, cannot track message")
                return False

        if expire_after is None:
            expire_after = self.config.FILE_MESSAGE_EXPIRY
        now = datetime.now()
        message_data = {
            "channel_id": channel_id,
            "user_id": user_id,
            "message_type": message_type,
            "tracked_at": now.isoformat(),
            "expire_after": expire_after,
            "expires_at": now.timestamp() + expire_after,
        }

        async with self._tracking_lock:
            try:
                tracking_data = await self._load_tracking_data()
                tracking_data[str(message_id)] = message_data
                success = await self._save_tracking_data(tracking_data)
                if success:
                    self.logger.debug("Tracking message %s for deletion", message_id)
                return success
            except Exception as e:
                self.logger.error("Error tracking message %s: %s", message_id, e)
                return False

    async def check_and_delete_expired_messages(self) -> int:
        """Check for and delete all expired messages."""
        try:
            expired_messages = await self._get_expired_messages()
            if not expired_messages:
                return 0

            deleted_count = await self._delete_expired_messages(expired_messages)

            if deleted_count > 0:
                self.logger.info("Successfully deleted %s expired messages during cleanup", deleted_count)

            return deleted_count
        except Exception as e:
            self.logger.error("Error in check_and_delete_expired_messages: %s", e)
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
            success = await self._try_delete_message(message_id, channel_id)
            if success:
                await self._remove_message_tracking(message_id)
                return True
        except Exception as e:
            self.logger.error("Error processing deletion for message %s: %s", message_id, e)

        return False

    async def _periodic_cleanup_loop(self):
        try:
            await asyncio.sleep(10)
            while self.is_initialized:
                try:
                    deleted_count = await self.check_and_delete_expired_messages()
                    if deleted_count > 0:
                        self.logger.debug("Cleaned up %s expired messages", deleted_count)
                except asyncio.CancelledError:
                    self.logger.info("Message cleanup task cancelled")
                    break
                except Exception as e:
                    self.logger.error("Error in scheduled message cleanup: %s", e)
                await asyncio.sleep(self.cleanup_interval)
        except asyncio.CancelledError:
            self.logger.info("Periodic message cleanup task cancelled")
        except Exception as e:
            self.logger.error("Unexpected error in periodic message cleanup task: %s", e)

    async def _get_expired_messages(self) -> list[tuple[int, int]]:
        async with self._tracking_lock:
            tracking_data = await self._load_tracking_data()

        current_time = datetime.now().timestamp()
        expired_messages = []
        for message_id_str, data in tracking_data.items():
            try:
                expires_at = data.get("expires_at")
                if expires_at is not None and current_time >= expires_at:
                    expired_messages.append((int(message_id_str), data["channel_id"]))
            except (ValueError, KeyError) as e:
                self.logger.warning("Invalid tracking data for message %s: %s", message_id_str, e)
        return expired_messages

    async def _load_tracking_data(self) -> dict[str, Any]:
        if not os.path.exists(self.tracking_file):
            return {}
        try:
            with open(self.tracking_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            self.logger.warning("Corrupted tracking file. Creating new.")
            return {}
        except Exception as e:
            self.logger.error("Error loading tracking data: %s", e)
            return {}

    async def _save_tracking_data(self, data: dict[str, Any]) -> bool:
        try:
            with open(self.tracking_file, "w", encoding="utf-8") as file:
                json.dump(data, file)
            return True
        except Exception as e:
            self.logger.error("Error saving tracking data: %s", e)
            return False

    async def _remove_message_tracking(self, message_id: int) -> None:
        async with self._tracking_lock:
            tracking_data = await self._load_tracking_data()
            tracking_data.pop(str(message_id), None)
            await self._save_tracking_data(tracking_data)

    async def _try_delete_message(self, message_id: int, channel_id: int) -> bool:
        try:
            deleted = await self._delete_message(message_id, channel_id)
            if deleted:
                self.logger.debug("Successfully deleted message %s", message_id)
            return deleted
        except discord.NotFound:
            self.logger.debug("Message %s already deleted (NotFound)", message_id)
            return True
        except Exception as e:
            self.logger.error("Error deleting message %s: %s", message_id, e)
            return False

    @retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def _delete_message(self, message_id: int, channel_id: int) -> bool:
        if not self.bot or self.bot.is_closed():
            self.logger.warning("Bot closed, cannot delete message %s", message_id)
            return False

        channel = self.bot.get_channel(channel_id)
        if not channel:
            self.logger.warning("Channel %s not found for message %s; removing tracking entry", channel_id, message_id)
            return True

        try:
            message = await channel.fetch_message(message_id)
            await message.delete()
            return True
        except discord.NotFound:
            return True
        except discord.Forbidden:
            self.logger.warning("Missing permissions to delete message %s", message_id)
            return False
        except discord.HTTPException as e:
            self.logger.error("Failed to delete message %s due to HTTP error: %s", message_id, e)
            return False

    async def shutdown(self):
        """Clean up resources and cancel all background tasks."""
        try:
            self.is_initialized = False
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
            self.cleanup_task = None
            self.logger.info("DiscordFileHandler shutdown complete")
        except Exception as e:
            self.logger.error("Error during DiscordFileHandler shutdown: %s", e)
