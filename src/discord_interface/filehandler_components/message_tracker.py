"""
Message tracking manager.
Handles the core logic for tracking messages and determining expired messages.
"""
import asyncio
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.config import ConfigProtocol


class MessageTracker:
    """Handles message tracking logic and expiration management."""
    
    def __init__(self, persistence_handler, logger, config: "ConfigProtocol"):
        """Initialize MessageTracker with persistence handler, logger, and config.
        
        Args:
            persistence_handler: TrackingPersistence instance
            logger: Logger instance
            config: ConfigProtocol instance for message expiry settings
        """
        self.persistence = persistence_handler
        self.logger = logger
        self.config = config
        self._tracking_lock = asyncio.Lock()
    
    async def track_message(self, message_id: int, channel_id: int, user_id: int, 
                          message_type: str = "general", expire_after: Optional[int] = None) -> bool:
        """Track a message for automatic deletion."""
        if expire_after is None:
            expire_after = self.config.FILE_MESSAGE_EXPIRY
        
        message_data = self._create_message_data(channel_id, user_id, message_type, expire_after)
        
        async with self._tracking_lock:
            return await self._save_message_tracking(message_id, message_data)
    
    def _create_message_data(self, channel_id: int, user_id: int, 
                           message_type: str, expire_after: int) -> Dict[str, Any]:
        """Create message tracking data structure."""
        now = datetime.now()
        expiry_time = now.timestamp() + expire_after
        
        return {
            "channel_id": channel_id,
            "user_id": user_id,
            "message_type": message_type,
            "tracked_at": now.isoformat(),
            "expire_after": expire_after,
            "expires_at": expiry_time
        }
    
    async def _save_message_tracking(self, message_id: int, message_data: Dict[str, Any]) -> bool:
        """Save message tracking data."""
        try:
            tracking_data = await self.persistence.load_tracking_data()
            tracking_data[str(message_id)] = message_data
            success = await self.persistence.save_tracking_data(tracking_data)
            
            if success:
                self.logger.debug(f"Tracking message {message_id} for deletion")
            return success
        except Exception as e:
            self.logger.error(f"Error tracking message {message_id}: {e}")
            return False
    
    async def get_expired_messages(self) -> List[Tuple[int, int]]:
        """Get all expired messages that need deletion."""
        async with self._tracking_lock:
            tracking_data = await self.persistence.load_tracking_data()
            
        current_time = datetime.now().timestamp()
        expired_messages = []
        
        for message_id_str, data in tracking_data.items():
            if self._is_message_expired(data, current_time):
                try:
                    message_id = int(message_id_str)
                    channel_id = data['channel_id']
                    expired_messages.append((message_id, channel_id))
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Invalid tracking data for message {message_id_str}: {e}")
        
        return expired_messages
    
    def _is_message_expired(self, message_data: Dict[str, Any], current_time: float) -> bool:
        """Check if a message has expired."""
        try:
            expires_at = message_data.get('expires_at')
            return expires_at is not None and current_time >= expires_at
        except Exception:
            return False
    
    async def remove_message_tracking(self, message_id: int) -> None:
        """Remove tracking for a specific message."""
        async with self._tracking_lock:
            await self.persistence.remove_message_tracking(message_id)
    
    async def get_tracking_stats(self) -> Dict[str, int]:
        """Get statistics about tracked messages."""
        tracking_data = await self.persistence.load_tracking_data()
        current_time = datetime.now().timestamp()
        
        total_tracked = len(tracking_data)
        expired_count = sum(
            1 for data in tracking_data.values()
            if self._is_message_expired(data, current_time)
        )
        
        return {
            "total_tracked": total_tracked,
            "expired_count": expired_count,
            "active_count": total_tracked - expired_count
        }
