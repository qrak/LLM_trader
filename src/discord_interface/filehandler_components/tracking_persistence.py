"""
Message tracking persistence handler.
Handles loading and saving of message tracking data to JSON files.
"""
import json
import os
from typing import Dict, Any


class TrackingPersistence:
    """Handles persistence of message tracking data."""
    
    def __init__(self, tracking_file: str, logger):
        self.tracking_file = tracking_file
        self.logger = logger
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
    
    async def load_tracking_data(self) -> Dict[str, Any]:
        """Load all message tracking data."""
        if not os.path.exists(self.tracking_file):
            return {}
            
        try:
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.logger.warning("Corrupted tracking file. Creating new.")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading tracking data: {e}")
            return {}
    
    async def save_tracking_data(self, data: Dict[str, Any]) -> bool:
        """Save tracking data."""
        try:
            if data:
                with open(self.tracking_file, 'w') as f:
                    json.dump(data, f)
            elif os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'w') as f:
                    json.dump({}, f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving tracking data: {e}")
            return False
    
    async def remove_message_tracking(self, message_id: int) -> None:
        """Remove tracking data for a specific message."""
        tracking_data = await self.load_tracking_data()
        str_message_id = str(message_id)
        if str_message_id in tracking_data:
            del tracking_data[str_message_id]
            await self.save_tracking_data(tracking_data)
