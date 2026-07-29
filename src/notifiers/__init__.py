"""
Discord interface package - Send-only notification with message expiration.
Also provides ConsoleNotifier as fallback when Discord is disabled.
"""
from .base_notifier import BaseNotifier
from .console_notifier import ConsoleNotifier
from .filehandler import DiscordFileHandler
from .notifier import DiscordNotifier

__all__ = ["BaseNotifier", "ConsoleNotifier", "DiscordFileHandler", "DiscordNotifier"]
