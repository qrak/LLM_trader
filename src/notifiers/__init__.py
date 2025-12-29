"""
Discord interface package - Send-only notification with message expiration.
Also provides ConsoleNotifier as fallback when Discord is disabled.
"""
from .base_notifier import BaseNotifier
from .notifier import DiscordNotifier
from .filehandler import DiscordFileHandler
from .console_notifier import ConsoleNotifier

__all__ = ['BaseNotifier', 'DiscordNotifier', 'DiscordFileHandler', 'ConsoleNotifier']

