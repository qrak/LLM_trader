# Discord Interface Documentation

**Parent Instructions**: See `/AGENTS.md` for global project context and universal coding guidelines.

---

## Overview

The Discord interface layer provides **send-only notification** capability with automatic message expiration. This is NOT a command bot - it only sends AI trading analysis to Discord channels.

## Architecture

### Core Principle
- **Send-only**: No user commands, no interaction handlers
- **Automatic expiration**: Messages are automatically deleted after configurable cooldown
- **Simple integration**: Single `DiscordNotifier` class for all notification needs

## Directory Structure

```
src/discord_interface/
├── __init__.py              # Package exports
├── AGENTS.md                # This documentation
├── notifier.py              # Main notifier class (send-only)
├── filehandler.py           # Message tracking orchestration
└── filehandler_components/  # Message lifecycle management
    ├── __init__.py
    ├── message_tracker.py   # Core tracking logic
    ├── tracking_persistence.py  # JSON persistence
    ├── cleanup_scheduler.py # Background cleanup tasks
    └── message_deleter.py   # Safe message deletion
```

## Components

### DiscordNotifier

**Location**: `notifier.py`

**Purpose**: Send-only Discord client with message expiration tracking.

**Key Methods**:
- `send_message(message, channel_id, expire_after)` - Send text message
- `send_trading_analysis(analysis_text, channel_id, symbol, expire_after)` - Send formatted trading analysis
- `wait_until_ready()` - Wait for bot initialization
- `start()` - Start the bot (run in background task)

**Usage**:
```python
notifier = DiscordNotifier(logger, config)

# Start bot in background
asyncio.create_task(notifier.start())
await notifier.wait_until_ready()

# Send trading analysis
await notifier.send_trading_analysis(
    analysis_text="AI response text...",
    channel_id=config.MAIN_CHANNEL_ID,
    symbol="BTC/USDT"
)
```

### DiscordFileHandler

**Location**: `filehandler.py`

**Purpose**: Orchestrates message tracking and automatic deletion.

**Responsibilities**:
- Track sent messages with expiration times
- Schedule periodic cleanup tasks
- Delete expired messages from Discord

### Message Lifecycle Components

**Location**: `filehandler_components/`

| Component | File | Responsibility |
|-----------|------|----------------|
| `MessageTracker` | `message_tracker.py` | Core tracking logic, expiration management |
| `TrackingPersistence` | `tracking_persistence.py` | JSON file persistence for tracked messages |
| `CleanupScheduler` | `cleanup_scheduler.py` | Background task scheduling |
| `MessageDeleter` | `message_deleter.py` | Safe message deletion with retry |

## Configuration

**Required in `keys.env`**:
```env
BOT_TOKEN_DISCORD=your_bot_token
```

**Required in `config.ini`**:
```ini
[cooldowns]
# Message expiry in hours (converted to seconds at runtime)
file_message_expiry = 168
```

## Message Expiration Flow

```
send_message() / send_trading_analysis()
    ↓
Discord's delete_after (primary expiration)
    ↓
FileHandler.track_message() (backup tracking)
    ↓
CleanupScheduler (periodic cleanup loop)
    ↓
MessageDeleter (handles edge cases)
```

**Dual-layer expiration**:
1. Discord's built-in `delete_after` parameter (primary)
2. FileHandler tracking (backup for reliability)

## Integration Example

```python
from src.discord_interface import DiscordNotifier

class CryptoTradingBot:
    async def initialize(self):
        # Initialize notifier
        self.discord_notifier = DiscordNotifier(self.logger, config)
        
        # Start bot in background
        self._discord_task = asyncio.create_task(self.discord_notifier.start())
        await self.discord_notifier.wait_until_ready()
    
    async def _execute_trading_check(self):
        # ... run analysis ...
        
        # Send to Discord
        await self.discord_notifier.send_trading_analysis(
            analysis_text=result["response"],
            channel_id=config.MAIN_CHANNEL_ID,
            symbol=self.current_symbol
        )
    
    async def shutdown(self):
        async with self.discord_notifier:
            pass  # __aexit__ handles cleanup
```

## What Was Removed

The following components were removed from the original Discord interface (command-based):

- **Cogs directory**: Command handlers, anti-spam, reaction handling
- **Command processing**: User commands like `!analyze BTC/USDT`
- **Response builders**: Embed creation for analysis results
- **Symbol validation**: User input validation (not needed for send-only)

This simplification aligns with the trading bot's purpose: automated AI analysis sent to Discord, not user-initiated commands.
