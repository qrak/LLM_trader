import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from utils.dataclass import Position, TradeDecision


class DataPersistence:
    def __init__(self, logger, data_dir: str = "trading_data") -> None:
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.positions_file = self.data_dir / "positions.json"
        self.history_file = self.data_dir / "trade_history.json"
        self.previous_response_file = self.data_dir / "previous_response.json"

    def save_previous_response(self, response: str) -> None:
        with open(self.previous_response_file, 'w') as f:
            json.dump({"response": response, "timestamp": datetime.now().isoformat()}, f)

    def load_previous_response(self) -> Optional[str]:
        if not self.previous_response_file.exists():
            return None
        try:
            with open(self.previous_response_file, 'r') as f:
                data = json.load(f)
                return data["response"]
        except Exception as e:
            self.logger.error(f"Unknown error: {e}")
            return None

    def save_position(self, position: Optional[Position]) -> None:
        try:
            if position is None:
                if self.positions_file.exists():
                    self.positions_file.unlink()
                return

            data = {
                "entry_price": position.entry_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "size": position.size,
                "entry_time": position.entry_time.isoformat(),
                "confidence": position.confidence,
                "direction": position.direction
            }

            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving position: {e}")

    def save_trade_decision(self, decision: TradeDecision) -> None:
        try:
            history = self.load_trade_history()

            trade_data = {
                "timestamp": decision.timestamp.isoformat(),
                "action": decision.action,
                "price": decision.price,
                "confidence": decision.confidence,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "position_size": decision.position_size,
                "reasoning": decision.reasoning
            }

            if "CLOSE" in decision.action.upper():
                trade_data["pnl"] = self._calculate_pnl(history, decision)

            history.append(trade_data)

            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving trade decision: {e}")

    def load_position(self) -> Optional[Position]:
        if not self.positions_file.exists():
            return None
        try:
            with open(self.positions_file, 'r') as f:
                data = json.load(f)
                return Position(
                    entry_price=data["entry_price"],
                    stop_loss=data["stop_loss"],
                    take_profit=data["take_profit"],
                    size=data["size"],
                    entry_time=datetime.fromisoformat(data["entry_time"]),
                    confidence=data.get("confidence", "MEDIUM"),
                    direction=data.get("direction", "LONG")
                )
        except Exception as e:
            self.logger.error(f"Unknown error: {e}")
            return None


    def _calculate_pnl(self, history: List[Dict[str, Any]],
                       current_decision: TradeDecision) -> float:
        if not history:
            return 0.0

        last_entry = None
        for trade in reversed(history):
            if trade["action"].upper() in ["BUY", "SELL"]:
                last_entry = trade
                break

        if not last_entry:
            return 0.0

        entry_price = float(last_entry["price"])
        exit_price = current_decision.price
        position_size = float(last_entry["position_size"])

        if last_entry["action"].upper() == "BUY":
            return ((exit_price - entry_price) / entry_price) * position_size * 100
        else:
            return ((entry_price - exit_price) / entry_price) * position_size * 100

    def load_last_n_decisions(self, n: int = 4) -> List[Dict[str, Any]]:
        history = self.load_trade_history()
        valid_actions = {"BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT"}
        filtered_history = [
            decision for decision in history
            if decision["action"].upper() in valid_actions
        ]
        filtered_history.sort(
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        return filtered_history[:n]

    def load_trade_history(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            valid_history = []
            required_fields = {
                "timestamp", "action", "price", "confidence",
                "stop_loss", "take_profit", "position_size", "reasoning"
            }
            for entry in history:
                if all(field in entry for field in required_fields):
                    valid_history.append(entry)
            return valid_history
        except Exception:
            return []