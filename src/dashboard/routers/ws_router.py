"""WebSocket router for real-time dashboard updates."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from typing import Set, Dict, Any

router = APIRouter(tags=["websocket"])

connected_clients: Set[WebSocket] = set()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception:
        connected_clients.discard(websocket)


async def broadcast(data: Dict[str, Any]) -> None:
    """Broadcast data to all connected WebSocket clients."""
    if not connected_clients:
        return
    for client in connected_clients.copy():
        try:
            await client.send_json(data)
        except Exception:
            connected_clients.discard(client)


@router.get("/api/status/countdown")
async def get_countdown(request: Request) -> Dict[str, Any]:
    """Get countdown to next analysis (REST fallback for WebSocket)."""
    dashboard_state = getattr(request.app.state, 'dashboard_state', None)
    if dashboard_state:
        return dashboard_state.get_countdown_data()
    return {"next_check_utc": None, "seconds_remaining": None}
