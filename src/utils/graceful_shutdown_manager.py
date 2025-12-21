import asyncio
import signal
import sys

class GracefulShutdownManager:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def setup_signal_handlers(self):
        if sys.platform != 'win32':
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.loop.add_signal_handler(sig, lambda s=sig, *args: self.handle_signal(s))
        else:
            signal.signal(signal.SIGINT, lambda s, f, *args: self.handle_signal(s))

    def handle_signal(self, sig: int):
        print(f"Received signal {sig}, initiating shutdown...")
        if self.loop.is_running() and not self.loop.is_closed():
            self.loop.create_task(self.shutdown_gracefully())

    async def shutdown_gracefully(self):
        print("Performing graceful shutdown...")
        pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if pending_tasks:
            print(f"Cancelling {len(pending_tasks)} tasks...")
            for task in pending_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.wait(pending_tasks), timeout=10.0)
            except asyncio.TimeoutError:
                task_details = []
                for t in pending_tasks:
                    if not t.done():
                        if hasattr(t, 'get_name') and callable(t.get_name):
                            name = t.get_name()
                        elif hasattr(t, 'name'):
                            name = t.name
                        else:
                            name = str(t)
                        task_details.append(name)
                print(f"Some tasks didn't complete in time: {task_details}")
        try:
            await asyncio.wait_for(self.loop.shutdown_asyncgens(), timeout=2.0)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Error shutting down async generators: {e}")
        self.loop.stop()
