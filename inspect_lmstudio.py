
import asyncio
import inspect
from typing import Any
try:
    import lmstudio as lms
except ImportError:
    print("lmstudio not installed")
    exit(1)

async def main():
    print("--- Inspecting lmstudio package ---")
    print(f"lmstudio version (if avail): {getattr(lms, '__version__', 'unknown')}")
    print(f"lmstudio dir: {dir(lms)}")

    print("\n--- Inspecting lms.llm module ---")
    try:
        from lmstudio import llm
    except Exception as e:
        print(f"Error inspecting model: {type(e).__name__}: {e}")

    print("\n=== LlmPredictionConfig Fields ===")
    try:
        from lmstudio.json_api import LlmPredictionConfig
        # inspect is already imported at the top of the file
        print(inspect.signature(LlmPredictionConfig))
        print(LlmPredictionConfig.__annotations__)
    except ImportError:
        print("Could not import LlmPredictionConfig")
    except Exception as e:
        print(f"Error inspecting config: {e}")

    print("\n--- Inspecting Loaded Model Object ---")
    try:
        async with lms.AsyncClient() as client:
            loaded = await client.llm.list_loaded()
            if loaded:
                model_identifier = loaded[0].identifier
                print(f"Loading model handle for: {model_identifier}")
                model_handle = await client.llm.model(model_identifier)
                print(f"Model Handle: {model_handle}")
                
                if hasattr(model_handle, 'respond_stream'):
                    rs = model_handle.respond_stream
                    print(f"\nrespond_stream: {rs}")
                    print(f"Type: {type(rs)}")
                    print(f"Is Coroutine Function: {inspect.iscoroutinefunction(rs)}")
                    print(f"Docstring: {rs.__doc__}")
            else:
                print("No loaded models found to inspect.")
    except Exception as e:
        print(f"Error inspecting model handle: {e}")

if __name__ == "__main__":
    asyncio.run(main())
