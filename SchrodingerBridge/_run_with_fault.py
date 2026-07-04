"""Run training with explicit faulthandler and exception catching."""
import sys
import os
import faulthandler
import traceback

# Enable faulthandler to dump tracebacks on segfault
faulthandler.enable(all_threads=True)
# Also enable faulthandler to dump tracebacks after a timeout (in seconds)
faulthandler.dump_traceback_later(60, repeat=True, exit=False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("[runner] Starting training...", flush=True)
try:
    from run import main
    main()
except Exception as e:
    print(f"[runner] EXCEPTION: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
except BaseException as e:
    print(f"[runner] BASE EXCEPTION: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
print("[runner] Training completed.", flush=True)
