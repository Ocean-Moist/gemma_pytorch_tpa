import logging
import torch
import sys
import os

# --- Debugging Configuration ---
# Set environment variables like DEBUG=1, DEBUG_LAYER=0, DEBUG_HEAD=0, DEBUG_STEP_END=5 to control logging
DEBUG = os.environ.get("DEBUG", "0") == "1"
DEBUG_LAYER = int(os.environ.get("DEBUG_LAYER", -1)) # Target specific layer, -1 for all
DEBUG_HEAD = int(os.environ.get("DEBUG_HEAD", -1))   # Target specific head
DEBUG_STEP_START = int(os.environ.get("DEBUG_STEP_START", 0)) # Start logging from this step
DEBUG_STEP_END = int(os.environ.get("DEBUG_STEP_END", 9999))  # Stop logging after this step
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", 1))      # Log every N steps

# Configure logger to output DEBUG messages to stderr
logger = logging.getLogger("gcb_dbg")
logger.propagate = False # Prevent duplicate logging if root logger is configured
if not logger.handlers: # Add handler only if it doesn't exist
    handler = logging.StreamHandler(sys.stderr) # Output debug info to stderr
    formatter = logging.Formatter('%(message)s') # Simple format
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def should_log(l_idx, h_idx, step):
    """Checks if logging should occur based on environment variables."""
    if not DEBUG:
        return False
    if not (DEBUG_STEP_START <= step < DEBUG_STEP_END):
        return False
    # Log every step within the range if interval is 1, otherwise check interval
    if LOG_INTERVAL > 1 and step % LOG_INTERVAL != 0:
         # Always log step 0 if within range, regardless of interval
         if step != DEBUG_STEP_START and step != 0:
             return False
    # Layer/Head filtering
    if DEBUG_LAYER != -1 and l_idx != DEBUG_LAYER:
        return False
    if DEBUG_HEAD != -1 and h_idx != DEBUG_HEAD:
        return False
    return True

def dbg(tag: str, x: torch.Tensor, l_idx=-1, h_idx=-1, step=-1):
    """Logs tensor statistics if conditions are met."""
    if not should_log(l_idx, h_idx, step):
        return

    if x is None:
        logger.debug(f"[{l_idx},{h_idx},{step}] {tag:16s}  None")
        return

    try:
        x_float = x.float()
        has_nan = torch.isnan(x_float).any().item()
        has_inf = torch.isinf(x_float).any().item()
        shape = tuple(x.shape)
        dtype = x.dtype
        mean_val = x_float.mean().item()
        rms_val = x_float.pow(2).mean().sqrt().item()
        min_val = x_float.min().item()
        max_val = x_float.max().item()

        stats_str = (
            f"shape={str(shape):<15} dtype={str(dtype):<12} "
            f"mean={mean_val:+9.3e}  rms={rms_val:9.3e}  "
            f"min={min_val:+9.3e}  max={max_val:+9.3e}"
        )
        nan_inf_str = ""
        if has_nan: nan_inf_str += " !!!HAS NAN!!!"
        if has_inf: nan_inf_str += " !!!HAS INF!!!"

        logger.debug(f"[{l_idx:02d},{h_idx:02d},{step:03d}] {tag:18s}  {stats_str}{nan_inf_str}")

    except Exception as e:
        logger.error(f"[{l_idx},{h_idx},{step}] Error logging tag '{tag}': {e}")

# Function to log assertion failures/warnings to stdout
def check(condition, message, l_idx=-1, h_idx=-1, step=-1, level="ERROR"):
    if not condition:
        log_prefix = f"[{l_idx:02d},{h_idx:02d},{step:03d}] {level}:"
        print(f"{log_prefix} {message}", file=sys.stdout) # Use stdout for failures
        if level == "ERROR":
             raise AssertionError(message) # Optional: Stop execution on critical errors
        return False
    return True