%%writefile bootloader.py
import os
import sys

# 1. HARDWARE ALLOCATION STRICT PRE-LOAD
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
os.environ["NCCL_P2P_LEVEL"] = "SYS"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = ':'.join([p for p in os.environ['LD_LIBRARY_PATH'].split(':') if 'cuda' not in p.lower() and 'cudnn' not in p.lower()])

import gc
import subprocess
import logging
import importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np

# 2. SILENT I/O LOGGING (Only to RAM Disk, No Console Output)
base_dir = '/dev/shm/arc_runs'
if not os.path.exists('/dev/shm') or not os.access('/dev/shm', os.W_OK):
    base_dir = '/tmp/arc_runs'
run_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

logger = logging.getLogger("FAUCON_BOOT")
logger.setLevel(logging.INFO)
logger.handlers.clear()
file_handler = logging.FileHandler(os.path.join(run_dir, 'boot.log'), mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | ZERO_LOSS | %(message)s"))
logger.addHandler(file_handler)

# 3. KAGGLE OFFLINE INJECTION
KAGGLE_INPUT_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
WHEELS_DIR = KAGGLE_INPUT_DIR / "arc_agi_3_wheels"
AGENTS_DIR = KAGGLE_INPUT_DIR / "ARC-AGI-3-Agents"

def eradicate_zombie_modules():
    keys_to_delete = [k for k in sys.modules.keys() if any(t in k for t in ['pydantic', 'pydantic_core', 'arc_agi', 'arcengine'])]
    for k in keys_to_delete: del sys.modules[k]

if WHEELS_DIR.exists():
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pydantic", "pydantic-core", "arc_agi", "arcengine"], capture_output=True)
    eradicate_zombie_modules()
    all_wheels = list(WHEELS_DIR.glob("*.whl"))
    for tier in [ [w for w in all_wheels if 'pydantic' not in w.name.lower() and 'arc' not in w.name.lower()],
                  [w for w in all_wheels if 'pydantic' in w.name.lower()],
                  [w for w in all_wheels if 'arc' in w.name.lower()] ]:
        if tier: subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS_DIR}", "--upgrade", "--force-reinstall", "--no-deps", "--quiet"] + [str(w) for w in tier], capture_output=True)

if AGENTS_DIR.exists() and str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

# 4. TENSOR CORES & MOCKS
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False 

try:
    from agents.agent import Agent
    from arcengine import FrameData, GameAction, GameState
except ImportError:
    class Agent:
        def __init__(self, game_id="mock", *args, **kwargs): self.game_id = game_id; self.frames = []; self.action_counter = 0
    class GameState: NOT_PLAYED="NOT_PLAYED"; PLAYING="PLAYING"; WIN="WIN"; GAME_OVER="GAME_OVER"
    class GameAction:
        ACTION1="A1"; RESET="R"; def __init__(self, name): self.name = name; self.data = {}; def set_data(self, d): self.data = d
    class FrameData:
        def __init__(self, frame=None, score=0, levels_completed=0, state=GameState.PLAYING, guid=""):
            self.frame = frame if frame is not None else np.zeros((64, 64), dtype=int); self.score = score; self.levels_completed = levels_completed; self.state = state; self.guid = guid

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    logger.info("SYSTEM READY. VRAM CLEARED.")
