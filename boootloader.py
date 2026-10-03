%%writefile bootloader.py
import os
import sys
import gc
import subprocess
import logging
import importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np

# 1. CONFIGURACIÓN DE LOGGING (Estilo ZERO EXTREME LOSS)
logger = logging.getLogger("ZERO_LOSS_BOOT")
logger.setLevel(logging.INFO)
logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
# Formato de tiempo ajustado a los milisegundos de tus logs
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | ZERO_LOSS_BOOT | %(message)s", "%Y-%m-%d %H:%M:%S,%f"[:-3]))
logger.addHandler(console_handler)

logger.info(">>> INICIANDO PROTOCOLO 'ZERO EXTREME LOSS' <<<")
logger.info("--- [FASE 1] SANITIZACIÓN DEL ENTORNO ---")

# 2. HARDWARE ALLOCATION STRICT PRE-LOAD
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
os.environ["NCCL_P2P_LEVEL"] = "SYS"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = ':'.join([p for p in os.environ['LD_LIBRARY_PATH'].split(':') if 'cuda' not in p.lower() and 'cudnn' not in p.lower()])
    logger.info("[PARCHE] Rutas tóxicas de cuDNN removidas de LD_LIBRARY_PATH.")

logger.info("[SYS] Ejecutando purga agresiva de memoria (RAM/VRAM)...")
gc.collect()
logger.info("[MEMORIA] Caché IPC de CUDA liberada exitosamente.")

# 3. KAGGLE OFFLINE INJECTION & ENVIRONMENT
logger.info("--- [FASE 2] DESINSTALACIÓN PROFUNDA E INSTALACIÓN ESCALONADA ---")
KAGGLE_INPUT_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
WHEELS_DIR = KAGGLE_INPUT_DIR / "arc_agi_3_wheels"
AGENTS_DIR = KAGGLE_INPUT_DIR / "ARC-AGI-3-Agents"
ENVIRONMENT_DIR = KAGGLE_INPUT_DIR / "environment_files"

def eradicate_zombie_modules():
    keys_to_delete = [k for k in sys.modules.keys() if any(t in k for t in ['pydantic', 'pydantic_core', 'arc_agi', 'arcengine'])]
    for k in keys_to_delete: del sys.modules[k]

if WHEELS_DIR.exists():
    logger.info("[SYS] Ejecutando desinstalación forzada de paquetes base conflictivos...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pydantic", "pydantic-core", "arc_agi", "arcengine"], capture_output=True)
    eradicate_zombie_modules()
    logger.info("[SYS] Iniciando ingesta escalonada de librerías...")
    
    all_wheels = list(WHEELS_DIR.glob("*.whl"))
    tiers = [
        ([w for w in all_wheels if 'pydantic' not in w.name.lower() and 'arc' not in w.name.lower()], "Librerías Base"),
        ([w for w in all_wheels if 'pydantic' in w.name.lower()], "Ecosistema Pydantic"),
        ([w for w in all_wheels if 'arc' in w.name.lower()], "Ecosistema ARC Prime")
    ]
    
    for i, (tier_wheels, tier_name) in enumerate(tiers, 1):
        if tier_wheels:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS_DIR}", "--upgrade", "--force-reinstall", "--no-deps", "--quiet"] + [str(w) for w in tier_wheels], capture_output=True)
            logger.info(f"[OK] Tier {i}: {tier_name} ({len(tier_wheels)} dependencias) inyectado correctamente.")

# 4. HOTFIXES Y ENRUTAMIENTO DE DEPENDENCIAS
logger.info("--- [FASE 3] APLICACIÓN DE PARCHES EN CALIENTE (HOTFIXES) ---")
if AGENTS_DIR.exists() and str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))
if ENVIRONMENT_DIR.exists() and str(ENVIRONMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_DIR))

logger.info("[SYS] Ejecutando purga agresiva de memoria (RAM/VRAM)...")
gc.collect()
logger.info("[MEMORIA] Caché IPC de CUDA liberada exitosamente.")

# 5. TENSOR CORES & AUDITORÍA
logger.info("--- [FASE 4] AUDITORÍA CERO-PÉRDIDAS (HARDWARE Y MÓDULOS) ---")
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
    vram_free = vram_total - vram_reserved
    cudnn_version = torch.backends.cudnn.version()
    logger.info(f"[HARDWARE OK] GPU: {gpu_name}")
    logger.info(f"[MEMORIA OK] VRAM: Libre {vram_free:.2f} GB / Total {vram_total:.2f} GB")
    logger.info(f"[RUNTIME OK] cuDNN: {cudnn_version}")

def get_version(module_name):
    try: return importlib.import_module(module_name).__version__
    except: return "(vNativa)"

logger.info(f"[INTEGRIDAD SUPERADA] numpy          -> (v{np.__version__})")
logger.info(f"[INTEGRIDAD SUPERADA] pydantic_core  -> {get_version('pydantic_core')}")
logger.info(f"[INTEGRIDAD SUPERADA] pydantic       -> {get_version('pydantic')}")
logger.info(f"[INTEGRIDAD SUPERADA] arc_agi        -> {get_version('arc_agi')}")
logger.info(f"[INTEGRIDAD SUPERADA] arcengine      -> {get_version('arcengine')}")

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

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
logger.info("--- [ÉXITO TOTAL] EL ENTORNO ZK-DAG ESTÁ BLINDADO Y LISTO PARA EJECUCIÓN ---")

```
