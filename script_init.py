"""
Faucon Sentinel - ZERO Extreme Loss & Hardware Maximization Bootloader
Architecture: NVIDIA RTX 6000 Ada / Blackwell G4 (48 GB)
Environment: Strict Offline (Kaggle / Jupyter)
"""

import os
import sys

# ==============================================================================
# FASE 0: CONFIGURACIÓN DE BAJO NIVEL (DEBE OCURRIR ANTES DE IMPORTAR TORCH)
# ==============================================================================
# Prevención de fragmentación de VRAM agresiva
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
os.environ["NCCL_P2P_LEVEL"] = "SYS"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Limpieza de LD_LIBRARY_PATH para evitar choques de cuDNN
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = ':'.join(
        [p for p in os.environ['LD_LIBRARY_PATH'].split(':') if 'cuda' not in p.lower() and 'cudnn' not in p.lower()]
    )

import gc
import subprocess
import logging
import importlib.util
from pathlib import Path
from datetime import datetime
import numpy as np

# ==============================================================================
# FASE 1: I/O DE ULTRA BAJA LATENCIA (RAM DISK) Y LOGGING
# ==============================================================================
def setup_high_speed_io() -> tuple[str, logging.Logger]:
    """Redirige el I/O al disco RAM (/dev/shm) para evitar cuellos de botella SSD."""
    base_dir = '/dev/shm/arc_runs'
    if not os.path.exists('/dev/shm') or not os.access('/dev/shm', os.W_OK):
        base_dir = '/tmp/arc_runs'
        
    run_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    logger = logging.getLogger("FAUCON_BOOT")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | ZERO_LOSS | %(message)s")
    
    file_handler = logging.FileHandler(os.path.join(run_dir, 'boot.log'), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return run_dir, logger

RUN_DIR, logger = setup_high_speed_io()
logger.info(f"[IO_SETUP] RAM Disk configurado en: {RUN_DIR}")

# ==============================================================================
# FASE 2: PURGA DE MÓDULOS ZOMBIE E INSTALACIÓN OFFLINE ESCALONADA
# ==============================================================================
KAGGLE_INPUT_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
WHEELS_DIR = KAGGLE_INPUT_DIR / "arc_agi_3_wheels"
AGENTS_DIR = KAGGLE_INPUT_DIR / "ARC-AGI-3-Agents"

def force_hardware_sync():
    """Vaciado absoluto de caché IPC, VRAM y Garbage Collector."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

def eradicate_zombie_modules():
    targets = ['pydantic', 'pydantic_core', 'arc_agi', 'arcengine']
    keys_to_delete = [k for k in sys.modules.keys() if any(t in k for t in targets)]
    for k in keys_to_delete:
        del sys.modules[k]
    if keys_to_delete:
        logger.info(f"[MEM_PURGE] {len(keys_to_delete)} sub-módulos aniquilados de sys.modules.")

def inject_offline_dependencies():
    if not WHEELS_DIR.exists():
        logger.warning(f"[WARN] Directorio Wheels ausente: {WHEELS_DIR}")
        return

    logger.info("[SYS] Forzando purga de paquetes conflictivos...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                    "pydantic", "pydantic-core", "pydantic-settings", "arc_agi", "arcengine"], 
                   capture_output=True)
    
    eradicate_zombie_modules()
    
    all_wheels = list(WHEELS_DIR.glob("*.whl"))
    if not all_wheels: return

    tiers = {
        "Base": [w for w in all_wheels if 'pydantic' not in w.name.lower() and 'arc' not in w.name.lower()],
        "Pydantic": [w for w in all_wheels if 'pydantic' in w.name.lower()],
        "ARC": [w for w in all_wheels if 'arc' in w.name.lower()]
    }

    for tier_name, wheels in tiers.items():
        if not wheels: continue
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS_DIR}", 
               "--upgrade", "--force-reinstall", "--no-deps", "--quiet"] + [str(w) for w in wheels]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"[INJECT OK] Ecosistema {tier_name} instalado.")
        except subprocess.CalledProcessError as e:
            logger.error(f"[INJECT FAIL] {tier_name}: {e.stderr.decode('utf-8')[:200]}")

    if AGENTS_DIR.exists() and str(AGENTS_DIR) not in sys.path:
        sys.path.insert(0, str(AGENTS_DIR))

# ==============================================================================
# FASE 3: IMPORTACIÓN DE TENSORES, PARCHES Y AUDITORÍA FINAL
# ==============================================================================
inject_offline_dependencies()

import torch
import torch.nn as nn
import torch.nn.functional as F

# CONFIGURACIÓN TENSOR CORES (ADA / BLACKWELL MAXIMIZATION)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# CRÍTICO: False para ARC AGI. Las variaciones en los grids destruirían el rendimiento si es True.
torch.backends.cudnn.benchmark = False 

# HOTFIX PYDANTIC
try:
    import pydantic._internal._typing_extra as typing_extra
    if not hasattr(typing_extra, 'signature_no_eval'):
        import inspect
        setattr(typing_extra, 'signature_no_eval', lambda *a, **kw: inspect.signature(*a, **kw))
        
    import pydantic_core.core_schema as core_schema
    if not hasattr(core_schema, 'iter_union_choices'):
        setattr(core_schema, 'iter_union_choices', lambda s: s.get('choices', []) if isinstance(s, dict) else [])
    logger.info("[HOTFIX] Runtime hooks inyectados con éxito.")
except Exception as e:
    pass

# MOCK CLASSES FALLBACK
try:
    from agents.agent import Agent
    from arcengine import FrameData, GameAction, GameState
    logger.info("[MODULOS] Clases de ARC nativas cargadas.")
except ImportError:
    logger.warning("[FALLBACK] Utilizando Mock Classes (Entorno ARC no disponible).")
    class Agent:
        def __init__(self, game_id: str="mock", *args, **kwargs): self.game_id = game_id; self.frames = []; self.action_counter = 0
    class GameState: NOT_PLAYED="NOT_PLAYED"; PLAYING="PLAYING"; WIN="WIN"; GAME_OVER="GAME_OVER"
    class GameAction:
        ACTION1="ACTION1"; ACTION2="ACTION2"; ACTION3="ACTION3"; ACTION4="ACTION4"; ACTION5="ACTION5"; ACTION6="ACTION6"; RESET="RESET"
        def __init__(self, name: str): self.name = name; self.reasoning = ""; self.data = {}
        def set_data(self, data: dict): self.data = data
    class FrameData:
        def __init__(self, frame=None, score=0, levels_completed=0, state=GameState.PLAYING, guid=""):
            self.frame = frame if frame is not None else np.zeros((64, 64), dtype=int)
            self.score = score; self.levels_completed = levels_completed; self.state = state; self.guid = guid; self.available_actions = [1, 2, 3, 4, 5, 6]

# FINAL AUDIT
force_hardware_sync()
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    vram_free, vram_total = torch.cuda.mem_get_info(device)
    logger.info(f"=== [SISTEMA LISTO] GPU: {torch.cuda.get_device_name(device)} | VRAM Libre: {vram_free/1e9:.2f}GB / {vram_total/1e9:.2f}GB ===")
else:
    logger.error("=== [CRÍTICO] HARDWARE CUDA NO DETECTADO ===")
