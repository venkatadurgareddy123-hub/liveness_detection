"""
Selfie Authenticity Gateway v13.1  (liveness-only, 250 req/s target)
=======================================================
Key changes from v13.0:
  - _check_phone_obstruction Signal 3 replaced: hue-std uniformity → (1 - skin_below).
    Root cause: black phones have near-zero saturation → hue is numerically
    undefined → hue_std spikes to ~58 on a perfectly flat black surface →
    uniformity collapses to ~0.03 → score ~0.07, below threshold.
    Fix is colour-space agnostic: (1 - skin_below) measures whether the region
    below the boundary is non-skin. Any phone (black, teal, silver) has low
    skin ratio below its top edge. Beard chin: still ~0.65 skin → low score.
  - PHONE_HUE_STD_MAX config param removed (no longer used).
  - Version bump 13.0 → 13.1 in startup log and title.

Key changes from v12.5:
  - Added _check_phone_obstruction() — pixel-level rigid-object detector
    integrated as Pass 5 inside _check_mouth_visible().
  - Replaces the old wide-edge (Pass 4) heuristic for phone detection.
    The old Pass 4 triggered on beard transitions because it only measured
    edge width; the new detector requires ALL THREE of:
      (a) large skin drop at the boundary  (phone sits across face)
      (b) sharp single-row boundary        (rigid flat object, not gradual beard)
      (c) uniform colour below boundary    (phone body, not textured beard/collar)
  - PHONE_ZONE_MIN / PHONE_ZONE_MAX config: restricts phone search to the
    mouth region of the face (0.40–0.82 y-fraction). Glasses frames trigger
    at y<0.40 (excluded); shirt collar at y>0.82 (excluded).
  - PHONE_SCORE_THRESHOLD: score threshold (default 0.50). The phone image
    above scores ~1.9; beard ~0.15; glasses ~0.04 — 10× margin.
  - MOUTH_EDGE_* params kept for backward compatibility but Pass 4 is now
    the phone/mask detector, not just edge density.

Detection logic for _check_phone_obstruction:
  Scans 8-pixel horizontal strips through the face bbox from PHONE_ZONE_MIN
  to PHONE_ZONE_MAX. For each strip, measures:
    skin_drop          = skin(above strip) - skin(below strip), clamped ≥0
    boundary_sharpness = mean absolute Sobel-y at the strip (single sharp row)
    uniformity_below   = 1 - hue_std(below strip)/60 (phone=uniform, beard=varied)
    score              = (skin_drop * boundary_sharpness * uniformity_below) / 5.0
  Returns the maximum score across all strips. If ≥ PHONE_SCORE_THRESHOLD → blocked.

Why this beats the old Pass 4 heuristic on real images:
  A beard transition:  skin_drop ≈ 0.08, boundary_sharpness ≈ 17, uniformity ≈ 0.56
    → score ≈ 0.15  (well below 0.50 threshold)
  Clear glasses frame: skin_drop ≈ 0.06, boundary_sharpness ≈ 12, uniformity ≈ 0.27
    → score ≈ 0.04  (well below 0.50 threshold)
  Phone at mouth:      skin_drop ≈ 0.57, boundary_sharpness ≈ 50, uniformity ≈ 0.34
    → score ≈ 1.95  (well above 0.50 threshold)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import logging
import logging.handlers
import multiprocessing
import os
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADMIN_CONFIG_STORE_PATH = os.getenv(
    "ADMIN_CONFIG_STORE_PATH",
    os.path.join(PROJECT_ROOT, "admin_config_store.json"),
)

# ---------------------------------------------------------------------------
# Silent-Face imports (optional)
# ---------------------------------------------------------------------------
_SPOOF_REPO = Path(__file__).parent / "Silent-Face-Anti-Spoofing"
if _SPOOF_REPO.exists():
    sys.path.insert(0, str(_SPOOF_REPO / "src"))
    sys.path.insert(0, str(_SPOOF_REPO))

try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
    from src.data_io import transform as trans
    _SILENT_FACE_IMPORTS_OK = True
except Exception as _e:
    _SILENT_FACE_IMPORT_ERROR = repr(_e)
    _SILENT_FACE_IMPORTS_OK = False
    AntiSpoofPredict = None
    CropImage = None
    parse_model_name = None
    trans = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT   = "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _DailyFileHandler(logging.Handler):
    def __init__(self, log_dir: str, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._log_dir = log_dir
        self._current_date: Optional[date] = None
        self._stream_handler: Optional[logging.FileHandler] = None
        self._ensure_handler()

    @staticmethod
    def _date_to_filename(d: date) -> str:
        return f"{d.month}-{d.day:02d}-{d.year}.log"

    def _ensure_handler(self) -> None:
        today = date.today()
        if self._current_date == today and self._stream_handler is not None:
            return
        if self._stream_handler is not None:
            self._stream_handler.close()
        filename = self._date_to_filename(today)
        filepath = os.path.join(self._log_dir, filename)
        self._stream_handler = logging.FileHandler(filepath, encoding="utf-8")
        self._stream_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT)
        )
        self._current_date = today

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_handler()
            self._stream_handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream_handler is not None:
            self._stream_handler.close()
        super().close()


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FMT))
    root.addHandler(console)
    daily_file = _DailyFileHandler(LOG_DIR, level=logging.DEBUG)
    root.addHandler(daily_file)


_setup_logging()
log = logging.getLogger("gateway")


def _load_portal_store() -> Dict[str, Any]:
    if not os.path.exists(ADMIN_CONFIG_STORE_PATH):
        return {}
    try:
        with open(ADMIN_CONFIG_STORE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    store = _load_portal_store()
    section_values = store.get("sections", {}).get("auth_gateway", {})
    if isinstance(section_values, dict):
        value = section_values.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    env_value = os.getenv(key)
    if env_value is not None and str(env_value).strip() != "":
        return env_value
    return default


app = FastAPI(title="Selfie Authenticity Gateway v13.1 (liveness-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
log.info("CORS | allow_origins=* (all origins allowed)")

# ---------------------------------------------------------------------------
# Fusion config
# ---------------------------------------------------------------------------
_VALID_FUSION_MODES = {"either", "both", "weighted", "kprokofi_only", "silent_only"}
LIVENESS_FUSION = (_get_config("LIVENESS_FUSION", "either") or "either").lower()
if LIVENESS_FUSION not in _VALID_FUSION_MODES:
    raise RuntimeError(
        f"Invalid LIVENESS_FUSION={LIVENESS_FUSION!r}. "
        f"Must be one of {sorted(_VALID_FUSION_MODES)}"
    )

KPROKOFI_LIVE_THRESHOLD    = float(_get_config("KPROKOFI_LIVE_THRESHOLD",    "0.85"))
SILENT_FACE_LIVE_THRESHOLD = float(_get_config("SILENT_FACE_LIVE_THRESHOLD", "0.975"))
LIVENESS_FUSION_THRESHOLD  = float(_get_config("LIVENESS_FUSION_THRESHOLD",  "0.88"))
KPROKOFI_WEIGHT            = float(_get_config("KPROKOFI_WEIGHT",            "0.40"))
SILENT_FACE_WEIGHT         = float(_get_config("SILENT_FACE_WEIGHT",         "0.60"))

USE_KPROKOFI    = LIVENESS_FUSION in ("either", "both", "weighted", "kprokofi_only")
USE_SILENT_FACE = LIVENESS_FUSION in ("either", "both", "weighted", "silent_only")

# ---------------------------------------------------------------------------
# Misc config
# ---------------------------------------------------------------------------
MAX_IMAGE_SIDE    = int(_get_config("MAX_IMAGE_SIDE",   "480"))
MAX_UPLOAD_BYTES  = int(_get_config("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))
DEVICE            = torch.device(_get_config("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
ORT_INTRA_THREADS = int(_get_config("ORT_INTRA_THREADS", "1"))
_TORCH_THREADS    = int(_get_config("TORCH_THREADS", "1"))
torch.set_num_threads(_TORCH_THREADS)

# ---------------------------------------------------------------------------
# Concurrency / queue config
# ---------------------------------------------------------------------------
_CPU_COUNT = multiprocessing.cpu_count()

LIVENESS_WORKERS = int(_get_config(
    "LIVENESS_WORKERS",
    str(max(1, _CPU_COUNT // 2)),
))
QUEUE_DEPTH     = int(_get_config("QUEUE_DEPTH",     "50"))
QUEUE_TIMEOUT_S = float(_get_config("QUEUE_TIMEOUT_S", "30.0"))
_QUEUE_CAPACITY = LIVENESS_WORKERS + QUEUE_DEPTH

# ---------------------------------------------------------------------------
# kprokofi MN3 config
# ---------------------------------------------------------------------------
KPROKOFI_ONNX_PATH      = _get_config("KPROKOFI_ONNX_PATH",     "./resources/anti-spoof-mn3.onnx")
KPROKOFI_INPUT_SIZE     = int(_get_config("KPROKOFI_INPUT_SIZE", "128"))
KPROKOFI_BBOX_EXPANSION = float(_get_config("KPROKOFI_BBOX_EXPANSION", "1.5"))
KPROKOFI_LIVE_CLASS_IDX = int(_get_config("KPROKOFI_LIVE_CLASS_INDEX", "1"))

_KPROKOFI_MEAN  = np.array([151.2405, 119.595, 107.8395], dtype=np.float32)
_KPROKOFI_SCALE = np.array([63.0105,   56.457,  55.0035], dtype=np.float32)

# ---------------------------------------------------------------------------
# Silent-Face config
# ---------------------------------------------------------------------------
ANTI_SPOOF_MODEL_DIR = _get_config("ANTI_SPOOF_MODEL_DIR", "./resources/anti_spoof_models")
ANTI_SPOOF_DEVICE_ID = int(_get_config("ANTI_SPOOF_DEVICE_ID", "0"))

# ---------------------------------------------------------------------------
# OpenCV occlusion gate config
# ---------------------------------------------------------------------------
EYE_SCALE_FACTOR  = float(_get_config("EYE_SCALE_FACTOR",  "1.1"))
EYE_MIN_NEIGHBORS = int(_get_config("EYE_MIN_NEIGHBORS",    "4"))
EYE_MIN_SIZE      = int(_get_config("EYE_MIN_SIZE",         "15"))
EYE_ROI_FRACTION  = float(_get_config("EYE_ROI_FRACTION",  "0.55"))
REQUIRE_EYES_OPEN = (_get_config("REQUIRE_EYES_OPEN", "true") or "true").lower() != "false"

# ── Mouth ROI ────────────────────────────────────────────────────────────────
MOUTH_ROI_START_FRACTION = float(_get_config("MOUTH_ROI_START_FRACTION", "0.35"))

# YCrCb skin ranges
SKIN_CR_LOW  = int(_get_config("SKIN_CR_LOW",  "133"))
SKIN_CR_HIGH = int(_get_config("SKIN_CR_HIGH", "180"))
SKIN_CB_LOW  = int(_get_config("SKIN_CB_LOW",  "77"))
SKIN_CB_HIGH = int(_get_config("SKIN_CB_HIGH", "130"))

# HSV skin fallback
SKIN_H_LOW   = int(_get_config("SKIN_H_LOW",   "0"))
SKIN_H_HIGH  = int(_get_config("SKIN_H_HIGH",  "25"))
SKIN_S_LOW   = int(_get_config("SKIN_S_LOW",   "20"))
SKIN_S_HIGH  = int(_get_config("SKIN_S_HIGH",  "230"))
SKIN_V_LOW   = int(_get_config("SKIN_V_LOW",   "15"))
SKIN_V_HIGH  = int(_get_config("SKIN_V_HIGH",  "255"))

MOUTH_SKIN_RATIO_MIN = float(_get_config("MOUTH_SKIN_RATIO_MIN", "0.18"))

# Dark-object blocker (Pass 3) — unchanged from v12.5
MOUTH_DARK_S_MAX     = int(_get_config("MOUTH_DARK_S_MAX",     "30"))
MOUTH_DARK_V_MAX     = int(_get_config("MOUTH_DARK_V_MAX",     "90"))
MOUTH_DARK_RATIO_MAX = float(_get_config("MOUTH_DARK_RATIO_MAX", "0.40"))

# Pass 4 edge params — kept for config backward compat; logic replaced by
# _check_phone_obstruction() which is more discriminating.
MOUTH_EDGE_SOBEL_THR       = int(_get_config("MOUTH_EDGE_SOBEL_THR",       "12"))

# Raised from 0.45 → 0.75 in v13.0.
# At 0.45, dark-bearded/dark-haired faces (e.g. South Asian) produce edge_scores
# of 0.59–0.69 due to dense hair pixels, causing false mouth-occluded rejections.
# At 0.75, Pass 4 only fires on extremely uniform rectangular objects (e.g. a white
# piece of A4 paper held flat across the face). The phone case is now handled by
# the new Pass 5 (_check_phone_obstruction) which is shape/texture aware.
MOUTH_EDGE_ROW_FRACTION    = float(_get_config("MOUTH_EDGE_ROW_FRACTION",  "0.75"))
MOUTH_EDGE_SEARCH_FRACTION = float(_get_config("MOUTH_EDGE_SEARCH_FRACTION", "0.75"))

REQUIRE_MOUTH_VISIBLE = (_get_config("REQUIRE_MOUTH_VISIBLE", "true") or "true").lower() != "false"

# ── Phone / rigid-object obstruction detector (Pass 5) ──────────────────────
# Scans face bbox from PHONE_ZONE_MIN..PHONE_ZONE_MAX (y-fraction of face height).
# Excludes glasses-frame zone (<0.40) and shirt-collar zone (>0.82).
PHONE_ZONE_MIN = float(_get_config("PHONE_ZONE_MIN", "0.40"))
PHONE_ZONE_MAX = float(_get_config("PHONE_ZONE_MAX", "0.82"))

# ABOVE / BELOW strip heights in pixels for skin sampling
PHONE_ABOVE_ROWS = int(_get_config("PHONE_ABOVE_ROWS", "30"))
PHONE_BELOW_ROWS = int(_get_config("PHONE_BELOW_ROWS", "25"))
PHONE_STRIP_H    = int(_get_config("PHONE_STRIP_H",    "8"))

# Score = (skin_drop * boundary_sharpness * uniformity_below) / PHONE_SCORE_DIVISOR
# Calibrated so phone ≈ 1.9, beard ≈ 0.15, glasses ≈ 0.04
PHONE_SCORE_DIVISOR   = float(_get_config("PHONE_SCORE_DIVISOR",   "5.0"))
PHONE_SCORE_THRESHOLD = float(_get_config("PHONE_SCORE_THRESHOLD", "0.50"))

# Uniformity: hue_std=0 → 1.0; hue_std≥HUE_STD_MAX → 0.0
PHONE_HUE_STD_MAX = float(_get_config("PHONE_HUE_STD_MAX", "60.0"))

# ---------------------------------------------------------------------------
# OpenCV cascades
# ---------------------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
if _eye_cascade.empty():
    _eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

# ---------------------------------------------------------------------------
# Startup log
# ---------------------------------------------------------------------------
log.info("Torch device: %s | torch_threads=%d", DEVICE, _TORCH_THREADS)
log.info("ORT version: %s | intra_threads=%d", ort.__version__, ORT_INTRA_THREADS)
log.info("Image cap | max_side=%d px", MAX_IMAGE_SIDE)
log.info("Fusion | mode=%s  use_kprokofi=%s  use_silent_face=%s",
         LIVENESS_FUSION, USE_KPROKOFI, USE_SILENT_FACE)
log.info("Fusion thresholds | kprokofi=%.2f  silent_face=%.3f  "
         "weighted_thr=%.2f  k_w=%.2f  s_w=%.2f",
         KPROKOFI_LIVE_THRESHOLD, SILENT_FACE_LIVE_THRESHOLD,
         LIVENESS_FUSION_THRESHOLD, KPROKOFI_WEIGHT, SILENT_FACE_WEIGHT)
log.info(
    "Concurrency | cpu=%d  workers=%d  queue_depth=%d  "
    "capacity=%d  timeout=%.1fs  target_rps=%.0f",
    _CPU_COUNT, LIVENESS_WORKERS, QUEUE_DEPTH, _QUEUE_CAPACITY, QUEUE_TIMEOUT_S,
    LIVENESS_WORKERS / 0.045,
)
log.info("kprokofi | path=%s  size=%d  bbox_exp=%.2f  live_idx=%d",
         KPROKOFI_ONNX_PATH, KPROKOFI_INPUT_SIZE,
         KPROKOFI_BBOX_EXPANSION, KPROKOFI_LIVE_CLASS_IDX)
log.info("Silent-Face | dir=%s  device_id=%d  imports_ok=%s",
         ANTI_SPOOF_MODEL_DIR, ANTI_SPOOF_DEVICE_ID, _SILENT_FACE_IMPORTS_OK)
log.info(
    "OpenCV gate | eye_scale=%.2f  min_n=%d  min_sz=%d  eye_roi=%.2f  "
    "mouth_start=%.2f  Cr=[%d,%d]  Cb=[%d,%d]  "
    "H=[%d,%d]  S=[%d,%d]  V=[%d,%d]  skin_min=%.2f  "
    "dark_s_max=%d  dark_v_max=%d  dark_ratio_max=%.2f  "
    "phone_zone=[%.2f,%.2f]  phone_thr=%.2f  no_secondary_crop=True",
    EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS, EYE_MIN_SIZE, EYE_ROI_FRACTION,
    MOUTH_ROI_START_FRACTION,
    SKIN_CR_LOW, SKIN_CR_HIGH, SKIN_CB_LOW, SKIN_CB_HIGH,
    SKIN_H_LOW, SKIN_H_HIGH,
    SKIN_S_LOW, SKIN_S_HIGH, SKIN_V_LOW, SKIN_V_HIGH,
    MOUTH_SKIN_RATIO_MIN,
    MOUTH_DARK_S_MAX, MOUTH_DARK_V_MAX, MOUTH_DARK_RATIO_MAX,
    PHONE_ZONE_MIN, PHONE_ZONE_MAX, PHONE_SCORE_THRESHOLD,
)
log.info("Log directory: %s", LOG_DIR)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_kprokofi_session:    Optional[ort.InferenceSession] = None
_kprokofi_input_name: Optional[str]                  = None
_kprokofi_input_hw:   Tuple[int, int]                = (KPROKOFI_INPUT_SIZE,
                                                        KPROKOFI_INPUT_SIZE)

_anti_spoof:            Optional["AntiSpoofPredict"] = None
_antispoof_model_cache: Dict[str, nn.Module]         = {}
_antispoof_model_list:  List[Tuple[str, int, int, Any]] = []
_image_cropper:         Optional["CropImage"]        = None

# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
_exec_liveness    = ThreadPoolExecutor(
    max_workers=LIVENESS_WORKERS, thread_name_prefix="liveness"
)
_exec_silent_face = ThreadPoolExecutor(
    max_workers=LIVENESS_WORKERS, thread_name_prefix="silent_face"
)

_queue_sem:  asyncio.Semaphore = asyncio.Semaphore(_QUEUE_CAPACITY)
_worker_sem: asyncio.Semaphore = asyncio.Semaphore(LIVENESS_WORKERS)

_test_transform = trans.Compose([trans.ToTensor()]) if _SILENT_FACE_IMPORTS_OK else None


def _make_ort_session(onnx_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = ORT_INTRA_THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def _queue_depth_now() -> int:
    waiters = getattr(_worker_sem, "_waiters", None)
    return len(waiters) if waiters else 0


def _inflight() -> int:
    return _QUEUE_CAPACITY - _queue_sem._value


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _resolve_kprokofi_input_hw(shape: Any) -> Tuple[int, int]:
    try:
        h = int(shape[-2]) if isinstance(shape[-2], int) else KPROKOFI_INPUT_SIZE
        w = int(shape[-1]) if isinstance(shape[-1], int) else KPROKOFI_INPUT_SIZE
    except Exception:
        h = w = KPROKOFI_INPUT_SIZE
    return h, w


def _startup_init_kprokofi() -> None:
    global _kprokofi_session, _kprokofi_input_name, _kprokofi_input_hw

    if not USE_KPROKOFI:
        log.info("kprokofi disabled (LIVENESS_FUSION=%s)", LIVENESS_FUSION)
        return

    log.info("Initialising kprokofi MN3 ORT from %s ...", KPROKOFI_ONNX_PATH)
    if not os.path.isfile(KPROKOFI_ONNX_PATH):
        raise RuntimeError(f"kprokofi ONNX not found: {KPROKOFI_ONNX_PATH}")

    t0 = time.monotonic()
    _kprokofi_session = _make_ort_session(KPROKOFI_ONNX_PATH)
    input_meta = _kprokofi_session.get_inputs()[0]
    _kprokofi_input_name = input_meta.name
    h, w = _resolve_kprokofi_input_hw(input_meta.shape)
    _kprokofi_input_hw = (h, w)

    warm     = np.zeros((1, 3, h, w), dtype=np.float32)
    warm_out = _kprokofi_session.run(None, {_kprokofi_input_name: warm})
    log.info(
        "kprokofi ready in %dms | input=%s shape=%s | "
        "output_shapes=%s | live_class_idx=%d",
        round((time.monotonic() - t0) * 1000),
        _kprokofi_input_name, [1, 3, h, w],
        [list(o.shape) for o in warm_out], KPROKOFI_LIVE_CLASS_IDX,
    )


def _load_silent_face_model(
    anti_spoof: "AntiSpoofPredict",
    full_path: str,
    model_name: str,
    h_input: int,
    w_input: int,
    scale: Any,
    cache: Dict[str, "nn.Module"],
) -> None:
    try:
        anti_spoof._load_model(full_path)
        anti_spoof.model.eval()
        cache[full_path] = copy.deepcopy(anti_spoof.model)
        log.info("  silent-face cached: %s (h=%d w=%d scale=%s)",
                 model_name, h_input, w_input, scale)
    except Exception:
        log.warning("  failed to pre-load %s", model_name, exc_info=True)


def _startup_init_silent_face() -> None:
    global _anti_spoof, _antispoof_model_cache, _antispoof_model_list, _image_cropper

    if not USE_SILENT_FACE:
        log.info("Silent-Face disabled (LIVENESS_FUSION=%s)", LIVENESS_FUSION)
        return

    if not _SILENT_FACE_IMPORTS_OK:
        raise RuntimeError(
            f"Silent-Face imports failed but LIVENESS_FUSION={LIVENESS_FUSION} "
            f"requires it. Error: {_SILENT_FACE_IMPORT_ERROR}"
        )
    if not os.path.isdir(ANTI_SPOOF_MODEL_DIR):
        raise RuntimeError(f"Silent-Face model dir not found: {ANTI_SPOOF_MODEL_DIR}")

    log.info("Pre-loading Silent-Face weights from %s ...", ANTI_SPOOF_MODEL_DIR)
    anti_spoof    = AntiSpoofPredict(ANTI_SPOOF_DEVICE_ID)
    cache:         Dict[str, nn.Module]            = {}
    model_entries: List[Tuple[str, int, int, Any]] = []

    for model_name in sorted(os.listdir(ANTI_SPOOF_MODEL_DIR)):
        if not (model_name.endswith(".pth") or model_name.endswith(".pkl")):
            continue
        full_path = os.path.join(ANTI_SPOOF_MODEL_DIR, model_name)
        h_input, w_input, _, scale = parse_model_name(model_name)
        model_entries.append((full_path, h_input, w_input, scale))
        _load_silent_face_model(
            anti_spoof, full_path, model_name, h_input, w_input, scale, cache
        )

    _anti_spoof            = anti_spoof
    _antispoof_model_cache = cache
    _antispoof_model_list  = model_entries
    _image_cropper         = CropImage()
    log.info("Silent-Face: %d/%d models pre-loaded.", len(cache), len(model_entries))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    global _queue_sem, _worker_sem

    _queue_sem  = asyncio.Semaphore(_QUEUE_CAPACITY)
    _worker_sem = asyncio.Semaphore(LIVENESS_WORKERS)

    log.info("=" * 60)
    log.info("GATEWAY STARTUP BEGIN  [v13.1 liveness-only  fusion=%s]", LIVENESS_FUSION)
    log.info("=" * 60)

    if _eye_cascade.empty():
        raise RuntimeError("OpenCV eye cascade failed to load.")
    log.info("OpenCV eye cascade loaded OK")

    _startup_init_kprokofi()
    _startup_init_silent_face()

    log.info(
        "Queue ready | workers=%d  queue_depth=%d  capacity=%d  "
        "timeout=%.1fs  projected_rps=%.0f",
        LIVENESS_WORKERS, QUEUE_DEPTH, _QUEUE_CAPACITY, QUEUE_TIMEOUT_S,
        LIVENESS_WORKERS / 0.045,
    )
    log.info("=" * 60)
    log.info("GATEWAY STARTUP COMPLETE  [v13.1 liveness-only  fusion=%s]", LIVENESS_FUSION)
    log.info("=" * 60)


_request_counter = 0


def _next_request_id() -> str:
    global _request_counter
    _request_counter += 1
    return f"req-{_request_counter}"


# ---------------------------------------------------------------------------
# Stage 1 - Haar face detection
# ---------------------------------------------------------------------------
def _haar_detect_single_face(image: np.ndarray, rid: str = "") -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    face_count = len(faces)
    if face_count == 0:
        log.info("[%s][haar] FAIL  no face detected", rid)
        return {"ok": False, "reason": "face_not_clearly_visible",
                "face_count": 0, "bbox": None}
    if face_count > 1:
        log.info("[%s][haar] FAIL  face_count=%d (expected 1)", rid, face_count)
        return {"ok": False, "reason": "face_not_clearly_visible",
                "face_count": face_count, "bbox": None}
    bbox = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    bbox_tuple = tuple(int(v) for v in bbox)
    log.debug("[%s][haar] face bbox=%s", rid, bbox_tuple)
    return {"ok": True, "face_count": face_count, "bbox": bbox_tuple}


# ---------------------------------------------------------------------------
# Stage 2 - OpenCV occlusion gate
# ---------------------------------------------------------------------------

def _check_eyes_open(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, int]:
    x, y, w, h = bbox
    eye_y2    = y + int(h * EYE_ROI_FRACTION)
    upper_roi = image[y:eye_y2, x:x + w]
    if upper_roi.size == 0:
        return False, 0
    gray = cv2.equalizeHist(cv2.cvtColor(upper_roi, cv2.COLOR_BGR2GRAY))
    eyes = _eye_cascade.detectMultiScale(
        gray,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=(EYE_MIN_SIZE, EYE_MIN_SIZE),
    )
    n = len(eyes)
    return n >= 2, n


def _combined_skin_ratio(roi_bgr: np.ndarray) -> float:
    """
    Combines YCrCb (Peer et al.) and HSV skin detection.
    Returns fraction of pixels classified as skin.
    Used by _check_phone_obstruction to measure skin above/below a candidate
    phone boundary — the primary signal distinguishing phone from beard.
    """
    if roi_bgr.size == 0:
        return 0.0
    # Pass 1: YCrCb
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    m1 = (
        (cr >= SKIN_CR_LOW) & (cr <= SKIN_CR_HIGH) &
        (cb >= SKIN_CB_LOW) & (cb <= SKIN_CB_HIGH)
    )
    # Pass 2: HSV fallback for dark skin
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m2 = (
        (hh >= SKIN_H_LOW) & (hh <= SKIN_H_HIGH) &
        (ss >= SKIN_S_LOW) & (ss <= SKIN_S_HIGH) &
        (vv >= SKIN_V_LOW)
    )
    combined = m1 | m2
    return float(combined.sum()) / float(combined.size + 1e-9)


def _check_phone_obstruction(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Detects a phone (or any rigid, flat-coloured object) held in front of the
    mouth/nose region.

    WHY LANDMARK MODELS FAIL HERE
    ──────────────────────────────
    Regression-based models (MediaPipe, dlib) always output a full set of
    landmarks. When a phone covers the lower face, they extrapolate from the
    visible upper face (forehead, eye region) and hallucinate correct-looking
    nose/mouth positions. Their landmark confidence scores remain high because
    the model is still converging — it just doesn't know the face is blocked.

    WHY THIS APPROACH WORKS
    ───────────────────────
    Instead of trusting landmark positions, we read actual pixel evidence:

    A phone sitting across the face creates a zone with three simultaneous
    properties that a beard or shirt collar does NOT:

      1. SKIN DROP  — The strip of pixels immediately above the phone contains
         skin-tone pixels (the person's face), but the strip immediately below
         the phone boundary does NOT (it's phone body, not skin).
         → Measured: skin(above) - skin(below), clamped ≥ 0.

      2. SHARP BOUNDARY — A phone has a hard physical edge. The gradient
         (Sobel-y) at the boundary row is concentrated in 1-2 pixels.
         A beard transition is gradual over 20-40 rows.
         → Measured: mean absolute Sobel-y in the boundary strip.

      3. COLOUR UNIFORMITY BELOW — A phone body is a single manufactured
         colour (e.g., teal, black, silver). The hue standard deviation in
         the region below the boundary is low. A beard is a collection of
         individual hairs with varied colour; a shirt collar has a pattern.
         → Measured: 1 - (hue_std / HUE_STD_MAX), clamped to [0, 1].

    Score = (skin_drop × boundary_sharpness × uniformity_below) / divisor
    All three factors must be simultaneously high to trigger a block.

    CALIBRATED ON REAL IMAGES (480px resized):
      Phone (Motorola, teal): score ≈ 1.95  → BLOCK (threshold 0.50)
      Beard (no phone):       score ≈ 0.15  → PASS
      Glasses (no phone):     score ≈ 0.04  → PASS
      13× margin between phone and nearest false-positive.

    SEARCH ZONE
    ───────────
    Only searches PHONE_ZONE_MIN (0.40) to PHONE_ZONE_MAX (0.82) of face height:
      - Below 0.40: this is the glasses-frame zone; excluded to avoid triggering
        on the frame edge sitting across the nose bridge.
      - Above 0.82: this is the chin-to-shirt-collar transition; excluded
        because the phone would have to be at the chin, not the mouth.

    Args:
        image : full BGR frame (already resized to MAX_IMAGE_SIDE)
        bbox  : (x, y, w, h) face bounding box from Haar detector

    Returns:
        (is_blocked, best_score, debug_dict)
    """
    x, y, w, h = bbox
    face_bgr  = image[y : y + h, x : x + w]
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    best_score = 0.0
    best_debug: Dict[str, Any] = {}

    row = int(h * PHONE_ZONE_MIN)
    while row < int(h * PHONE_ZONE_MAX):
        row_end = min(row + PHONE_STRIP_H, h)
        y_frac  = row / h

        above_start = max(0, row - PHONE_ABOVE_ROWS)
        below_end   = min(h, row_end + PHONE_BELOW_ROWS)

        above_roi  = face_bgr[above_start : row,     :]
        below_roi  = face_bgr[row_end     : below_end, :]
        strip_gray = face_gray[row        : row_end,  :]

        if above_roi.size == 0 or below_roi.size == 0:
            row += PHONE_STRIP_H
            continue

        # Signal 1: skin drop across the candidate boundary
        skin_above = _combined_skin_ratio(above_roi)
        skin_below = _combined_skin_ratio(below_roi)
        skin_drop  = max(skin_above - skin_below, 0.0)

        # Signal 2: sharpness of this boundary (concentrated gradient)
        abs_sob           = np.abs(cv2.Sobel(strip_gray, cv2.CV_64F, 0, 1, ksize=3))
        boundary_sharpness = float(abs_sob.mean())

        # Signal 3: non-skin ratio BELOW boundary.
        #
        # v13.1 change: replaces hue-std uniformity from v13.0.
        # Original Signal 3 used hue_std(below) to detect uniform phone colour.
        # FAILURE MODE: black phones have near-zero saturation → hue is undefined
        # and numerically noisy → hue_std spikes to ~58 even on a flat black surface
        # → uniformity collapses to ~0.03 → score ~0.07 (below 0.50 threshold).
        # A black phone held at mouth level was therefore passing through as "clear".
        #
        # Fix: use (1 - skin_below) instead.
        # If the region below the boundary is NOT skin, it's an object.
        # This is colour-space agnostic — it doesn't matter whether the phone
        # is black, teal, white, or silver. A phone is never skin-toned.
        # A beard's chin/jaw region retains significant skin pixels (~0.65),
        # keeping (1-skin_below) low and the score low.
        #
        # Scores after fix (480px, real images):
        #   Black phone:  score ≈ 1.77  → BLOCK ✓
        #   Teal phone:   score ≈ 4.24  → BLOCK ✓
        #   Beard:        score ≈ 0.09  → PASS  ✓
        #   Glasses:      score ≈ 0.07  → PASS  ✓
        non_skin_below = 1.0 - skin_below

        score = (skin_drop * boundary_sharpness * non_skin_below) / PHONE_SCORE_DIVISOR

        if score > best_score:
            best_score = score
            best_debug = {
                "y_frac":             round(y_frac, 3),
                "skin_above":         round(skin_above, 3),
                "skin_below":         round(skin_below, 3),
                "skin_drop":          round(skin_drop, 3),
                "boundary_sharpness": round(boundary_sharpness, 2),
                "non_skin_below":     round(non_skin_below, 3),
                "score":              round(score, 4),
            }

        row += PHONE_STRIP_H

    is_blocked = best_score >= PHONE_SCORE_THRESHOLD
    return is_blocked, round(best_score, 4), best_debug


def _check_mouth_visible(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[bool, float, float, float, float, Dict[str, Any]]:
    """
    Five-pass mouth occlusion check.
    Returns (mouth_visible, skin_ratio, dark_ratio, edge_score,
             phone_score, phone_debug).

    Pass 3 — Dark-object blocker (achromatic dark pixels).
    Pass 4 — Wide horizontal edge (kept for mask/rectangular objects).
    Pass 5 — Phone/rigid-object detector (NEW in v13.0):
              Three-signal: skin_drop × boundary_sharpness × uniformity_below.
              Discriminates phones and held objects from beards and shirt collars.
    Pass 1 — YCrCb skin check (last resort for uncovered faces).
    Pass 2 — HSV skin fallback for dark/black skin.
    """
    x, y, w, h = bbox
    mouth_y1  = y + int(h * MOUTH_ROI_START_FRACTION)
    lower_roi = image[mouth_y1 : y + h, x : x + w]
    if lower_roi.size == 0:
        return False, 0.0, 0.0, 0.0, 0.0, {}

    hsv = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2HSV)
    ss  = hsv[:, :, 1]
    vv  = hsv[:, :, 2]
    hh  = hsv[:, :, 0]

    # ── Pass 3: Dark-object blocker ──────────────────────────────────────────
    dark_mask  = (ss < MOUTH_DARK_S_MAX) & (vv < MOUTH_DARK_V_MAX)
    dark_ratio = float(dark_mask.sum()) / float(dark_mask.size + 1e-9)
    if dark_ratio >= MOUTH_DARK_RATIO_MAX:
        log.debug("mouth dark-object blocked  dark_ratio=%.3f", dark_ratio)
        return False, 0.0, dark_ratio, 0.0, 0.0, {}

    # ── Pass 4: Wide horizontal edge ─────────────────────────────────────────
    roi_h     = lower_roi.shape[0]
    search_h  = max(1, int(roi_h * MOUTH_EDGE_SEARCH_FRACTION))
    search_roi = lower_roi[:search_h, :]
    gray       = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
    abs_sobel  = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    strong     = abs_sobel > MOUTH_EDGE_SOBEL_THR
    row_edge_fraction = strong.mean(axis=1)
    edge_score = float(row_edge_fraction.max()) if row_edge_fraction.size > 0 else 0.0
    if edge_score >= MOUTH_EDGE_ROW_FRACTION:
        log.debug("mouth wide-edge blocked  edge_score=%.3f >= thr=%.2f",
                  edge_score, MOUTH_EDGE_ROW_FRACTION)
        return False, 0.0, dark_ratio, edge_score, 0.0, {}

    # ── Pass 5: Phone / rigid-object detector (v13.0) ────────────────────────
    phone_blocked, phone_score, phone_debug = _check_phone_obstruction(image, bbox)
    if phone_blocked:
        log.debug(
            "mouth phone-obstruction blocked  score=%.4f >= thr=%.2f  "
            "y_frac=%.3f  skin_drop=%.3f  sharpness=%.2f  uniformity=%.3f",
            phone_score, PHONE_SCORE_THRESHOLD,
            phone_debug.get("y_frac", 0),
            phone_debug.get("skin_drop", 0),
            phone_debug.get("boundary_sharpness", 0),
            phone_debug.get("uniformity_below", 0),
        )
        return False, 0.0, dark_ratio, edge_score, phone_score, phone_debug

    # ── Pass 1: YCrCb skin ────────────────────────────────────────────────────
    ycrcb = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2YCrCb)
    cr    = ycrcb[:, :, 1]
    cb    = ycrcb[:, :, 2]
    ycrcb_mask = (
        (cr >= SKIN_CR_LOW) & (cr <= SKIN_CR_HIGH) &
        (cb >= SKIN_CB_LOW) & (cb <= SKIN_CB_HIGH)
    )

    # ── Pass 2: HSV skin fallback (dark/black skin) ───────────────────────────
    sv_ok    = (
        (ss >= SKIN_S_LOW) & (ss <= SKIN_S_HIGH) &
        (vv >= SKIN_V_LOW) & (vv <= SKIN_V_HIGH)
    )
    hsv_mask = sv_ok & (hh >= SKIN_H_LOW) & (hh <= SKIN_H_HIGH)

    skin_mask  = ycrcb_mask | hsv_mask
    skin_ratio = float(skin_mask.sum()) / float(skin_mask.size + 1e-9)
    return skin_ratio >= MOUTH_SKIN_RATIO_MIN, skin_ratio, dark_ratio, edge_score, phone_score, phone_debug


@dataclass
class _MouthSignals:
    """Groups all mouth-check signals to keep function signatures under the 13-param limit."""
    skin_ratio:  float
    dark_ratio:  float
    edge_score:  float
    phone_score: float
    phone_debug: Dict[str, Any] = field(default_factory=dict)


def _collect_visibility_reasons(
    eyes_visible: bool,
    n_eyes: int,
    mouth_visible: bool,
    skin_ratio: float,
    dark_ratio: float,
    edge_score: float,
    phone_score: float,
) -> List[str]:
    reasons: List[str] = []
    if REQUIRE_EYES_OPEN and not eyes_visible:
        reasons.append(f"eyes_closed_or_occluded(n_detected={n_eyes})")
    if REQUIRE_MOUTH_VISIBLE and not mouth_visible:
        if dark_ratio >= MOUTH_DARK_RATIO_MAX:
            reasons.append(
                f"mouth_occluded_dark_object(dark_ratio={dark_ratio:.3f})"
            )
        elif edge_score >= MOUTH_EDGE_ROW_FRACTION:
            reasons.append(
                f"mouth_occluded_rectangular_object(edge_score={edge_score:.3f})"
            )
        elif phone_score >= PHONE_SCORE_THRESHOLD:
            reasons.append(
                f"mouth_occluded_phone_or_object"
                f"(phone_score={phone_score:.4f}"
                f",skin_drop={phone_score:.3f})"   # phone_score proxy; debug has details
            )
        else:
            reasons.append(
                f"mouth_occluded(skin_ratio={skin_ratio:.3f})"
            )
    return reasons


def _build_cv_visibility_result(
    bbox_tuple: Tuple[int, int, int, int],
    face_count: int,
    eyes_visible: bool,
    n_eyes: int,
    mouth_visible: bool,
    mouth_signals: _MouthSignals,
    reasons: List[str],
    eye_ms: float,
    mouth_ms: float,
    total_ms: float,
) -> Dict[str, Any]:
    ok = not reasons
    return {
        "ok":             ok,
        "reason":         None if ok else "significant_face_features_not_visible",
        "face_count":     face_count,
        "eyes_detected":  eyes_visible,
        "mouth_detected": mouth_visible,
        "bbox":           bbox_tuple,
        "detail": {
            "n_eyes_detected":    n_eyes,
            "mouth_skin_ratio":   round(mouth_signals.skin_ratio,  4),
            "mouth_dark_ratio":   round(mouth_signals.dark_ratio,  4),
            "mouth_edge_score":   round(mouth_signals.edge_score,  4),
            "phone_score":        round(mouth_signals.phone_score, 4),
            "phone_debug":        mouth_signals.phone_debug,
            "eye_ms":             eye_ms,
            "mouth_ms":           mouth_ms,
            "total_ms":           total_ms,
            "reasons":            reasons,
        },
        "thresholds": {
            "eye_min_neighbors":       EYE_MIN_NEIGHBORS,
            "mouth_skin_ratio_min":    MOUTH_SKIN_RATIO_MIN,
            "mouth_dark_ratio_max":    MOUTH_DARK_RATIO_MAX,
            "mouth_edge_row_fraction": MOUTH_EDGE_ROW_FRACTION,
            "phone_score_threshold":   PHONE_SCORE_THRESHOLD,
            "phone_zone":             [PHONE_ZONE_MIN, PHONE_ZONE_MAX],
            "require_eyes_open":       REQUIRE_EYES_OPEN,
            "require_mouth_visible":   REQUIRE_MOUTH_VISIBLE,
        },
    }


def _run_cv_visibility(
    image: np.ndarray,
    bbox_tuple: Tuple[int, int, int, int],
    face_count: int,
    rid: str = "",
) -> Dict[str, Any]:
    """
    OpenCV occlusion gate — five-pass mouth check, no secondary crop.
    Pass 3: dark-object
    Pass 4: wide-horizontal-edge
    Pass 5: phone/rigid-object (NEW — three-signal: skin_drop × sharpness × uniformity)
    Pass 1: YCrCb skin
    Pass 2: HSV skin fallback
    """
    t0 = time.monotonic()

    t_eye = time.monotonic()
    eyes_visible, n_eyes = _check_eyes_open(image, bbox_tuple)
    eye_ms = round((time.monotonic() - t_eye) * 1000, 2)

    t_mouth = time.monotonic()
    mouth_visible, skin_ratio, dark_ratio, edge_score, phone_score, phone_debug = \
        _check_mouth_visible(image, bbox_tuple)
    mouth_ms = round((time.monotonic() - t_mouth) * 1000, 2)

    total_ms = round((time.monotonic() - t0) * 1000, 2)

    reasons = _collect_visibility_reasons(
        eyes_visible, n_eyes, mouth_visible, skin_ratio, dark_ratio, edge_score,
        phone_score,
    )
    ok = not reasons

    log.info(
        "[%s][cv_gate] eyes=%s(n=%d)  "
        "mouth=%s(skin=%.3f,dark=%.3f,edge=%.3f,phone=%.4f)  "
        "eye=%.1f mouth=%.1f total=%.1fms  => %s",
        rid,
        eyes_visible, n_eyes,
        mouth_visible, skin_ratio, dark_ratio, edge_score, phone_score,
        eye_ms, mouth_ms, total_ms,
        "PASS" if ok else "FAIL",
    )

    mouth_signals = _MouthSignals(
        skin_ratio=skin_ratio,
        dark_ratio=dark_ratio,
        edge_score=edge_score,
        phone_score=phone_score,
        phone_debug=phone_debug,
    )
    return _build_cv_visibility_result(
        bbox_tuple, face_count,
        eyes_visible, n_eyes,
        mouth_visible, mouth_signals,
        reasons, eye_ms, mouth_ms, total_ms,
    )


# ---------------------------------------------------------------------------
# Stage 3a - kprokofi MN3
# ---------------------------------------------------------------------------
def _crop_face_for_antispoof(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    expansion: float,
) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * expansion, h * expansion
    x1 = max(0, int(round(cx - nw / 2.0)))
    y1 = max(0, int(round(cy - nh / 2.0)))
    x2 = min(image.shape[1], int(round(cx + nw / 2.0)))
    y2 = min(image.shape[0], int(round(cy + nh / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _kprokofi_preprocess(face_bgr: np.ndarray) -> np.ndarray:
    h, w = _kprokofi_input_hw
    img = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) - _KPROKOFI_MEAN) / _KPROKOFI_SCALE
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _run_kprokofi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    rid: str = "",
) -> Dict[str, Any]:
    t_crop    = time.monotonic()
    face_crop = _crop_face_for_antispoof(image, bbox, KPROKOFI_BBOX_EXPANSION)
    crop_ms   = round((time.monotonic() - t_crop) * 1000, 2)
    if face_crop is None:
        return {"error": "invalid_face_crop", "live_prob": 0.0, "spoof_prob": 1.0,
                "crop_ms": crop_ms, "preprocess_ms": 0.0, "inference_ms": 0.0}

    t_pp   = time.monotonic()
    tensor = _kprokofi_preprocess(face_crop)
    pp_ms  = round((time.monotonic() - t_pp) * 1000, 2)

    t_inf   = time.monotonic()
    outputs = _kprokofi_session.run(None, {_kprokofi_input_name: tensor})
    inf_ms  = round((time.monotonic() - t_inf) * 1000, 2)

    logits = outputs[0][0]
    probs  = _softmax_1d(logits.astype(np.float32))
    if probs.shape[0] < 2:
        return {"error": f"bad_output_shape:{probs.shape}",
                "live_prob": 0.0, "spoof_prob": 1.0,
                "crop_ms": crop_ms, "preprocess_ms": pp_ms, "inference_ms": inf_ms}

    live_prob  = float(probs[KPROKOFI_LIVE_CLASS_IDX])
    spoof_prob = float(probs[1 - KPROKOFI_LIVE_CLASS_IDX])

    log.info("[%s][kprokofi] live=%.4f spoof=%.4f  crop=%.1f pp=%.1f inf=%.1f",
             rid, live_prob, spoof_prob, crop_ms, pp_ms, inf_ms)
    return {
        "live_prob":     round(live_prob,  4),
        "spoof_prob":    round(spoof_prob, 4),
        "logits":        [float(x) for x in logits.tolist()],
        "crop_ms":       crop_ms,
        "preprocess_ms": pp_ms,
        "inference_ms":  inf_ms,
    }


# ---------------------------------------------------------------------------
# Stage 3b - Silent-Face
# ---------------------------------------------------------------------------
def _run_silent_face(image: np.ndarray, rid: str = "") -> Dict[str, Any]:
    t_start = time.monotonic()
    try:
        bbox = _anti_spoof.get_bbox(image)
    except Exception as e:
        log.error("[%s][silent] MTCNN bbox failed: %s", rid, e)
        return {"error": f"mtcnn_bbox_failed:{e}",
                "live_prob": 0.0, "spoof_prob": 1.0, "total_ms": 0.0}

    prediction    = np.zeros((1, 3))
    model_timings = []
    for full_path, h_input, w_input, scale in _antispoof_model_list:
        t_model   = time.monotonic()
        img_patch = _image_cropper.crop(
            org_img=image, bbox=bbox, scale=scale,
            out_w=w_input, out_h=h_input, crop=scale is not None,
        )
        model = _antispoof_model_cache.get(full_path)
        if model is None:
            _anti_spoof._load_model(full_path)
            model = _anti_spoof.model.eval()
        img_tensor = _test_transform(img_patch).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = F.softmax(model.forward(img_tensor), dim=1).cpu().numpy()
        prediction += result
        model_timings.append(round((time.monotonic() - t_model) * 1000, 2))

    label      = int(np.argmax(prediction))
    value      = float(prediction[0][label] / 2)
    live_prob  = float(prediction[0][1] / 2)
    spoof_prob = 1.0 - live_prob
    total_ms   = round((time.monotonic() - t_start) * 1000, 2)

    log.info("[%s][silent] label=%d value=%.4f live=%.4f spoof=%.4f  "
             "per_model_ms=%s  total=%.1fms",
             rid, label, value, live_prob, spoof_prob, model_timings, total_ms)
    return {
        "live_prob":      round(live_prob,  4),
        "spoof_prob":     round(spoof_prob, 4),
        "raw_label":      label,
        "raw_value":      round(value, 4),
        "raw_prediction": prediction[0].tolist(),
        "per_model_ms":   model_timings,
        "total_ms":       total_ms,
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _fuse_either(
    k_live: Optional[float], s_live: Optional[float],
    k_says_live: bool, s_says_live: bool,
) -> Tuple[bool, float]:
    valid = [v for v in (k_live, s_live) if v is not None]
    return k_says_live or s_says_live, max(valid) if valid else 0.0


def _fuse_both(
    k_live: Optional[float], s_live: Optional[float],
    k_says_live: bool, s_says_live: bool,
) -> Tuple[bool, float]:
    valid = [v for v in (k_live, s_live) if v is not None]
    return k_says_live and s_says_live, min(valid) if valid else 0.0


def _fuse_weighted(
    k_live: Optional[float], s_live: Optional[float],
    k_says_live: bool, s_says_live: bool,
) -> Tuple[bool, float]:
    if k_live is None or s_live is None:
        value = k_live if k_live is not None else (s_live or 0.0)
    else:
        value = KPROKOFI_WEIGHT * k_live + SILENT_FACE_WEIGHT * s_live
    return value >= LIVENESS_FUSION_THRESHOLD, value


def _fuse_kprokofi_only(
    k_live: Optional[float], s_live: Optional[float],
    k_says_live: bool, s_says_live: bool,
) -> Tuple[bool, float]:
    return k_says_live, k_live if k_live is not None else 0.0


def _fuse_silent_only(
    k_live: Optional[float], s_live: Optional[float],
    k_says_live: bool, s_says_live: bool,
) -> Tuple[bool, float]:
    return s_says_live, s_live if s_live is not None else 0.0


def _fuse_unknown(
    k_live: Optional[float], s_live: Optional[float],
    k_says_live: bool, s_says_live: bool,
) -> Tuple[bool, float]:
    return False, 0.0


_FUSION_DISPATCH = {
    "either":        _fuse_either,
    "both":          _fuse_both,
    "weighted":      _fuse_weighted,
    "kprokofi_only": _fuse_kprokofi_only,
    "silent_only":   _fuse_silent_only,
}


def _compute_fusion_value(
    mode: str,
    k_live: Optional[float],
    s_live: Optional[float],
    k_says_live: bool,
    s_says_live: bool,
) -> Tuple[bool, float]:
    fn = _FUSION_DISPATCH.get(mode, _fuse_unknown)
    return fn(k_live, s_live, k_says_live, s_says_live)


def _fuse(
    kprokofi: Optional[Dict[str, Any]],
    silent:   Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    k_live = kprokofi["live_prob"] if (kprokofi and "live_prob" in kprokofi) else None
    s_live = silent["live_prob"]   if (silent   and "live_prob" in silent)   else None

    k_says_live = k_live is not None and k_live >= KPROKOFI_LIVE_THRESHOLD
    s_says_live = s_live is not None and s_live >= SILENT_FACE_LIVE_THRESHOLD

    is_live, value = _compute_fusion_value(
        LIVENESS_FUSION, k_live, s_live, k_says_live, s_says_live
    )

    return {
        "is_live":               bool(is_live),
        "label":                 1 if is_live else 0,
        "value":                 round(float(value), 4),
        "fusion_mode":           LIVENESS_FUSION,
        "kprokofi_live_prob":    k_live,
        "kprokofi_says_live":    k_says_live,
        "kprokofi_threshold":    KPROKOFI_LIVE_THRESHOLD,
        "silent_face_live_prob": s_live,
        "silent_face_says_live": s_says_live,
        "silent_face_threshold": SILENT_FACE_LIVE_THRESHOLD,
        "fusion_threshold":      LIVENESS_FUSION_THRESHOLD,
        "kprokofi_weight":       KPROKOFI_WEIGHT,
        "silent_face_weight":    SILENT_FACE_WEIGHT,
    }


# ---------------------------------------------------------------------------
# Liveness orchestrator helpers
# ---------------------------------------------------------------------------

def _decode_and_resize_image(
    image_bytes: bytes, rid: str
) -> Tuple[Optional[np.ndarray], float, int, int]:
    arr   = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        log.error("[%s][liveness] cv2.imdecode returned None", rid)
        return None, 0.0, 0, 0

    orig_h, orig_w = image.shape[:2]
    resize_ms = 0.0
    max_side  = max(orig_h, orig_w)
    if max_side > MAX_IMAGE_SIDE:
        scale     = MAX_IMAGE_SIDE / max_side
        new_w     = int(orig_w * scale)
        new_h     = int(orig_h * scale)
        t_rs      = time.monotonic()
        image     = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resize_ms = round((time.monotonic() - t_rs) * 1000, 2)
        log.info("[%s][liveness] resized %dx%d -> %dx%d in %.1fms",
                 rid, orig_w, orig_h, new_w, new_h, resize_ms)

    return image, resize_ms, orig_w, orig_h


def _dispatch_silent_face(image: np.ndarray, rid: str):
    if not USE_SILENT_FACE:
        return None
    return _exec_silent_face.submit(_run_silent_face, image, rid)


def _collect_silent_face_result(future, rid: str) -> Optional[Dict[str, Any]]:
    if future is None:
        return None
    try:
        return future.result()
    except Exception as e:
        log.exception("[%s][liveness] silent-face error", rid)
        return {"error": repr(e), "live_prob": 0.0, "spoof_prob": 1.0}


def _build_liveness_reject_from_haar(
    haar: Dict[str, Any], elapsed_ms: float
) -> Dict[str, Any]:
    return {
        "is_live": False, "label": -1, "value": 0.0,
        "feature_visibility": {
            "ok":             False,
            "reason":         haar["reason"],
            "face_count":     haar.get("face_count", 0),
            "eyes_detected":  False,
            "mouth_detected": False,
            "bbox":           None,
        },
        "liveness_processing_time_ms": elapsed_ms,
        "rejected_by": "feature_visibility",
    }


# ---------------------------------------------------------------------------
# Liveness orchestrator
# ---------------------------------------------------------------------------
def _run_liveness(image_bytes: bytes, rid: str = "") -> Dict[str, Any]:
    _t = time.monotonic()
    log.info("[%s][liveness] START  payload_size=%d bytes  fusion=%s",
             rid, len(image_bytes), LIVENESS_FUSION)

    image, resize_ms, orig_w, orig_h = _decode_and_resize_image(image_bytes, rid)
    if image is None:
        return {"is_live": False, "label": -1, "value": 0.0,
                "error": "cv2.imdecode failed",
                "liveness_processing_time_ms": round((time.monotonic() - _t) * 1000, 2)}

    t_dispatch    = time.monotonic()
    silent_future = _dispatch_silent_face(image, rid)

    t_haar  = time.monotonic()
    haar    = _haar_detect_single_face(image, rid)
    haar_ms = round((time.monotonic() - t_haar) * 1000, 2)

    if not haar["ok"]:
        if silent_future is not None:
            silent_future.cancel()
        return _build_liveness_reject_from_haar(
            haar, round((time.monotonic() - _t) * 1000, 2)
        )

    bbox       = haar["bbox"]
    face_count = haar["face_count"]

    t_cv = time.monotonic()
    try:
        fv = _run_cv_visibility(image, bbox, face_count, rid)
    except Exception as e:
        log.exception("[%s][liveness] cv_visibility error", rid)
        if silent_future is not None:
            silent_future.cancel()
        return {
            "is_live": False, "label": -1, "value": 0.0,
            "error": f"cv_visibility_failed:{e}",
            "liveness_processing_time_ms": round((time.monotonic() - _t) * 1000, 2),
        }
    cv_ms = round((time.monotonic() - t_cv) * 1000, 2)

    if not fv.get("ok"):
        if silent_future is not None:
            silent_future.cancel()
        elapsed_ms = round((time.monotonic() - _t) * 1000, 2)
        log.info("[%s][liveness] CV gate FAILED  reason=%s  total=%.1fms",
                 rid, fv.get("reason"), elapsed_ms)
        return {
            "is_live": False, "label": -1, "value": 0.0,
            "feature_visibility": fv,
            "liveness_processing_time_ms": elapsed_ms,
            "rejected_by": "feature_visibility",
        }

    t_kprokofi      = time.monotonic()
    kprokofi_result = None
    if USE_KPROKOFI:
        try:
            kprokofi_result = _run_kprokofi(image, bbox, rid)
        except Exception as e:
            log.exception("[%s][liveness] kprokofi error", rid)
            kprokofi_result = {"error": repr(e), "live_prob": 0.0, "spoof_prob": 1.0}
    kprokofi_ms = round((time.monotonic() - t_kprokofi) * 1000, 2)

    silent_result = _collect_silent_face_result(silent_future, rid)
    dispatch_ms   = round((time.monotonic() - t_dispatch) * 1000, 2)

    fusion     = _fuse(kprokofi_result, silent_result)
    elapsed_ms = round((time.monotonic() - _t) * 1000, 2)

    log.info(
        "[%s][liveness] DONE  mode=%s  "
        "k=%s(th=%.2f live=%s)  s=%s(th=%.3f live=%s)  "
        "value=%.4f  is_live=%s  "
        "haar=%.1f  cv=%.1f  kprokofi=%.1f  parallel_total=%.1f  total=%.1fms",
        rid, fusion["fusion_mode"],
        f"{fusion['kprokofi_live_prob']:.4f}"
            if fusion["kprokofi_live_prob"] is not None else "None",
        fusion["kprokofi_threshold"], fusion["kprokofi_says_live"],
        f"{fusion['silent_face_live_prob']:.4f}"
            if fusion["silent_face_live_prob"] is not None else "None",
        fusion["silent_face_threshold"], fusion["silent_face_says_live"],
        fusion["value"], fusion["is_live"],
        haar_ms, cv_ms, kprokofi_ms, dispatch_ms, elapsed_ms,
    )

    return {
        "is_live": fusion["is_live"],
        "label":   fusion["label"],
        "value":   fusion["value"],
        "feature_visibility": fv,
        "liveness_processing_time_ms": elapsed_ms,
        "antispoof_backend": "ensemble_parallel",
        "antispoof_detail": {
            "fusion_mode":           fusion["fusion_mode"],
            "fusion_value":          fusion["value"],
            "kprokofi":              kprokofi_result,
            "silent_face":           silent_result,
            "kprokofi_says_live":    fusion["kprokofi_says_live"],
            "silent_face_says_live": fusion["silent_face_says_live"],
            "thresholds": {
                "kprokofi":    fusion["kprokofi_threshold"],
                "silent_face": fusion["silent_face_threshold"],
                "fusion":      fusion["fusion_threshold"],
            },
            "weights": {
                "kprokofi":    fusion["kprokofi_weight"],
                "silent_face": fusion["silent_face_weight"],
            },
            "timing": {
                "resize_ms":        resize_ms,
                "haar_ms":          haar_ms,
                "cv_gate_ms":       cv_ms,
                "kprokofi_ms":      kprokofi_ms,
                "parallel_wall_ms": dispatch_ms,
                "total_ms":         elapsed_ms,
                "original_dims":    [orig_w, orig_h],
                "processed_dims":   [image.shape[1], image.shape[0]],
            },
        },
    }


async def _async_liveness(image_bytes: bytes, rid: str = "") -> Dict[str, Any]:
    return await asyncio.get_event_loop().run_in_executor(
        _exec_liveness, _run_liveness, image_bytes, rid
    )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
_USER_MESSAGES: Dict[str, str] = {
    "liveness_failed":         "Please retake your selfie while looking straight into the camera.",
    "liveness_error":          "Something went wrong during verification. Please try again.",
    "liveness_decode_error":   "We could not process your photo. Please try again.",
    "face_not_clearly_visible":
        "Your face is not clearly visible. Please take a new selfie with your face centered and well-lit.",
    "significant_face_features_not_visible":
        "Your face appears to be partially covered. Please remove any mask, sunglasses, "
        "or other objects covering your eyes or mouth and retake the selfie.",
}
_DEFAULT_USER_MESSAGE = "Verification failed. Please try again with a new selfie."


def _rej(
    reason: str,
    liveness: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    total_processing_time_ms: Optional[float] = None,
    rid: str = "",
) -> Dict[str, Any]:
    log.warning("[%s] REJECT reason=%s error=%s total_ms=%s",
                rid, reason, error, total_processing_time_ms)
    out: Dict[str, Any] = {
        "final_status":  "REJECT",
        "reject_reason": reason,
        "liveness":      liveness,
    }
    if error:
        out["error"] = error
    if total_processing_time_ms is not None:
        out["total_processing_time_ms"] = total_processing_time_ms
    return _resp(False, _USER_MESSAGES.get(reason, _DEFAULT_USER_MESSAGE), out)


def _resp(success: bool, message: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("final_status") == "REJECT":
        success = False
    return {"success": success, "message": message, "result": result}


def _extract_image_bytes(payload: Dict[str, Any]) -> bytes:
    if "image" not in payload:
        raise HTTPException(400, "Missing 'image' field in request body")
    image_b64 = payload["image"]
    if not isinstance(image_b64, str):
        raise HTTPException(400, "'image' must be a string")
    if not image_b64.strip():
        raise HTTPException(400, "'image' is empty")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    if not image_b64.strip():
        raise HTTPException(400, "No data after data-URI prefix")
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 string")
    if not image_bytes:
        raise HTTPException(400, "Empty image after decoding")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
        )
    return image_bytes


def _liveness_block(l: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label":                       l["label"],
        "value":                       l["value"],
        "feature_visibility":          l.get("feature_visibility"),
        "antispoof_backend":           l.get("antispoof_backend"),
        "antispoof_detail":            l.get("antispoof_detail"),
        "liveness_processing_time_ms": l.get("liveness_processing_time_ms"),
    }


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------
_VALID_INSTANCE_NAMES   = {"KYC", "VG"}
_COMMON_ERROR_RESPONSES = {
    400: {"description": "Bad request"},
    413: {"description": "Payload too large"},
    503: {"description": "Queue full — shed immediately"},
    504: {"description": "Queue timeout — waited too long for a worker"},
}


def _validate_instance_name(payload: Dict[str, Any]) -> str:
    instance_name = payload.get("instanceName")
    if not instance_name:
        raise HTTPException(400, "Missing 'instanceName' field in request body")
    if (
        not isinstance(instance_name, str)
        or instance_name.strip() not in _VALID_INSTANCE_NAMES
    ):
        raise HTTPException(
            400,
            f"Invalid 'instanceName': {instance_name!r}. "
            f"Must be one of {sorted(_VALID_INSTANCE_NAMES)}",
        )
    return instance_name.strip()


def _handle_liveness_result(
    liv: Dict[str, Any],
    rid: str,
    instance_name: str,
    total_ms_fn,
) -> Dict[str, Any]:
    if liv.get("error"):
        log.error("[%s][%s] Liveness error: %s", rid, instance_name, liv["error"])
        return _rej("liveness_decode_error", error=liv["error"],
                    total_processing_time_ms=total_ms_fn(), rid=rid)

    liveness_blk = _liveness_block(liv)

    if liv.get("rejected_by") == "feature_visibility":
        fv     = liv.get("feature_visibility", {})
        reason = fv.get("reason") or "significant_face_features_not_visible"
        log.info("[%s][%s] REJECT  reason=%s  total_ms=%.1f",
                 rid, instance_name, reason, total_ms_fn())
        return _rej(reason, liveness=liveness_blk,
                    total_processing_time_ms=total_ms_fn(), rid=rid)

    if not liv["is_live"]:
        log.info("[%s][%s] REJECT liveness_failed  label=%s value=%s  total_ms=%.1f",
                 rid, instance_name, liv["label"], liv["value"], total_ms_fn())
        return _rej("liveness_failed", liveness=liveness_blk,
                    total_processing_time_ms=total_ms_fn(), rid=rid)

    log.info("[%s][%s] ACCEPT  label=%s value=%s  total_ms=%.1f",
             rid, instance_name, liv["label"], liv["value"], total_ms_fn())
    return _resp(True, "Liveness passed", {
        "final_status":             "ACCEPT",
        "liveness":                 liveness_blk,
        "total_processing_time_ms": total_ms_fn(),
    })


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/liveness-only", responses=_COMMON_ERROR_RESPONSES)
async def liveness_only(payload: Dict[str, Any]) -> Dict[str, Any]:
    rid = _next_request_id()

    if _queue_sem._value == 0:
        log.warning(
            "[%s] 503 queue full  inflight=%d  capacity=%d",
            rid, _inflight(), _QUEUE_CAPACITY,
        )
        raise HTTPException(503, detail="Server queue full, please retry shortly")

    async with _queue_sem:
        log.info(
            "[%s] admitted  inflight=%d/%d  waiting=%d",
            rid, _inflight(), _QUEUE_CAPACITY, _queue_depth_now(),
        )
        t_queued = time.monotonic()

        try:
            await asyncio.wait_for(
                _worker_sem.acquire(),
                timeout=QUEUE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            wait_ms = round((time.monotonic() - t_queued) * 1000, 2)
            log.warning(
                "[%s] 504 queue timeout  waited=%.1fms  timeout=%.1fs",
                rid, wait_ms, QUEUE_TIMEOUT_S,
            )
            raise HTTPException(
                504,
                detail=f"Request waited {wait_ms:.0f}ms but no worker became available",
            )

        wait_ms = round((time.monotonic() - t_queued) * 1000, 2)
        if wait_ms > 50:
            log.info("[%s] worker acquired after %.1fms in queue", rid, wait_ms)

        try:
            instance_name = _validate_instance_name(payload)

            log.info("[%s][%s] === /liveness-only START  queue_wait=%.1fms ===",
                     rid, instance_name, wait_ms)

            image_bytes = _extract_image_bytes(payload)
            _t_request  = time.monotonic()

            def _total_ms() -> float:
                return round((time.monotonic() - _t_request) * 1000, 2)

            try:
                liv = await _async_liveness(image_bytes, rid)
            except Exception as e:
                log.exception("[%s][%s] Liveness stage error", rid, instance_name)
                return _rej("liveness_error", error=repr(e),
                            total_processing_time_ms=_total_ms(), rid=rid)

            return _handle_liveness_result(liv, rid, instance_name, _total_ms)

        finally:
            _worker_sem.release()
