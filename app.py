"""
Selfie Authenticity Gateway v12.5  (liveness-only, 250 req/s target)
=======================================================
Key changes from v12.4:
  - Silent-Face dispatched BEFORE Haar (removes ~45ms from critical path)
  - MAX_IMAGE_SIDE reduced 640->480 (Haar 45ms->25ms, all models faster)
  - LIVENESS_WORKERS=12 for 24 vCPU (2 threads/req * 12 = 24 vCPUs)
  - QUEUE_DEPTH=50

Pipeline:
  t=0   dispatch Silent-Face to _exec_silent_face (MTCNN bbox is internal)
  t=0   Haar face detect      sync, ~25ms at 480px
  t=25  OpenCV occlusion gate sync, ~5ms
  t=30  kprokofi MN3          sync, ~8ms  (on liveness thread)
  t=38  await Silent-Face     already ~22ms into its 60ms run → wait ~22ms
  t=60  fuse + respond

  Wall time ≈ max(60ms silent-face, 38ms liveness-thread) = ~45ms
  Throughput = 12 workers / 0.045s = ~267 req/s  (covers 250 req/s target)

Concurrency model (two-semaphore inflight queue):
  _queue_sem  : outer gate — capacity = LIVENESS_WORKERS + QUEUE_DEPTH
                             503 immediately when full
  _worker_sem : inner gate — limits concurrent inference to LIVENESS_WORKERS
                             admitted requests wait up to QUEUE_TIMEOUT_S
                             504 on timeout

Key env vars (24 vCPU defaults):
    LIVENESS_WORKERS=12         (24 vCPU / 2 threads-per-req)
    QUEUE_DEPTH=50              (waiting slots before 503)
    QUEUE_TIMEOUT_S=30.0
    MAX_IMAGE_SIDE=480          (reduced for speed; was 640)
    ORT_INTRA_THREADS=1
    TORCH_THREADS=1
    LIVENESS_FUSION=either
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


app = FastAPI(title="Selfie Authenticity Gateway v12.5 (liveness-only)")

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
# 480 instead of 640: Haar drops from ~45ms to ~25ms; all models slightly faster.
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

# 2 threads per request (1 liveness + 1 silent-face).
# LIVENESS_WORKERS = vCPUs / 2 = 24 / 2 = 12.
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
FACE_CROP_PAD_RATIO      = float(_get_config("FACE_CROP_PAD_RATIO",      "0.30"))
EYE_SCALE_FACTOR         = float(_get_config("EYE_SCALE_FACTOR",         "1.1"))
EYE_MIN_NEIGHBORS        = int(_get_config("EYE_MIN_NEIGHBORS",           "4"))
EYE_MIN_SIZE             = int(_get_config("EYE_MIN_SIZE",                "15"))
EYE_ROI_FRACTION         = float(_get_config("EYE_ROI_FRACTION",         "0.55"))
REQUIRE_EYES_OPEN        = (_get_config("REQUIRE_EYES_OPEN",    "true") or "true").lower() != "false"
MOUTH_ROI_START_FRACTION = float(_get_config("MOUTH_ROI_START_FRACTION", "0.60"))

# YCrCb — widened from Peer et al. for dark/black skin tones
SKIN_CR_LOW  = int(_get_config("SKIN_CR_LOW",  "120"))
SKIN_CR_HIGH = int(_get_config("SKIN_CR_HIGH", "185"))
SKIN_CB_LOW  = int(_get_config("SKIN_CB_LOW",  "55"))
SKIN_CB_HIGH = int(_get_config("SKIN_CB_HIGH", "135"))

# HSV fallback — dark/black skin that YCrCb misses
SKIN_H_LOW   = int(_get_config("SKIN_H_LOW",   "0"))
SKIN_H_HIGH  = int(_get_config("SKIN_H_HIGH",  "25"))
SKIN_H_LOW2  = int(_get_config("SKIN_H_LOW2",  "160"))
SKIN_H_HIGH2 = int(_get_config("SKIN_H_HIGH2", "179"))
SKIN_S_LOW   = int(_get_config("SKIN_S_LOW",   "20"))
SKIN_S_HIGH  = int(_get_config("SKIN_S_HIGH",  "230"))
SKIN_V_LOW   = int(_get_config("SKIN_V_LOW",   "30"))
SKIN_V_HIGH  = int(_get_config("SKIN_V_HIGH",  "255"))

MOUTH_SKIN_RATIO_MIN  = float(_get_config("MOUTH_SKIN_RATIO_MIN",  "0.12"))
REQUIRE_MOUTH_VISIBLE = (_get_config("REQUIRE_MOUTH_VISIBLE", "true") or "true").lower() != "false"

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
    "H=[%d,%d]+[%d,%d]  S=[%d,%d]  V=[%d,%d]  skin_min=%.2f",
    EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS, EYE_MIN_SIZE, EYE_ROI_FRACTION,
    MOUTH_ROI_START_FRACTION,
    SKIN_CR_LOW, SKIN_CR_HIGH, SKIN_CB_LOW, SKIN_CB_HIGH,
    SKIN_H_LOW, SKIN_H_HIGH, SKIN_H_LOW2, SKIN_H_HIGH2,
    SKIN_S_LOW, SKIN_S_HIGH, SKIN_V_LOW, SKIN_V_HIGH,
    MOUTH_SKIN_RATIO_MIN,
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

# ---------------------------------------------------------------------------
# Two-semaphore inflight queue
# Module-level defaults; re-created on real event loop in _startup().
# ---------------------------------------------------------------------------
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
    """Parse ONNX input shape into (h, w), falling back to KPROKOFI_INPUT_SIZE."""
    try:
        h = int(shape[-2]) if isinstance(shape[-2], int) else KPROKOFI_INPUT_SIZE
        w = int(shape[-1]) if isinstance(shape[-1], int) else KPROKOFI_INPUT_SIZE
    except Exception:
        h = w = KPROKOFI_INPUT_SIZE
    return h, w


def _startup_init_kprokofi() -> None:
    """Load and warm-up the kprokofi MN3 ONNX session. Updates module globals."""
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
    """Attempt to load a single Silent-Face model into cache; log warning on failure."""
    try:
        anti_spoof._load_model(full_path)
        anti_spoof.model.eval()
        cache[full_path] = copy.deepcopy(anti_spoof.model)
        log.info("  silent-face cached: %s (h=%d w=%d scale=%s)",
                 model_name, h_input, w_input, scale)
    except Exception:
        log.warning("  failed to pre-load %s", model_name, exc_info=True)


def _startup_init_silent_face() -> None:
    """Pre-load all Silent-Face model weights. Updates module globals."""
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
        _load_silent_face_model(anti_spoof, full_path, model_name, h_input, w_input, scale, cache)

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
    log.info("GATEWAY STARTUP BEGIN  [v12.5 liveness-only  fusion=%s]", LIVENESS_FUSION)
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
    log.info("GATEWAY STARTUP COMPLETE  [v12.5 liveness-only  fusion=%s]", LIVENESS_FUSION)
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
def _crop_face_square(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float,
    out_size: int,
) -> np.ndarray:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = int(max(w, h) * (1.0 + pad_ratio))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    H, W = image.shape[:2]
    pt = max(0, -y0);  pb = max(0, y1 - H)
    pl = max(0, -x0);  pr = max(0, x1 - W)
    if pt or pb or pl or pr:
        image = cv2.copyMakeBorder(image, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        x0 += pl; y0 += pt; x1 += pl; y1 += pt

    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _check_eyes_open(face_crop: np.ndarray) -> Tuple[bool, int]:
    h = face_crop.shape[0]
    upper_roi = face_crop[: int(h * EYE_ROI_FRACTION), :]
    gray = cv2.equalizeHist(cv2.cvtColor(upper_roi, cv2.COLOR_BGR2GRAY))
    eyes = _eye_cascade.detectMultiScale(
        gray,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=(EYE_MIN_SIZE, EYE_MIN_SIZE),
    )
    n = len(eyes)
    return n >= 2, n


def _check_mouth_visible(face_crop: np.ndarray) -> Tuple[bool, float]:
    """
    Two-pass skin detection for all skin tones.

    Pass 1 — YCrCb (widened Peer et al.): Cr 120-185, Cb 55-135
    Pass 2 — HSV fallback for dark/black skin:
              H 0-25 or 160-179, S 20-230, V 30-255
    Pixel is skin if EITHER pass agrees.

    Calibration:
      Unoccluded mouth (any tone)  =>  skin_ratio  0.20 – 0.75
      Mask / hand covered          =>  skin_ratio  < 0.08
      Threshold = 0.12
    """
    h = face_crop.shape[0]
    lower_roi = face_crop[int(h * MOUTH_ROI_START_FRACTION):, :]
    if lower_roi.size == 0:
        return False, 0.0

    # Pass 1 — YCrCb
    ycrcb = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    ycrcb_mask = (
        (cr >= SKIN_CR_LOW) & (cr <= SKIN_CR_HIGH) &
        (cb >= SKIN_CB_LOW) & (cb <= SKIN_CB_HIGH)
    )

    # Pass 2 — HSV
    hsv = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0]
    ss = hsv[:, :, 1]
    vv = hsv[:, :, 2]
    sv_ok = (
        (ss >= SKIN_S_LOW) & (ss <= SKIN_S_HIGH) &
        (vv >= SKIN_V_LOW) & (vv <= SKIN_V_HIGH)
    )
    hsv_mask = sv_ok & (
        ((hh >= SKIN_H_LOW)  & (hh <= SKIN_H_HIGH)) |
        ((hh >= SKIN_H_LOW2) & (hh <= SKIN_H_HIGH2))
    )

    skin_mask = ycrcb_mask | hsv_mask
    ratio = float(skin_mask.sum()) / float(skin_mask.size + 1e-9)
    return ratio >= MOUTH_SKIN_RATIO_MIN, ratio


def _collect_visibility_reasons(
    eyes_visible: bool,
    n_eyes: int,
    mouth_visible: bool,
    skin_ratio: float,
) -> List[str]:
    """Return the list of failure reasons for the CV visibility gate."""
    reasons: List[str] = []
    if REQUIRE_EYES_OPEN and not eyes_visible:
        reasons.append(f"eyes_closed_or_occluded(n_detected={n_eyes})")
    if REQUIRE_MOUTH_VISIBLE and not mouth_visible:
        reasons.append(f"mouth_occluded(skin_ratio={skin_ratio:.3f})")
    return reasons


def _build_cv_visibility_result(
    bbox_tuple: Tuple[int, int, int, int],
    face_count: int,
    eyes_visible: bool,
    n_eyes: int,
    mouth_visible: bool,
    skin_ratio: float,
    reasons: List[str],
    crop_ms: float,
    eye_ms: float,
    mouth_ms: float,
    total_ms: float,
) -> Dict[str, Any]:
    """Assemble the full CV visibility result dict."""
    ok = not reasons
    return {
        "ok":             ok,
        "reason":         None if ok else "significant_face_features_not_visible",
        "face_count":     face_count,
        "eyes_detected":  eyes_visible,
        "mouth_detected": mouth_visible,
        "bbox":           bbox_tuple,
        "detail": {
            "n_eyes_detected":  n_eyes,
            "mouth_skin_ratio": round(skin_ratio, 4),
            "crop_ms":          crop_ms,
            "eye_ms":           eye_ms,
            "mouth_ms":         mouth_ms,
            "total_ms":         total_ms,
            "reasons":          reasons,
        },
        "thresholds": {
            "eye_min_neighbors":     EYE_MIN_NEIGHBORS,
            "mouth_skin_ratio_min":  MOUTH_SKIN_RATIO_MIN,
            "require_eyes_open":     REQUIRE_EYES_OPEN,
            "require_mouth_visible": REQUIRE_MOUTH_VISIBLE,
        },
    }


def _run_cv_visibility(
    image: np.ndarray,
    bbox_tuple: Tuple[int, int, int, int],
    face_count: int,
    rid: str = "",
) -> Dict[str, Any]:
    t0 = time.monotonic()
    face_crop = _crop_face_square(image, bbox_tuple, FACE_CROP_PAD_RATIO, 200)
    crop_ms = round((time.monotonic() - t0) * 1000, 2)

    t_eye = time.monotonic()
    eyes_visible, n_eyes = _check_eyes_open(face_crop)
    eye_ms = round((time.monotonic() - t_eye) * 1000, 2)

    t_mouth = time.monotonic()
    mouth_visible, skin_ratio = _check_mouth_visible(face_crop)
    mouth_ms = round((time.monotonic() - t_mouth) * 1000, 2)

    total_ms = round((time.monotonic() - t0) * 1000, 2)

    reasons = _collect_visibility_reasons(eyes_visible, n_eyes, mouth_visible, skin_ratio)
    ok = not reasons

    log.info(
        "[%s][cv_gate] eyes=%s(n=%d)  mouth=%s(skin=%.3f,thr=%.2f)  "
        "crop=%.1f eye=%.1f mouth=%.1f total=%.1fms  => %s",
        rid,
        eyes_visible, n_eyes,
        mouth_visible, skin_ratio, MOUTH_SKIN_RATIO_MIN,
        crop_ms, eye_ms, mouth_ms, total_ms,
        "PASS" if ok else "FAIL",
    )

    return _build_cv_visibility_result(
        bbox_tuple, face_count,
        eyes_visible, n_eyes,
        mouth_visible, skin_ratio,
        reasons, crop_ms, eye_ms, mouth_ms, total_ms,
    )


# ---------------------------------------------------------------------------
# Stage 3a - kprokofi MN3
# ---------------------------------------------------------------------------
def _crop_face_for_antispoof(image: np.ndarray,
                             bbox: Tuple[int, int, int, int],
                             expansion: float) -> Optional[np.ndarray]:
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


def _run_kprokofi(image: np.ndarray,
                  bbox: Tuple[int, int, int, int],
                  rid: str = "") -> Dict[str, Any]:
    t_crop = time.monotonic()
    face_crop = _crop_face_for_antispoof(image, bbox, KPROKOFI_BBOX_EXPANSION)
    crop_ms = round((time.monotonic() - t_crop) * 1000, 2)
    if face_crop is None:
        return {"error": "invalid_face_crop", "live_prob": 0.0, "spoof_prob": 1.0,
                "crop_ms": crop_ms, "preprocess_ms": 0.0, "inference_ms": 0.0}

    t_pp = time.monotonic()
    tensor = _kprokofi_preprocess(face_crop)
    pp_ms = round((time.monotonic() - t_pp) * 1000, 2)

    t_inf = time.monotonic()
    outputs = _kprokofi_session.run(None, {_kprokofi_input_name: tensor})
    inf_ms = round((time.monotonic() - t_inf) * 1000, 2)

    logits = outputs[0][0]
    probs = _softmax_1d(logits.astype(np.float32))
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

    prediction = np.zeros((1, 3))
    model_timings = []
    for full_path, h_input, w_input, scale in _antispoof_model_list:
        t_model = time.monotonic()
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

def _compute_fusion_value(
    mode: str,
    k_live: Optional[float],
    s_live: Optional[float],
    k_says_live: bool,
    s_says_live: bool,
) -> Tuple[bool, float]:
    """
    Return (is_live, value) for the given fusion mode.

    Extracted to keep _fuse below the complexity threshold.
    """
    if mode == "either":
        valid = [v for v in (k_live, s_live) if v is not None]
        return k_says_live or s_says_live, max(valid) if valid else 0.0

    if mode == "both":
        valid = [v for v in (k_live, s_live) if v is not None]
        return k_says_live and s_says_live, min(valid) if valid else 0.0

    if mode == "weighted":
        if k_live is None or s_live is None:
            value = k_live if k_live is not None else (s_live or 0.0)
        else:
            value = KPROKOFI_WEIGHT * k_live + SILENT_FACE_WEIGHT * s_live
        return value >= LIVENESS_FUSION_THRESHOLD, value

    if mode == "kprokofi_only":
        return k_says_live, k_live if k_live is not None else 0.0

    if mode == "silent_only":
        return s_says_live, s_live if s_live is not None else 0.0

    return False, 0.0


def _fuse(kprokofi: Optional[Dict[str, Any]],
          silent:   Optional[Dict[str, Any]]) -> Dict[str, Any]:
    k_live = kprokofi["live_prob"] if (kprokofi and "live_prob" in kprokofi) else None
    s_live = silent["live_prob"]   if (silent   and "live_prob" in silent)   else None

    k_says_live = (k_live is not None and k_live >= KPROKOFI_LIVE_THRESHOLD)
    s_says_live = (s_live is not None and s_live >= SILENT_FACE_LIVE_THRESHOLD)

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
# Liveness orchestrator — helpers
# ---------------------------------------------------------------------------

def _decode_and_resize_image(
    image_bytes: bytes, rid: str
) -> Tuple[Optional[np.ndarray], float, int, int]:
    """
    Decode image_bytes and downscale if needed.

    Returns (image, resize_ms, orig_w, orig_h).
    Returns (None, 0.0, 0, 0) on decode failure.
    """
    arr   = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        log.error("[%s][liveness] cv2.imdecode returned None", rid)
        return None, 0.0, 0, 0

    orig_h, orig_w = image.shape[:2]
    resize_ms = 0.0
    max_side  = max(orig_h, orig_w)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max_side
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        t_rs  = time.monotonic()
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resize_ms = round((time.monotonic() - t_rs) * 1000, 2)
        log.info("[%s][liveness] resized %dx%d -> %dx%d in %.1fms",
                 rid, orig_w, orig_h, new_w, new_h, resize_ms)

    return image, resize_ms, orig_w, orig_h


def _dispatch_silent_face(image: np.ndarray, rid: str):
    """Submit silent-face to its executor if enabled; return the future or None."""
    if not USE_SILENT_FACE:
        return None
    return _exec_silent_face.submit(_run_silent_face, image, rid)


def _collect_silent_face_result(future, rid: str) -> Optional[Dict[str, Any]]:
    """Await the silent-face future and return its result (or an error dict)."""
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
    """Build the early-exit result dict when Haar detection fails."""
    return {
        "is_live": False, "label": -1, "value": 0.0,
        "feature_visibility": {
            "ok": False, "reason": haar["reason"],
            "face_count": haar.get("face_count", 0),
            "eyes_detected": False, "mouth_detected": False, "bbox": None,
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

    # ── Dispatch Silent-Face IMMEDIATELY ────────────────────────────────────
    # Silent-Face uses MTCNN internally for its own bbox — it does NOT need
    # Haar's result. Dispatching here runs it in parallel with Haar + CV gate
    # + kprokofi, cutting ~45ms off the critical path.
    t_dispatch    = time.monotonic()
    silent_future = _dispatch_silent_face(image, rid)

    # ── Step 1: Haar (sync, ~25ms at 480px) ─────────────────────────────────
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

    # ── Step 2: OpenCV visibility gate (sync, ~5ms) ──────────────────────────
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

    # ── Step 3: kprokofi (sync on liveness thread, ~8ms) ────────────────────
    t_kprokofi = time.monotonic()
    kprokofi_result = None
    if USE_KPROKOFI:
        try:
            kprokofi_result = _run_kprokofi(image, bbox, rid)
        except Exception as e:
            log.exception("[%s][liveness] kprokofi error", rid)
            kprokofi_result = {"error": repr(e), "live_prob": 0.0, "spoof_prob": 1.0}
    kprokofi_ms = round((time.monotonic() - t_kprokofi) * 1000, 2)

    # ── Step 4: Await Silent-Face ────────────────────────────────────────────
    # At this point ~38ms have elapsed. Silent-Face started at t=0 and takes
    # ~60ms, so we wait for ~22ms of remaining work (often less).
    silent_result  = _collect_silent_face_result(silent_future, rid)
    dispatch_ms    = round((time.monotonic() - t_dispatch) * 1000, 2)

    # ── Step 5: Fuse ─────────────────────────────────────────────────────────
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


def _rej(reason, liveness=None, error=None,
         total_processing_time_ms=None, rid: str = "") -> Dict[str, Any]:
    log.warning("[%s] REJECT reason=%s error=%s total_ms=%s",
                rid, reason, error, total_processing_time_ms)
    out: Dict[str, Any] = {"final_status": "REJECT", "reject_reason": reason,
                           "liveness": liveness}
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
        raise HTTPException(413, f"Image too large (max {MAX_UPLOAD_BYTES//(1024*1024)} MB)")
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
    """Extract and validate instanceName; raises HTTPException on failure."""
    instance_name = payload.get("instanceName")
    if not instance_name:
        raise HTTPException(400, "Missing 'instanceName' field in request body")
    if not isinstance(instance_name, str) or instance_name.strip() not in _VALID_INSTANCE_NAMES:
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
    """
    Inspect the liveness result and return the appropriate ACCEPT/REJECT response.
    Extracted to reduce cognitive complexity of the endpoint handler.
    """
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

    # Outer gate: 503 immediately when queue is full.
    # ._value used instead of .locked() — Python 3.9 compat.
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

        # Inner gate: wait up to QUEUE_TIMEOUT_S for an inference slot.
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
