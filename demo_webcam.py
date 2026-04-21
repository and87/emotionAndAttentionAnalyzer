from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from attention_analyzer import AttentionAnalyzer


WINDOW_NAME = "Emotion And Attention Analyzer - Webcam Demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam demo for emotion, gaze, and engagement analysis.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index used by OpenCV.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda. Defaults to auto.")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam capture height.")
    parser.add_argument(
        "--analyze-every",
        type=int,
        default=3,
        help="Analyze one webcam frame every N displayed frames.",
    )
    parser.add_argument(
        "--engagement-window",
        type=int,
        default=128,
        help="Number of analyzed face frames used by the temporal EmotiEffLib engagement classifier.",
    )
    parser.add_argument(
        "--partial-engagement",
        action="store_true",
        help="Show an engagement estimate before the full temporal window is available.",
    )
    parser.add_argument(
        "--gaze-yaw-threshold-deg",
        type=float,
        default=20.0,
        help="Absolute OpenFace gaze yaw threshold used to mark attention=0.",
    )
    parser.add_argument(
        "--gaze-pitch-threshold-deg",
        type=float,
        default=18.0,
        help="Absolute OpenFace gaze pitch threshold used to mark attention=0.",
    )
    parser.add_argument(
        "--ear-attention-threshold",
        type=float,
        default=0.20,
        help="Minimum reliable EAR required to mark attention=1.",
    )
    parser.add_argument(
        "--head-yaw-attention-threshold-deg",
        type=float,
        default=35.0,
        help="Maximum absolute head yaw allowed to mark attention=1.",
    )
    parser.add_argument(
        "--head-pitch-attention-threshold-deg",
        type=float,
        default=25.0,
        help="Maximum absolute head pitch allowed to mark attention=1.",
    )
    parser.add_argument(
        "--ear-max-head-yaw-deg",
        type=float,
        default=40.0,
        help="Do not trust EAR when absolute head yaw is above this value.",
    )
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror the webcam image.")
    parser.set_defaults(mirror=True)
    return parser.parse_args()


def put_latest_frame(frame_queue: queue.Queue[np.ndarray], frame: np.ndarray) -> None:
    try:
        while True:
            frame_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass


def analysis_worker(
    analyzer: AttentionAnalyzer,
    frame_queue: queue.Queue[np.ndarray],
    stop_event: threading.Event,
    state_lock: threading.Lock,
    state: dict[str, Any],
    engagement_window: int,
    partial_engagement: bool,
) -> None:
    feature_window: deque[np.ndarray] = deque(maxlen=max(engagement_window, 1))

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        started_at = time.perf_counter()
        try:
            result = analyzer.analyze_frame(frame, include_temporal_features=True)
            engagement = {"engaged": None, "engagement_label": None}
            warmup = None

            if result.get("faces", 0) > 0:
                features = result[0].pop("_engagement_features", None)
                if features is not None:
                    feature_window.append(features)
                    if len(feature_window) >= engagement_window or partial_engagement:
                        engagement = analyzer.predict_temporal_engagement(list(feature_window))
                    else:
                        warmup = f"{len(feature_window)}/{engagement_window}"

            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            with state_lock:
                state["result"] = result
                state["engagement"] = engagement
                state["warmup"] = warmup
                state["inference_ms"] = elapsed_ms
                state["completed_at"] = time.perf_counter()
                state["analyzed_frames"] += 1
                state["error"] = None
        except Exception as exc:  # Keep the preview alive if one inference fails.
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            with state_lock:
                state["error"] = str(exc)
                state["inference_ms"] = elapsed_ms
                state["completed_at"] = time.perf_counter()


def format_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def draw_text_lines(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (16, 28),
    line_height: int = 24,
) -> None:
    x, y = origin
    panel_width = 520
    panel_height = line_height * len(lines) + 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def draw_faces(frame: np.ndarray, result: dict[str, Any] | None) -> None:
    if not result:
        return

    faces = int(result.get("faces", 0))
    for face_index in range(faces):
        face = result.get(face_index)
        if not face:
            continue

        bbox = face.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        attention = face.get("attention")
        attention_reason = face.get("attention_reason", "unknown")
        color = (80, 220, 120) if attention == 1 else (70, 70, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{face.get('main_emotion', 'Unknown')} | att={attention} {attention_reason}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def overlay_state(frame: np.ndarray, snapshot: dict[str, Any], display_fps: float) -> None:
    result = snapshot.get("result")
    engagement = snapshot.get("engagement") or {}
    error = snapshot.get("error")
    warmup = snapshot.get("warmup")
    inference_ms = snapshot.get("inference_ms")
    completed_at = snapshot.get("completed_at")
    analysis_age_ms = (time.perf_counter() - completed_at) * 1000.0 if completed_at else None

    if error:
        lines = [
            "Analysis error",
            error[:70],
            "Press q or ESC to quit",
        ]
        draw_text_lines(frame, lines)
        return

    if not result:
        lines = [
            "Waiting for first analysis...",
            f"display_fps={display_fps:.1f}",
            "Press q or ESC to quit",
        ]
        draw_text_lines(frame, lines)
        return

    faces = int(result.get("faces", 0))
    primary = result.get(0, {}) if faces else {}
    if not primary:
        lines = [
            "No face detected",
            f"display_fps={display_fps:.1f} inference_ms={format_float(inference_ms, 0)} age_ms={format_float(analysis_age_ms, 0)}",
            "Press q or ESC to quit",
        ]
        draw_text_lines(frame, lines)
        return

    engaged = engagement.get("engaged")
    engagement_label = engagement.get("engagement_label")
    engagement_text = "warming " + warmup if warmup else f"{format_float(engaged)}% {engagement_label or 'None'}"

    lines = [
        f"faces={faces} attention={primary.get('attention')} reason={primary.get('attention_reason')}",
        f"emotion={primary.get('main_emotion')} arousal={format_float(primary.get('arousal'), 3)}",
        f"engaged={engagement_text}",
        (
            "gaze yaw/pitch="
            f"{format_float(primary.get('openface_gaze_yaw_deg'))}/"
            f"{format_float(primary.get('openface_gaze_pitch_deg'))} deg"
        ),
        (
            "head yaw/pitch/roll="
            f"{format_float(primary.get('head_yaw_deg'))}/"
            f"{format_float(primary.get('head_pitch_deg'))}/"
            f"{format_float(primary.get('head_roll_deg'))} deg"
        ),
        (
            f"ear={format_float(primary.get('ear'), 4)} raw={format_float(primary.get('raw_ear'), 4)} "
            f"L/R={format_float(primary.get('left_ear'), 4)}/{format_float(primary.get('right_ear'), 4)}"
        ),
        f"display_fps={display_fps:.1f} inference_ms={format_float(inference_ms, 0)} age_ms={format_float(analysis_age_ms, 0)}",
        "Press q or ESC to quit",
    ]
    draw_text_lines(frame, lines)


def main() -> None:
    args = parse_args()
    analyze_every = max(args.analyze_every, 1)

    print("Loading analysis models...")
    analyzer = AttentionAnalyzer(
        device=args.device,
        gaze_yaw_threshold_deg=args.gaze_yaw_threshold_deg,
        gaze_pitch_threshold_deg=args.gaze_pitch_threshold_deg,
        ear_attention_threshold=args.ear_attention_threshold,
        head_yaw_attention_threshold_deg=args.head_yaw_attention_threshold_deg,
        head_pitch_attention_threshold_deg=args.head_pitch_attention_threshold_deg,
        ear_max_head_yaw_deg=args.ear_max_head_yaw_deg,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open webcam index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "result": None,
        "engagement": {"engaged": None, "engagement_label": None},
        "warmup": None,
        "inference_ms": None,
        "completed_at": None,
        "analyzed_frames": 0,
        "error": None,
    }

    worker = threading.Thread(
        target=analysis_worker,
        args=(
            analyzer,
            frame_queue,
            stop_event,
            state_lock,
            state,
            args.engagement_window,
            args.partial_engagement,
        ),
        daemon=True,
    )
    worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_index = 0
    previous_time = time.perf_counter()
    display_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_index += 1
            if frame_index % analyze_every == 0:
                put_latest_frame(frame_queue, frame.copy())

            now = time.perf_counter()
            delta = now - previous_time
            previous_time = now
            if delta > 0:
                display_fps = display_fps * 0.9 + (1.0 / delta) * 0.1 if display_fps else 1.0 / delta

            with state_lock:
                snapshot = dict(state)

            draw_faces(frame, snapshot.get("result"))
            overlay_state(frame, snapshot, display_fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
