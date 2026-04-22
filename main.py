from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from attention_analyzer import AttentionAnalyzer, resolve_torch_device

import torch


VIDEO_ROOT = Path("video")
CSV_ROOT = Path("csv")
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


CSV_COLUMNS = [
    "VideoTime",
    "numOfPersons",
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Arousal",
    "Engaged",
    "EngagementLabel",
    "mainEmotion",
    "OpenFaceGazeYawRad",
    "OpenFaceGazePitchRad",
    "OpenFaceGazeYawDeg",
    "OpenFaceGazePitchDeg",
    "OpenFaceGazeVectorX",
    "OpenFaceGazeVectorY",
    "OpenFaceGazeVectorZ",
    "HeadYawDeg",
    "HeadPitchDeg",
    "HeadRollDeg",
    "Ear",
]


FATAL_ACCELERATOR_ERROR_MARKERS = (
    "CUDA error",
    "no kernel image is available",
    "device-side assert",
    "CUBLAS_STATUS_ARCH_MISMATCH",
    "not compatible with the current PyTorch installation",
)


def csv_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.2f}".replace(".", ",")
    return str(value).replace(".", ",")


def output_path_for(video_path: Path, video_root: Path, output_root: Path) -> Path:
    relative = video_path.relative_to(video_root)
    flattened_name = "$".join(relative.with_suffix("").parts) + ".csv"
    return output_root / flattened_name


def find_videos(video_root: Path) -> list[Path]:
    return sorted(
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def is_fatal_accelerator_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in FATAL_ACCELERATOR_ERROR_MARKERS)


class VideoAnalysis:
    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        analyzer: AttentionAnalyzer,
        frame_step: int = 2,
        engagement_window: int = 128,
    ) -> None:
        self.video_path = video_path
        self.output_path = output_path
        self.analyzer = analyzer
        self.frame_step = max(frame_step, 1)
        self.engagement_window = max(engagement_window, 1)
        self.engagement_features: deque[Any] = deque(maxlen=self.engagement_window)
        self.pending_initial_rows: list[list[Any]] = []
        self.has_full_engagement_window = False

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.dimension = (self.width, self.height)
        self.total_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS))) or 1
        self.duration_in_seconds = int(self.total_frame_count / self.fps)
        self.length = dt.timedelta(seconds=self.duration_in_seconds)

    def get_video_info(self) -> None:
        print(f"Processing: {self.video_path}")
        print("dimension", self.dimension)
        print("length:", self.length)
        print("duration_in_seconds:", self.duration_in_seconds)
        print("frame_count:", self.total_frame_count)
        print("fps:", self.fps)

    def write_header(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="") as output_file:
            writer = csv.writer(output_file, delimiter=";")
            writer.writerow(["Video Summary"])
            writer.writerow(["File Name", "Dimension", "Length", "Duration(/s)", "Total Frame Count", "FPS"])
            writer.writerow(
                [
                    str(self.video_path),
                    self.dimension,
                    self.length,
                    self.duration_in_seconds,
                    self.total_frame_count,
                    self.fps,
                ]
            )
            writer.writerow(CSV_COLUMNS)

    def process_frames(self) -> None:
        frame_number = 0
        aborted = False
        try:
            while frame_number < self.total_frame_count:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.cap.read()
                if not ret:
                    break

                timestamp = dt.timedelta(seconds=int(frame_number / self.fps))
                temporal_engagement = None
                try:
                    analysis = self.analyzer.analyze_frame(frame, include_temporal_features=True)
                    temporal_engagement = self._update_temporal_engagement(analysis)
                    row = self._row_from_analysis(timestamp, analysis, temporal_engagement)
                except Exception as exc:
                    if is_fatal_accelerator_error(exc):
                        aborted = True
                        raise RuntimeError(
                            f"Fatal accelerator error while processing frame {frame_number}: {exc}"
                        ) from exc
                    print(f"Frame {frame_number}: analysis failed: {exc}")
                    row = self._empty_row(timestamp)

                self._write_or_buffer_row(row, temporal_engagement)
                frame_number += self.frame_step
        finally:
            if not aborted:
                self._flush_pending_initial_rows()
            self.cap.release()

    def _update_temporal_engagement(self, analysis: dict[Any, Any]) -> dict[str, Any]:
        if analysis.get("faces", 0) <= 0:
            return {"engaged": None, "engagement_label": None}

        features = analysis[0].get("_engagement_features")
        if features is None:
            return {"engaged": None, "engagement_label": None}

        self.engagement_features.append(features)
        if len(self.engagement_features) < self.engagement_window:
            return {"engaged": None, "engagement_label": None}
        self.has_full_engagement_window = True
        return self.analyzer.predict_temporal_engagement(list(self.engagement_features))

    def _row_from_analysis(
        self,
        timestamp: dt.timedelta,
        analysis: dict[Any, Any],
        temporal_engagement: dict[str, Any],
    ) -> list[Any]:
        if analysis.get("faces", 0) <= 0:
            return self._empty_row(timestamp)

        primary_face = analysis[0]
        emotions = primary_face.get("emotions", [None] * 7)
        return [
            timestamp,
            analysis["faces"],
            *emotions[:7],
            primary_face.get("arousal"),
            temporal_engagement.get("engaged"),
            temporal_engagement.get("engagement_label"),
            primary_face.get("main_emotion"),
            primary_face.get("openface_gaze_yaw_rad"),
            primary_face.get("openface_gaze_pitch_rad"),
            primary_face.get("openface_gaze_yaw_deg"),
            primary_face.get("openface_gaze_pitch_deg"),
            primary_face.get("openface_gaze_vector_x"),
            primary_face.get("openface_gaze_vector_y"),
            primary_face.get("openface_gaze_vector_z"),
            primary_face.get("head_yaw_deg"),
            primary_face.get("head_pitch_deg"),
            primary_face.get("head_roll_deg"),
            primary_face.get("ear"),
        ]

    @staticmethod
    def _empty_row(timestamp: dt.timedelta) -> list[Any]:
        return [timestamp, 0, "None", *["None"] * (len(CSV_COLUMNS) - 3)]

    def _write_or_buffer_row(self, row: list[Any], temporal_engagement: dict[str, Any] | None) -> None:
        if self.has_full_engagement_window:
            if self.pending_initial_rows:
                self._fill_initial_engagement_values(temporal_engagement)
                self._append_rows(self.pending_initial_rows)
                self.pending_initial_rows.clear()
            self._append_row(row)
            return

        self.pending_initial_rows.append(row)

    def _flush_pending_initial_rows(self) -> None:
        if not self.pending_initial_rows:
            return

        if self.engagement_features:
            temporal_engagement = self.analyzer.predict_temporal_engagement(list(self.engagement_features))
            self._fill_initial_engagement_values(temporal_engagement)

        self._append_rows(self.pending_initial_rows)
        self.pending_initial_rows.clear()

    def _fill_initial_engagement_values(self, temporal_engagement: dict[str, Any] | None) -> None:
        if not temporal_engagement:
            return
        engaged_index = CSV_COLUMNS.index("Engaged")
        label_index = CSV_COLUMNS.index("EngagementLabel")
        for row in self.pending_initial_rows:
            row[engaged_index] = temporal_engagement.get("engaged")
            row[label_index] = temporal_engagement.get("engagement_label")

    def _append_rows(self, rows: list[list[Any]]) -> None:
        with self.output_path.open("a", newline="") as output_file:
            writer = csv.writer(output_file, delimiter=";")
            writer.writerows([csv_value(value) for value in row] for row in rows)

    def _append_row(self, row: list[Any]) -> None:
        with self.output_path.open("a", newline="") as output_file:
            writer = csv.writer(output_file, delimiter=";")
            writer.writerow([csv_value(value) for value in row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze face emotions and gaze in videos.")
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT, help="Input video directory.")
    parser.add_argument("--output-root", type=Path, default=CSV_ROOT, help="Output CSV directory.")
    parser.add_argument("--frame-step", type=int, default=2, help="Analyze one frame every N frames.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, cuda:1. Defaults to auto.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--gpu-index", type=int, default=None, help="Convenience shortcut for --device cuda:N.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Abort if the selected/auto device is not CUDA.",
    )
    parser.add_argument(
        "--engagement-window",
        type=int,
        default=128,
        help="Number of analyzed face frames used by the temporal EmotiEffLib engagement classifier.",
    )
    return parser.parse_args()


def requested_device_from_args(args: argparse.Namespace) -> str:
    if args.cpu:
        return "cpu"
    if args.gpu_index is not None:
        return f"cuda:{args.gpu_index}"
    return args.device


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    memory_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    return f"cuda:{index} ({name}, {memory_gb:.1f} GB)"


def assert_device_can_execute(device: torch.device) -> None:
    if device.type != "cuda":
        return

    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        torch.cuda.set_device(index)
        with torch.no_grad():
            sample = torch.ones((16, 16), device=device)
            _ = (sample @ sample).sum().item()
        torch.cuda.synchronize(device)
    except Exception as exc:
        name = torch.cuda.get_device_name(index)
        capability = ".".join(str(part) for part in torch.cuda.get_device_capability(index))
        supported_arches = ", ".join(torch.cuda.get_arch_list()) or "unknown"
        raise SystemExit(
            "CUDA is visible, but PyTorch cannot execute kernels on this GPU.\n"
            f"GPU: {name} (compute capability sm_{capability.replace('.', '')})\n"
            f"PyTorch: {torch.__version__}, built CUDA: {torch.version.cuda}\n"
            f"PyTorch supported CUDA arch list: {supported_arches}\n"
            "This usually means the installed torch CUDA wheel is not compatible with the GPU architecture. "
            "Install a PyTorch build whose supported arch list includes this GPU, or run with --cpu."
        ) from exc


def main() -> None:
    args = parse_args()
    requested_device = requested_device_from_args(args)

    try:
        resolved_device = resolve_torch_device(requested_device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if args.require_gpu and resolved_device.type != "cuda":
        raise SystemExit(
            "GPU execution was required, but CUDA is not available in this environment. "
            "Install the CUDA PyTorch wheels or rerun without --require-gpu."
        )

    videos = find_videos(args.video_root)
    if not videos:
        print(f"No videos found under {args.video_root}")
        return

    analyzer = AttentionAnalyzer(
        device=str(resolved_device),
    )
    assert_device_can_execute(analyzer.device)
    print(f"Using device: {describe_device(analyzer.device)}")
    print(f"Found {len(videos)} video(s).")

    for video_path in videos:
        output_path = output_path_for(video_path, args.video_root, args.output_root)
        video = VideoAnalysis(
            video_path,
            output_path,
            analyzer,
            frame_step=args.frame_step,
            engagement_window=args.engagement_window,
        )
        video.get_video_info()
        video.write_header()
        video.process_frames()


if __name__ == "__main__":
    main()
