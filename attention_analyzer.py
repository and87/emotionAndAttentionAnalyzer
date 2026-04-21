"""Face emotion, gaze, and head-pose analysis using EmotiEffLib and OpenFace-3.0."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from math import cos, degrees, sin
from pathlib import Path
from typing import Any

# An empty/blank value hides every GPU from CUDA. In this project it is safer to treat it as
# unset; callers can still force CPU execution with --cpu/--device cpu.
if (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip() == "":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from emotiefflib.utils import get_engagement_classification_weights
from openface3_runtime import OPENFACE3_ROOT, ensure_openface3_paths

ensure_openface3_paths()

from model.MLT import MLT  # noqa: E402
from Pytorch_Retinaface.data import cfg_mnet  # noqa: E402
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox  # noqa: E402
from Pytorch_Retinaface.models.retinaface import RetinaFace  # noqa: E402
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm  # noqa: E402
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms  # noqa: E402
from STAR.demo import GetCropMatrix, TransformPerspective  # noqa: E402
from STAR.lib import utility as star_utility  # noqa: E402


EMOTION_OUTPUT_ORDER = (
    "Neutral",
    "Happy",
    "Surprise",
    "Sad",
    "Anger",
    "Disgust",
    "Fear",
)

STAR_LANDMARK_WEIGHTS_PATH = OPENFACE3_ROOT / "weights" / "Landmark_98.pkl"
ENGAGEMENT_WEIGHTS_PATH = Path(get_engagement_classification_weights())
WFLW_RIGHT_EYE_INDICES = tuple(range(60, 68))
WFLW_LEFT_EYE_INDICES = tuple(range(68, 76))


def resolve_torch_device(device: str | None = None) -> torch.device:
    """Resolve user-friendly device names and fail loudly for unavailable CUDA."""
    requested = (device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested, but this PyTorch install cannot use CUDA. "
                "Install the CUDA PyTorch wheels inside .venv before running the batch on GPU."
            )
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} requested, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return torch.device("cuda", index)

    return resolved


@dataclass(frozen=True)
class FaceAnalysis:
    bbox: tuple[int, int, int, int]
    confidence: float
    emotions: list[float]
    main_emotion: str
    arousal: float | None
    attention: int
    attention_reason: str
    gaze_yaw_rad: float
    gaze_pitch_rad: float
    gaze_yaw_deg: float
    gaze_pitch_deg: float
    gaze_vector: tuple[float, float, float]
    head_yaw_deg: float | None
    head_pitch_deg: float | None
    head_roll_deg: float | None
    ear: float | None
    raw_ear: float | None
    left_ear: float | None
    right_ear: float | None
    engagement_features: np.ndarray | None


class StarLandmarkEstimator:
    """Minimal STAR landmark runtime that avoids the training/demo side effects."""

    def __init__(self, model_path: Path, device: torch.device) -> None:
        device_id = -1 if device.type == "cpu" else (device.index or 0)
        args = argparse.Namespace(
            config_name="alignment",
            device_id=device_id,
            ckpt_dir=str(OPENFACE3_ROOT / "STAR" / ".runtime"),
        )
        self.config = star_utility.get_config(args)
        self.config.device_id = device_id
        star_utility.set_environment(self.config)

        self.input_size = 256
        self.target_face_scale = 1.0
        self.get_crop_matrix = GetCropMatrix(
            image_size=self.input_size,
            target_face_scale=self.target_face_scale,
            align_corners=True,
        )
        self.transform_perspective = TransformPerspective(image_size=self.input_size)
        self.model = star_utility.get_net(self.config)

        checkpoint = torch.load(model_path, map_location=self.config.device)
        state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.config.device)
        self.model.eval()

    def analyze(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        if width <= 1 or height <= 1:
            return None

        scale = min(width, height) / 200.0 * 1.05
        center_w = (x2 + x1) / 2.0
        center_h = (y2 + y1) / 2.0
        input_tensor, matrix = self._preprocess(image, scale, center_w, center_h)

        with torch.no_grad():
            output = self.model(input_tensor)
        landmarks = output[-1][0]
        landmarks = self._denorm_points(landmarks)
        landmarks = landmarks.detach().cpu().numpy()[0]
        return self._postprocess(landmarks, np.linalg.inv(matrix))

    def _preprocess(
        self,
        image: np.ndarray,
        scale: float,
        center_w: float,
        center_h: float,
    ) -> tuple[torch.Tensor, np.ndarray]:
        matrix = self.get_crop_matrix.process(scale, center_w, center_h)
        input_tensor = self.transform_perspective.process(image, matrix)
        input_tensor = torch.from_numpy(input_tensor[np.newaxis, :])
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        return input_tensor.to(self.config.device), matrix

    def _denorm_points(self, points: torch.Tensor) -> torch.Tensor:
        image_size = torch.tensor([self.input_size, self.input_size], device=points.device).view(1, 1, 2)
        return ((points + 1.0) * image_size - 1.0) / 2.0

    @staticmethod
    def _postprocess(src_points: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        points_h = np.concatenate([src_points, np.ones_like(src_points[:, [0]])], axis=1)
        dst_points = points_h @ coeff.T
        return (dst_points[:, :2] / dst_points[:, [2]]).astype(np.float32)


class TorchEngagementClassifier(torch.nn.Module):
    """PyTorch runtime for EmotiEffLib's temporal engagement classifier."""

    def __init__(self, feature_vector_dim: int = 2560) -> None:
        super().__init__()
        self.feature_vector_dim = feature_vector_dim
        self.e = torch.nn.Linear(feature_vector_dim, 1)
        self.hidden_fc = torch.nn.Linear(feature_vector_dim, 512)
        self.output_fc = torch.nn.Linear(512, 2)

    @classmethod
    def from_h5(cls, weights_path: Path, feature_vector_dim: int = 2560) -> "TorchEngagementClassifier":
        model = cls(feature_vector_dim=feature_vector_dim)
        with h5py.File(weights_path, "r") as weights:
            model.e.weight.data.copy_(torch.from_numpy(weights["e/e/kernel:0"][()].T))
            model.e.bias.data.copy_(torch.from_numpy(weights["e/e/bias:0"][()]))
            model.hidden_fc.weight.data.copy_(
                torch.from_numpy(weights["hidden_FC/hidden_FC/kernel:0"][()].T)
            )
            model.hidden_fc.bias.data.copy_(torch.from_numpy(weights["hidden_FC/hidden_FC/bias:0"][()]))
            model.output_fc.weight.data.copy_(torch.from_numpy(weights["dense/dense/kernel:0"][()].T))
            model.output_fc.bias.data.copy_(torch.from_numpy(weights["dense/dense/bias:0"][()]))
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.e(x).squeeze(-1), dim=1)
        context = (x * alpha.unsqueeze(-1)).sum(dim=1)
        hidden = F.relu(self.hidden_fc(context))
        return F.softmax(self.output_fc(hidden), dim=1)


class AttentionAnalyzer:
    """Runs the new production pipeline for one frame at a time."""

    def __init__(
        self,
        device: str | None = None,
        face_confidence_threshold: float = 0.7,
        gaze_yaw_threshold_deg: float = 20.0,
        gaze_pitch_threshold_deg: float = 18.0,
        ear_attention_threshold: float = 0.20,
        head_yaw_attention_threshold_deg: float = 35.0,
        head_pitch_attention_threshold_deg: float = 25.0,
        ear_max_head_yaw_deg: float = 40.0,
    ) -> None:
        self.device = resolve_torch_device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True

        self.face_confidence_threshold = face_confidence_threshold
        self.gaze_yaw_threshold_deg = gaze_yaw_threshold_deg
        self.gaze_pitch_threshold_deg = gaze_pitch_threshold_deg
        self.ear_attention_threshold = ear_attention_threshold
        self.head_yaw_attention_threshold_deg = head_yaw_attention_threshold_deg
        self.head_pitch_attention_threshold_deg = head_pitch_attention_threshold_deg
        self.ear_max_head_yaw_deg = ear_max_head_yaw_deg

        self.face_detector = self._load_face_detector()
        self.gaze_model = self._load_gaze_model()
        self.landmark_estimator = self._load_landmark_estimator()
        self.temporal_affect_model = EmotiEffLibRecognizer(
            engine="torch",
            model_name="enet_b0_8_va_mtl",
            device=str(self.device),
        )
        self.temporal_engagement_model = self._load_temporal_engagement_model()
        self.emotion_model = EmotiEffLibRecognizer(
            engine="torch",
            model_name="enet_b2_7",
            device=str(self.device),
        )

        self.gaze_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_retinaface_weights(model: RetinaFace, weights_path: Path, load_to_cpu: bool) -> RetinaFace:
        map_location = torch.device("cpu") if load_to_cpu else torch.device("cuda", torch.cuda.current_device())
        pretrained_dict = torch.load(weights_path, map_location=map_location)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        pretrained_dict = {
            key.split("module.", 1)[-1] if key.startswith("module.") else key: value
            for key, value in pretrained_dict.items()
        }
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def analyze_frame(self, frame: np.ndarray, include_temporal_features: bool = False) -> dict[Any, Any]:
        """Return the legacy-compatible frame analysis dictionary."""
        detections = self._detect_faces(frame)
        analyses: list[FaceAnalysis] = []
        for detection in detections:
            if float(detection[4]) < self.face_confidence_threshold:
                continue

            raw_bbox = self._clamp_bbox(tuple(detection[:4].astype(int)), frame.shape)
            x1, y1, x2, y2 = self._clamp_bbox(self._expand_bbox(detection[:4], frame.shape), frame.shape)
            face_bgr = frame[y1:y2, x1:x2]
            if face_bgr.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            emotions, main_emotion = self._predict_emotions(face_rgb)
            arousal, engagement_features = self._predict_temporal_affect(face_rgb)
            gaze = self._predict_gaze(face_rgb)
            landmarks = self._predict_landmarks(frame, raw_bbox)
            head_pose = self._compute_head_pose(landmarks, frame.shape) if landmarks is not None else None
            head_yaw, head_pitch, head_roll = head_pose or (None, None, None)
            ear_metrics = self._compute_eye_aspect_ratios(landmarks) if landmarks is not None else {}
            raw_ear = ear_metrics.get("ear")
            ear = self._reliable_ear(raw_ear, head_yaw)
            attention, attention_reason = self._classify_attention(
                gaze_yaw_deg=gaze["yaw_deg"],
                gaze_pitch_deg=gaze["pitch_deg"],
                ear=ear,
                head_yaw_deg=head_yaw,
                head_pitch_deg=head_pitch,
            )

            analyses.append(
                FaceAnalysis(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(detection[4]),
                    emotions=emotions,
                    main_emotion=main_emotion,
                    arousal=arousal,
                    attention=attention,
                    attention_reason=attention_reason,
                    gaze_yaw_rad=gaze["yaw_rad"],
                    gaze_pitch_rad=gaze["pitch_rad"],
                    gaze_yaw_deg=gaze["yaw_deg"],
                    gaze_pitch_deg=gaze["pitch_deg"],
                    gaze_vector=gaze["vector"],
                    head_yaw_deg=head_yaw,
                    head_pitch_deg=head_pitch,
                    head_roll_deg=head_roll,
                    ear=ear,
                    raw_ear=raw_ear,
                    left_ear=ear_metrics.get("left_ear"),
                    right_ear=ear_metrics.get("right_ear"),
                    engagement_features=engagement_features,
                )
            )

        return self._to_legacy_response(analyses, include_temporal_features=include_temporal_features)

    def _load_face_detector(self) -> RetinaFace:
        detector = RetinaFace(cfg=cfg_mnet, phase="test")
        detector = self._load_retinaface_weights(
            detector,
            OPENFACE3_ROOT / "weights" / "mobilenet0.25_Final.pth",
            load_to_cpu=self.device.type == "cpu",
        )
        detector.eval()
        return detector.to(self.device)

    def _load_gaze_model(self) -> MLT:
        model = MLT()
        state_dict = torch.load(
            OPENFACE3_ROOT / "weights" / "stage2_epoch_7_loss_1.1606_acc_0.5589.pth",
            map_location=self.device,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(self.device)

    def _load_landmark_estimator(self) -> StarLandmarkEstimator | None:
        if not STAR_LANDMARK_WEIGHTS_PATH.exists():
            return None
        return StarLandmarkEstimator(STAR_LANDMARK_WEIGHTS_PATH, self.device)

    def _load_temporal_engagement_model(self) -> TorchEngagementClassifier | None:
        if not ENGAGEMENT_WEIGHTS_PATH.exists():
            return None
        model = TorchEngagementClassifier.from_h5(ENGAGEMENT_WEIGHTS_PATH)
        return model.to(self.device)

    def _detect_faces(
        self,
        frame: np.ndarray,
        resize: float = 1.0,
        confidence_threshold: float = 0.02,
        nms_threshold: float = 0.4,
    ) -> np.ndarray:
        img = np.float32(frame)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape
        scale = torch.tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], device=self.device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.face_detector(img_tensor)

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet["variance"])
        boxes = (boxes * scale / resize).cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet["variance"])
        scale_landmarks = torch.tensor(
            [
                img_tensor.shape[3],
                img_tensor.shape[2],
                img_tensor.shape[3],
                img_tensor.shape[2],
                img_tensor.shape[3],
                img_tensor.shape[2],
                img_tensor.shape[3],
                img_tensor.shape[2],
                img_tensor.shape[3],
                img_tensor.shape[2],
            ],
            device=self.device,
        )
        landms = (landms * scale_landmarks / resize).cpu().numpy()

        indices = np.where(scores > confidence_threshold)[0]
        boxes = boxes[indices]
        landms = landms[indices]
        scores = scores[indices]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        return np.concatenate((dets, landms), axis=1)

    def _predict_emotions(self, face_rgb: np.ndarray) -> tuple[list[float], str]:
        labels, scores = self.emotion_model.predict_emotions(face_rgb, logits=False)
        label_to_score = {
            label: float(score) * 100.0 for label, score in zip(self.emotion_model.idx_to_emotion_class.values(), scores[0])
        }

        neutral = label_to_score.get("Neutral", 0.0)
        happy = label_to_score.get("Happiness", 0.0)
        surprise = label_to_score.get("Surprise", 0.0)
        sad = label_to_score.get("Sadness", 0.0)
        anger = label_to_score.get("Anger", 0.0)
        disgust = label_to_score.get("Disgust", 0.0)
        fear = label_to_score.get("Fear", 0.0)

        emotions = [neutral, happy, surprise, sad, anger, disgust, fear]
        return [round(value, 2) for value in emotions], labels[0]

    def _predict_temporal_affect(self, face_rgb: np.ndarray) -> tuple[float | None, np.ndarray | None]:
        features = self.temporal_affect_model.extract_features(face_rgb)
        _, scores = self.temporal_affect_model.classify_emotions(features, logits=False)
        arousal = float(scores[0, -1])
        return round(arousal, 4), features[0].astype(np.float32, copy=False)

    def _predict_gaze(self, face_rgb: np.ndarray) -> dict[str, Any]:
        image = self.gaze_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, gaze_output, _ = self.gaze_model(image)
        gaze = gaze_output[0].detach().cpu().numpy()
        yaw_rad = float(gaze[0])
        pitch_rad = float(gaze[1])
        vector = (
            -cos(pitch_rad) * sin(yaw_rad),
            -sin(pitch_rad),
            -cos(pitch_rad) * cos(yaw_rad),
        )
        return {
            "yaw_rad": round(yaw_rad, 6),
            "pitch_rad": round(pitch_rad, 6),
            "yaw_deg": round(degrees(yaw_rad), 2),
            "pitch_deg": round(degrees(pitch_rad), 2),
            "vector": tuple(round(value, 6) for value in vector),
        }

    def _predict_landmarks(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        if self.landmark_estimator is None:
            return None
        try:
            return self.landmark_estimator.analyze(frame, bbox)
        except Exception as exc:
            print(f"STAR landmark analysis failed: {exc}")
            return None

    def predict_temporal_engagement(self, feature_window: list[np.ndarray]) -> dict[str, Any]:
        if self.temporal_engagement_model is None:
            return {"engaged": None, "engagement_label": None}
        features = np.asarray(feature_window, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            return {"engaged": None, "engagement_label": None}

        feature_std = np.std(features, axis=0, keepdims=True)
        feature_std = np.repeat(feature_std, features.shape[0], axis=0)
        model_input = np.concatenate((feature_std, features), axis=1)
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scores = self.temporal_engagement_model(tensor)[0].detach().cpu().numpy()

        engaged = float(scores[1]) * 100.0
        label = "Engaged" if scores[1] >= scores[0] else "Distracted"
        return {"engaged": round(engaged, 2), "engagement_label": label}

    def _classify_attention(
        self,
        gaze_yaw_deg: float,
        gaze_pitch_deg: float,
        ear: float | None,
        head_yaw_deg: float | None,
        head_pitch_deg: float | None,
    ) -> tuple[int, str]:
        if head_yaw_deg is not None and abs(head_yaw_deg) > self.head_yaw_attention_threshold_deg:
            return 0, "head_yaw"
        if head_pitch_deg is not None and abs(head_pitch_deg) > self.head_pitch_attention_threshold_deg:
            return 0, "head_pitch"
        if ear is None:
            return 0, "ear_unreliable"
        if ear < self.ear_attention_threshold:
            return 0, "eyes_closed"
        if abs(gaze_yaw_deg) > self.gaze_yaw_threshold_deg:
            return 0, "gaze_yaw"
        if abs(gaze_pitch_deg) > self.gaze_pitch_threshold_deg:
            return 0, "gaze_pitch"
        return 1, "ok"

    def _reliable_ear(self, raw_ear: float | None, head_yaw_deg: float | None) -> float | None:
        if raw_ear is None:
            return None
        if raw_ear > 0.45:
            return None
        if head_yaw_deg is not None and abs(head_yaw_deg) > self.ear_max_head_yaw_deg:
            return None
        return raw_ear

    @staticmethod
    def _compute_eye_aspect_ratios(landmarks: np.ndarray) -> dict[str, float | None]:
        def eye_ear(indices: tuple[int, ...]) -> float | None:
            eye = landmarks[list(indices)]
            horizontal = np.linalg.norm(eye[0] - eye[4])
            if horizontal <= 1e-6:
                return None
            vertical = (
                np.linalg.norm(eye[1] - eye[7])
                + np.linalg.norm(eye[2] - eye[6])
                + np.linalg.norm(eye[3] - eye[5])
            )
            return float(vertical / (3.0 * horizontal))

        right_ear = eye_ear(WFLW_RIGHT_EYE_INDICES)
        left_ear = eye_ear(WFLW_LEFT_EYE_INDICES)
        values = [
            value
            for value in (right_ear, left_ear)
            if value is not None and np.isfinite(value)
        ]
        mean_ear = round(float(np.mean(values)), 4) if values else None
        return {
            "ear": mean_ear,
            "left_ear": round(float(left_ear), 4) if left_ear is not None else None,
            "right_ear": round(float(right_ear), 4) if right_ear is not None else None,
        }

    @staticmethod
    def _compute_head_pose(
        landmarks: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> tuple[float, float, float] | None:
        if landmarks.shape[0] < 88:
            return None

        image_left_eye_center = landmarks[list(WFLW_RIGHT_EYE_INDICES)].mean(axis=0)
        image_right_eye_center = landmarks[list(WFLW_LEFT_EYE_INDICES)].mean(axis=0)
        image_points = np.array(
            [
                landmarks[54],  # nose tip in the WFLW/98 layout
                landmarks[16],  # chin
                image_left_eye_center,
                image_right_eye_center,
                landmarks[76],  # approximate image-left mouth corner
                landmarks[82],  # approximate image-right mouth corner
            ],
            dtype=np.float64,
        )
        if not np.isfinite(image_points).all():
            return None

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-165.0, 170.0, -135.0),
                (165.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

        height, width = frame_shape[:2]
        focal_length = float(width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()
        pitch, roll = AttentionAnalyzer._canonicalize_pitch_roll(float(pitch), float(roll))
        yaw = AttentionAnalyzer._normalize_angle(float(yaw))
        return round(float(yaw), 2), round(float(pitch), 2), round(float(roll), 2)

    @staticmethod
    def _canonicalize_pitch_roll(pitch: float, roll: float) -> tuple[float, float]:
        # cv2.decomposeProjectionMatrix can return an equivalent solution around +/-180 degrees.
        if pitch < -90.0:
            pitch += 180.0
            roll -= 180.0 if roll >= 0.0 else -180.0
        elif pitch > 90.0:
            pitch -= 180.0
            roll -= 180.0 if roll >= 0.0 else -180.0
        return AttentionAnalyzer._normalize_angle(pitch), AttentionAnalyzer._normalize_angle(roll)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    @staticmethod
    def _expand_bbox(bbox: np.ndarray, frame_shape: tuple[int, ...], margin: float = 0.15) -> tuple[int, int, int, int]:
        del frame_shape
        x1, y1, x2, y2 = bbox.astype(int)
        width = x2 - x1
        height = y2 - y1
        dx = int(width * margin)
        dy = int(height * margin)
        return x1 - dx, y1 - dy, x2 + dx, y2 + dy

    @staticmethod
    def _clamp_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        return int(max(x1, 0)), int(max(y1, 0)), int(min(x2, width)), int(min(y2, height))

    @staticmethod
    def _to_legacy_response(
        analyses: list[FaceAnalysis],
        include_temporal_features: bool = False,
    ) -> dict[Any, Any]:
        data: dict[Any, Any] = {"faces": len(analyses)}
        for index, analysis in enumerate(analyses):
            face_data = {
                "attention": analysis.attention,
                "attention_reason": analysis.attention_reason,
                "emotions": analysis.emotions,
                "main_emotion": analysis.main_emotion,
                "arousal": analysis.arousal,
                "openface_gaze_yaw_rad": analysis.gaze_yaw_rad,
                "openface_gaze_pitch_rad": analysis.gaze_pitch_rad,
                "openface_gaze_yaw_deg": analysis.gaze_yaw_deg,
                "openface_gaze_pitch_deg": analysis.gaze_pitch_deg,
                "openface_gaze_vector_x": analysis.gaze_vector[0],
                "openface_gaze_vector_y": analysis.gaze_vector[1],
                "openface_gaze_vector_z": analysis.gaze_vector[2],
                "head_yaw_deg": analysis.head_yaw_deg,
                "head_pitch_deg": analysis.head_pitch_deg,
                "head_roll_deg": analysis.head_roll_deg,
                "ear": analysis.ear,
                "raw_ear": analysis.raw_ear,
                "left_ear": analysis.left_ear,
                "right_ear": analysis.right_ear,
                "bbox": analysis.bbox,
                "confidence": round(analysis.confidence, 4),
            }
            if include_temporal_features:
                face_data["_engagement_features"] = analysis.engagement_features
            data[index] = face_data
        return data


_DEFAULT_ANALYZER: AttentionAnalyzer | None = None


def get_default_analyzer() -> AttentionAnalyzer:
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = AttentionAnalyzer()
    return _DEFAULT_ANALYZER


def get_people_attention(frame: np.ndarray) -> dict[Any, Any]:
    return get_default_analyzer().analyze_frame(frame)
