"""将单路原始视频的 NMS-Free 推理结果转换为帧检测。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .records import FrameDetections, FrameSequence, VideoSource


def detections_from_result(result, source: VideoSource, frame_id: int, timestamp_s: float) -> FrameDetections:
    if result.boxes is None or tuple(result.orig_shape) != (source.height, source.width):
        raise ValueError("检测结果缺少框或与原始视频尺寸不一致")
    rows = result.boxes.data.detach().cpu().numpy()
    return FrameDetections(source, frame_id, timestamp_s, rows)


def load_detector(weights: Path, score: float, device: str | None):
    if not weights.is_file():
        raise ValueError(f"找不到本地检测权重：{weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    if not getattr(model.model, "end2end", False):
        raise ValueError("视频入口需要第一部分的 NMS-Free 端到端检测权重")

    def detect(image):
        return model.predict(image, conf=score, device=device, verbose=False, save=False)[0]

    return detect


def video_detections(video: Path, detect) -> Iterator[FrameDetections]:
    """逐帧解码，以解码时间戳核对固定帧率。"""
    import cv2

    if not video.is_file():
        raise ValueError(f"找不到原始视频：{video}")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"无法解码视频：{video}")
        source = VideoSource(
            video.name,
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise ValueError("无法核对视频总帧数，请使用可验证的固定帧率视频")
        frame_id = 0
        sequence = FrameSequence()
        while True:
            ok, image = capture.read()
            if not ok:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if frame_id == 0:
                origin = pts
            timestamp_s = pts - origin
            frame = detections_from_result(detect(image), source, frame_id, timestamp_s)
            sequence.advance(frame)
            yield frame
            frame_id += 1
        if frame_id != total_frames:
            raise ValueError(f"视频帧数不一致，预期 {total_frames} 帧，实际解码 {frame_id} 帧")
    finally:
        capture.release()
