# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
腕部相机姿态校准脚本（Piper 专用）。

通过"在固定关节位处对比当前画面与参考画面"间接保证 wrist 相机视角和数据采集时一致，
不需要做手眼标定也不需要测量支架倾斜角。

两种模式：
    record  示教定位 -> 关节角锁定 -> 等到位 -> N 帧中位数融合得到参考图
    verify  机械臂走到记录位 -> 实时与参考图配准 -> overlay 与客观指标 (PASS/WARN/FAIL)

示例：
```shell
# 录制参考
python -m lerobot.scripts.lerobot_wrist_calib record \
  --name=v1 \
  --robot.type=piper --robot.can_name=can0 \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: "/dev/video1", width: 1280, height: 720, fps: 30} }' \
  --camera_key=wrist

# 校验
python -m lerobot.scripts.lerobot_wrist_calib verify \
  --name=v1 \
  --robot.type=piper --robot.can_name=can0 \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: "/dev/video1", width: 1280, height: 720, fps: 30} }' \
  --camera_key=wrist
```
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat

import cv2
import numpy as np
import yaml

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.motors.piper.piper import GRIPPER_FACTOR, JOINT_FACTOR
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    piper,
)
from lerobot.robots.piper.piper_robot import PiperRobot
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

PIPER_DOF = 7  # 6 joints (rad) + gripper (m)


# ============================================================
# Config
# ============================================================


@dataclass
class ThresholdConfig:
    # 默认 None 时按图像短边自动取 0.6%
    dx: float | None = None
    dy: float | None = None
    rot_deg: float = 0.6
    scale: float = 0.01
    # ECC 相关分数阈值
    ecc_pass: float = 0.90
    ecc_warn: float = 0.75
    # ORB 相关 inlier 比例阈值
    orb_inlier_pass: float = 0.6
    orb_inlier_warn: float = 0.4


@dataclass
class WristCalibConfig:
    robot: RobotConfig
    name: str = "v1"
    mode: str = "record"  # record | verify
    camera_key: str = "wrist"
    output_root: str = "outputs/wrist_calib"

    # 关节静止判据
    settle_eps: float = 0.002  # rad
    settle_window: int = 10  # 连续 N 次满足 eps
    settle_timeout_s: float = 15.0
    settle_poll_dt_s: float = 0.05

    # record 相关
    avg_frames: int = 10  # 中位数融合的帧数

    # verify 相关
    backend: str = "ecc"  # ecc | orb
    ecc_motion: str = "euclidean"  # euclidean | similarity
    ecc_pyramid_levels: int = 3
    display_max_height: int = 360  # 单个面板的高度（用于显示）

    thresh: ThresholdConfig = field(default_factory=ThresholdConfig)
    note: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("record", "verify"):
            raise ValueError(f"--mode 必须是 record 或 verify，当前: {self.mode}")
        if self.backend not in ("ecc", "orb"):
            raise ValueError(f"--backend 必须是 ecc 或 orb，当前: {self.backend}")
        if self.ecc_motion not in ("euclidean", "similarity"):
            raise ValueError(f"--ecc_motion 必须是 euclidean 或 similarity，当前: {self.ecc_motion}")


# ============================================================
# 关节读写 / 静止判据
# ============================================================


def _read_joints_rad(robot: PiperRobot) -> np.ndarray:
    """读取当前 7 维关节状态（前 6 维弧度，第 7 维夹爪米）。"""
    raw = robot.bus.read()
    joints = (
        np.array(
            [
                raw["joint_1"],
                raw["joint_2"],
                raw["joint_3"],
                raw["joint_4"],
                raw["joint_5"],
                raw["joint_6"],
            ],
            dtype=np.float64,
        )
        / JOINT_FACTOR
    )
    gripper = raw["gripper"] / GRIPPER_FACTOR
    return np.concatenate([joints, [gripper]])


def _drive_and_settle(robot: PiperRobot, target: np.ndarray, cfg: WristCalibConfig) -> None:
    """连续下发目标关节位并阻塞等待到位/静止。

    与 lerobot_replay.py 的 record_loop/replay 一样，需要把目标关节角作为"心跳"
    持续重发（这里 ~20Hz）。Piper SDK 与 ROS publisher 一样：单次写不足以维持运动，
    必须连续 publish。否则即便指令被收到，控制器也可能很快回到安全/保持模式，
    表现为关节误差迟迟不收敛。
    """
    target_list = target.tolist()
    deadline = time.monotonic() + cfg.settle_timeout_s
    consecutive_ok = 0
    last_err = float("inf")
    last_log = time.monotonic()

    while time.monotonic() < deadline:
        # 心跳：每个 poll 周期都重新下发一次目标，与 replay 主循环节奏相同
        robot.bus.write(target_list)

        cur = _read_joints_rad(robot)
        max_err = float(np.max(np.abs(cur[:6] - target[:6])))
        last_err = max_err

        if max_err < cfg.settle_eps:
            consecutive_ok += 1
            if consecutive_ok >= cfg.settle_window:
                logger.info("机械臂已稳定于目标位姿（max_err=%.5f rad）", max_err)
                return
        else:
            consecutive_ok = 0

        # 每 1s 打印一次进度，方便观察是否在向目标收敛
        if time.monotonic() - last_log > 1.0:
            logger.info(
                "到位中... max_err=%.4f rad (eps=%.4f, 连续命中=%d/%d)",
                max_err,
                cfg.settle_eps,
                consecutive_ok,
                cfg.settle_window,
            )
            last_log = time.monotonic()

        time.sleep(cfg.settle_poll_dt_s)

    logger.warning(
        "机械臂未在 %.1fs 内静止（max_err=%.5f rad）。常见原因："
        "1) 机械臂处于示教/拖动模式（按钮亮）→请切到位置控制（按钮灭）；"
        "2) 目标距当前位置过远 → 增大 --settle_timeout_s；"
        "3) CAN 连接异常或电机未使能。继续但参考帧/对比可能不精准。",
        cfg.settle_timeout_s,
        last_err,
    )


# ============================================================
# 相机配置一致性
# ============================================================


def _camera_id_dict(cfg) -> dict:
    """提取与图像几何一致性相关的字段。"""
    color_mode = getattr(cfg, "color_mode", None)
    if hasattr(color_mode, "value"):
        color_mode = color_mode.value
    rotation = getattr(cfg, "rotation", None)
    if hasattr(rotation, "value"):
        rotation = rotation.value
    return {
        "type": cfg.type,
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
        "color_mode": str(color_mode) if color_mode is not None else None,
        "rotation": int(rotation) if rotation is not None else None,
    }


def _camera_is_rgb(cam_cfg) -> bool:
    color_mode = getattr(cam_cfg, "color_mode", None)
    if color_mode is None:
        # RealSense 默认 RGB（见 camera_realsense._postprocess_image），OpenCV 默认 RGB
        return True
    if hasattr(color_mode, "value"):
        color_mode = color_mode.value
    return str(color_mode).lower() == "rgb"


def _capture_median_bgr(camera, n_frames: int, is_rgb: bool) -> np.ndarray:
    """连续读 n_frames 帧并取像素中位数，返回 BGR uint8 图。"""
    if n_frames < 1:
        raise ValueError("avg_frames 必须 >= 1")
    frames = []
    for _ in range(n_frames):
        f = camera.async_read(timeout_ms=1000)
        if is_rgb:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        frames.append(f.copy())
        time.sleep(0.05)
    stack = np.stack(frames, axis=0)
    fused = np.median(stack, axis=0).astype(np.uint8)
    return fused


# ============================================================
# 配准算法
# ============================================================


def _ecc_align(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    motion: str = "euclidean",
    n_levels: int = 3,
) -> tuple[np.ndarray | None, float]:
    """ECC 金字塔配准，返回 2x3 变换矩阵 W（前向：把 ref 坐标映射到 cur 坐标）和 ECC 分数。

    `cv2.findTransformECC(template, input, W, ...)` 的官方契约是
    `warpAffine(input, W, flags=WARP_INVERSE_MAP) ≈ template`，即返回的 W 把
    template 坐标映射到 input 坐标（template→input 的前向变换）。

    为了让 ECC 与 ORB(`estimateAffinePartial2D(src=ref, dst=cur)`) 返回方向一致，
    这里把 template 设为 cur、input 设为 ref，于是 W 把 cur 坐标映射到 ref 坐标。
    然后我们对 W 取仿射逆，得到 ref→cur 前向变换，便于：
        1) 与 ORB 路径共享下游解算和渲染逻辑；
        2) `warpAffine(ref, W_inv)`（默认前向语义）直接把参考图渲染到当前帧空间，
           方便做 overlay。

    返回的 W 解释：当前帧画面中的特征看起来相对参考帧位移 (dx, dy)、旋转 rot 度、
    缩放 scale；不直接等价于相机在世界中的位移方向。
    """
    motion_type = cv2.MOTION_EUCLIDEAN if motion == "euclidean" else cv2.MOTION_SIMILARITY

    n_levels = max(1, int(n_levels))
    pyr_ref = [ref_gray]
    pyr_cur = [cur_gray]
    for _ in range(n_levels - 1):
        pyr_ref.append(cv2.pyrDown(pyr_ref[-1]))
        pyr_cur.append(cv2.pyrDown(pyr_cur[-1]))

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

    score = 0.0
    for i in range(n_levels - 1, -1, -1):
        if i < n_levels - 1:
            warp[0, 2] *= 2.0
            warp[1, 2] *= 2.0
        try:
            score, warp = cv2.findTransformECC(
                pyr_cur[i],
                pyr_ref[i],
                warp,
                motion_type,
                criteria,
                None,
                5,
            )
        except cv2.error as e:
            logger.debug("ECC 在金字塔层 %d 失败: %s", i, e)
            return None, 0.0

    # ECC 返回的 W 是 cur→ref（template→input）；取仿射逆得到 ref→cur 前向变换。
    w_forward = cv2.invertAffineTransform(warp).astype(np.float32)
    return w_forward, float(score)


def _orb_align(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    n_features: int = 2000,
) -> tuple[np.ndarray | None, float, int, int]:
    """ORB 特征 + RANSAC 部分仿射，返回 (W 2x3 maps ref→cur, inlier_ratio, n_matches, n_inliers)。"""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0.0, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, 0.0, len(knn), 0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if M is None:
        return None, 0.0, len(good), 0
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    inlier_ratio = float(n_inliers / max(1, len(good)))
    return M.astype(np.float32), inlier_ratio, len(good), n_inliers


def _decompose_similarity(M: np.ndarray) -> tuple[float, float, float, float]:
    """从 2x3 矩阵分解为 (dx, dy, rot_deg, scale)。

    M = [[s*cos(t), -s*sin(t), tx],
         [s*sin(t),  s*cos(t), ty]]
    """
    a = float(M[0, 0])
    b = float(M[1, 0])
    scale = float(np.sqrt(a * a + b * b))
    rot_rad = float(np.arctan2(b, a))
    rot_deg = float(np.degrees(rot_rad))
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    return dx, dy, rot_deg, scale


# ============================================================
# 阈值判定
# ============================================================


def _classify(
    metrics: dict,
    thresh: ThresholdConfig,
    short_side_px: int,
    backend: str,
    align_ok: bool,
) -> str:
    if not align_ok:
        return "FAIL"

    dx_thr = thresh.dx if thresh.dx is not None else 0.006 * short_side_px
    dy_thr = thresh.dy if thresh.dy is not None else 0.006 * short_side_px
    abs_dx = abs(metrics["dx"])
    abs_dy = abs(metrics["dy"])
    abs_rot = abs(metrics["rot_deg"])
    abs_scale = abs(metrics["scale"] - 1.0)

    score = metrics["score"]
    if backend == "ecc":
        score_pass = score >= thresh.ecc_pass
        score_warn = score >= thresh.ecc_warn
    else:
        score_pass = score >= thresh.orb_inlier_pass
        score_warn = score >= thresh.orb_inlier_warn

    geom_pass = (
        abs_dx < dx_thr
        and abs_dy < dy_thr
        and abs_rot < thresh.rot_deg
        and abs_scale < thresh.scale
    )
    geom_warn = (
        abs_dx < 2 * dx_thr
        and abs_dy < 2 * dy_thr
        and abs_rot < 2 * thresh.rot_deg
        and abs_scale < 2 * thresh.scale
    )

    if score_pass and geom_pass:
        return "PASS"
    if score_warn and geom_warn:
        return "WARN"
    return "FAIL"


# ============================================================
# 可视化
# ============================================================


def _make_panel(img_bgr: np.ndarray, title: str, h_target: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = h_target / max(1, h)
    new_w = max(1, int(w * scale))
    resized = cv2.resize(img_bgr, (new_w, h_target))
    title_strip = np.zeros((28, new_w, 3), dtype=np.uint8)
    cv2.putText(
        title_strip,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([title_strip, resized])


_STATUS_COLOR = {
    "PASS": (0, 200, 0),
    "WARN": (0, 200, 220),
    "FAIL": (0, 0, 220),
}


def _render_view(
    ref_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    warp_matrix: np.ndarray | None,
    metrics: dict,
    status: str,
    overlay_mode: str,
    backend: str,
    motion: str,
    h_panel: int,
) -> np.ndarray:
    h, w = cur_bgr.shape[:2]

    if warp_matrix is not None and warp_matrix.shape == (2, 3):
        ref_warped = cv2.warpAffine(
            ref_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
    else:
        ref_warped = np.zeros_like(cur_bgr)

    if overlay_mode == "alpha":
        overlay = cv2.addWeighted(cur_bgr, 0.5, ref_warped, 0.5, 0)
    else:  # edge
        cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        ref_warp_gray = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2GRAY)
        cur_edges = cv2.Canny(cur_gray, 80, 160)
        ref_edges = cv2.Canny(ref_warp_gray, 80, 160)
        overlay = cur_bgr.copy()
        overlay[ref_edges > 0] = (0, 255, 0)  # 参考边缘 - 绿
        overlay[cur_edges > 0] = (0, 0, 255)  # 当前边缘 - 红

    diff = cv2.absdiff(ref_warped, cur_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)

    panels = [
        _make_panel(ref_bgr, "REFERENCE", h_panel),
        _make_panel(overlay, f"CURRENT + {overlay_mode.upper()} OVERLAY", h_panel),
        _make_panel(diff_color, "ABS DIFF (warped ref vs current)", h_panel),
    ]
    grid = np.hstack(panels)

    hud_h = 56
    hud = np.zeros((hud_h, grid.shape[1], 3), dtype=np.uint8)
    color = _STATUS_COLOR.get(status, (128, 128, 128))
    cv2.rectangle(hud, (0, 0), (160, hud_h), color, -1)
    cv2.putText(
        hud, status, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
    )

    score_label = "ecc" if backend == "ecc" else "inlier"
    line1 = (
        f"dx={metrics['dx']:+7.2f}px  dy={metrics['dy']:+7.2f}px  "
        f"rot={metrics['rot_deg']:+6.3f}deg  scale={metrics['scale']:.4f}"
    )
    line2 = (
        f"backend={backend}({motion if backend=='ecc' else 'partial-affine'})  "
        f"{score_label}={metrics['score']:.3f}"
    )
    cv2.putText(
        hud, line1, (180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
    )
    cv2.putText(
        hud, line2, (180, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA
    )

    return np.vstack([hud, grid])


# ============================================================
# Record / Verify
# ============================================================


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_record(cfg: WristCalibConfig, robot: PiperRobot, camera) -> None:
    out_dir = _ensure_dir(Path(cfg.output_root) / cfg.name)

    print("=" * 64)
    print(f"  Wrist 相机校准 - RECORD 模式 (name={cfg.name})")
    print("=" * 64)
    print("即将进入软件拖动示教模式：")
    print("  1) 手动把臂拖到目标观察位")
    print("  2) 同时把夹爪调整到稳定状态（开/合到希望对比时使用的位置）")
    print("  3) 完成后按回车键确认")
    print()
    robot.bus.enter_teach_mode()
    try:
        input(">>> 拖动完成后按回车: ")
    finally:
        robot.bus.exit_teach_mode()

    target = _read_joints_rad(robot)
    np_str = np.array2string(target, precision=4, suppress_small=True)
    logger.info("捕获目标关节位 (rad+m): %s", np_str)

    _drive_and_settle(robot, target, cfg)

    cam_cfg = camera.config
    is_rgb = _camera_is_rgb(cam_cfg)
    logger.info("正在抓取 %d 帧并按像素中位数融合...", cfg.avg_frames)
    fused_bgr = _capture_median_bgr(camera, cfg.avg_frames, is_rgb=is_rgb)

    ref_path = out_dir / "ref.png"
    if not cv2.imwrite(str(ref_path), fused_bgr):
        raise RuntimeError(f"参考图写入失败: {ref_path}")

    pose_path = out_dir / "pose.yaml"
    pose = {
        "joints_rad": [float(x) for x in target[:6]],
        "gripper_m": float(target[6]),
        "settle_eps": float(cfg.settle_eps),
        "settle_window": int(cfg.settle_window),
    }
    with open(pose_path, "w") as f:
        yaml.safe_dump(pose, f, sort_keys=False)

    meta_path = out_dir / "meta.json"
    meta = {
        "name": cfg.name,
        "camera_key": cfg.camera_key,
        "camera_config": _camera_id_dict(cam_cfg),
        "robot_type": getattr(cfg.robot, "type", "piper"),
        "image_shape_hwc": list(fused_bgr.shape),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": cfg.note,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"DONE. 校准包已保存至: {out_dir}")
    print(f"  ref:  {ref_path}")
    print(f"  pose: {pose_path}")
    print(f"  meta: {meta_path}")


def _validate_camera_consistency(saved: dict, cur_cfg) -> None:
    """对比保存的相机配置与当前 CLI 中的相机配置，几何相关字段必须一致。"""
    cur = _camera_id_dict(cur_cfg)
    mismatches = []
    for k in ("type", "width", "height", "fps", "color_mode", "rotation"):
        if saved.get(k) != cur.get(k):
            mismatches.append((k, cur.get(k), saved.get(k)))
    if mismatches:
        msg_lines = [f"  - {k}: cli={c} saved={s}" for k, c, s in mismatches]
        raise ValueError(
            "相机配置与参考图录制时不一致，无法保证指标可比：\n"
            + "\n".join(msg_lines)
            + "\n请用与 record 时完全相同的 --robot.cameras 配置重试。"
        )


def _run_verify(cfg: WristCalibConfig, robot: PiperRobot, camera) -> None:
    out_dir = Path(cfg.output_root) / cfg.name
    pose_path = out_dir / "pose.yaml"
    ref_path = out_dir / "ref.png"
    meta_path = out_dir / "meta.json"
    for p in (pose_path, ref_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"找不到校准包文件 {p}。请先用 `record` 模式录制 (name={cfg.name})。"
            )

    with open(pose_path) as f:
        pose = yaml.safe_load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    cam_cfg = camera.config
    _validate_camera_consistency(meta["camera_config"], cam_cfg)
    logger.info("相机配置一致性校验通过。")

    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise RuntimeError(f"无法读取参考图: {ref_path}")
    short_side = int(min(ref_bgr.shape[:2]))

    target = np.array(list(pose["joints_rad"]) + [pose["gripper_m"]], dtype=np.float64)
    if target.shape[0] != PIPER_DOF:
        raise ValueError(f"pose.yaml 期望 {PIPER_DOF} 维，得到 {target.shape[0]}")
    _drive_and_settle(robot, target, cfg)

    diag_dir = _ensure_dir(out_dir / "diagnostics")

    overlay_modes = ["alpha", "edge"]
    backends = ["ecc", "orb"]
    motions = ["euclidean", "similarity"]

    overlay_idx = 0
    backend = cfg.backend
    motion = cfg.ecc_motion

    is_rgb = _camera_is_rgb(cam_cfg)

    win_name = f"wrist-calib verify [{cfg.name}]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("=" * 64)
    print(f"  Wrist 相机校准 - VERIFY 模式 (name={cfg.name})")
    print("=" * 64)
    print("快捷键:")
    print("  q  退出")
    print("  s  保存当前快照（cur+视图+指标）至 diagnostics/")
    print("  e  切换 overlay 模式 (alpha / edge)")
    print("  b  切换配准 backend (ecc / orb)")
    print("  m  切换 ECC 运动模型 (euclidean / similarity)")
    print()

    try:
        while True:
            try:
                cur_raw = camera.async_read(timeout_ms=500)
            except TimeoutError:
                logger.warning("相机读取超时，跳过本帧")
                continue

            cur_bgr = cv2.cvtColor(cur_raw, cv2.COLOR_RGB2BGR) if is_rgb else cur_raw.copy()

            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

            if backend == "ecc":
                W, score = _ecc_align(
                    ref_gray, cur_gray, motion=motion, n_levels=cfg.ecc_pyramid_levels
                )
                metrics = {"score": float(score)}
            else:
                W, inlier_ratio, n_match, n_inliers = _orb_align(ref_gray, cur_gray)
                metrics = {
                    "score": float(inlier_ratio),
                    "n_matches": int(n_match),
                    "n_inliers": int(n_inliers),
                }

            align_ok = W is not None
            if align_ok:
                dx, dy, rot_deg, scale = _decompose_similarity(W)
            else:
                dx, dy, rot_deg, scale = 0.0, 0.0, 0.0, 1.0
            metrics.update({"dx": dx, "dy": dy, "rot_deg": rot_deg, "scale": scale})

            status = _classify(metrics, cfg.thresh, short_side, backend, align_ok)

            view = _render_view(
                ref_bgr,
                cur_bgr,
                W if align_ok else None,
                metrics,
                status,
                overlay_modes[overlay_idx],
                backend,
                motion,
                cfg.display_max_height,
            )
            cv2.imshow(win_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # ESC
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                cv2.imwrite(str(diag_dir / f"snap_{ts}_cur.png"), cur_bgr)
                cv2.imwrite(str(diag_dir / f"snap_{ts}_view.png"), view)
                payload = {
                    "status": status,
                    "backend": backend,
                    "ecc_motion": motion if backend == "ecc" else None,
                    "overlay_mode": overlay_modes[overlay_idx],
                    "metrics": {
                        k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                        for k, v in metrics.items()
                    },
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(diag_dir / f"snap_{ts}_metric.json", "w") as f:
                    json.dump(payload, f, indent=2)
                logger.info("已保存诊断快照: %s", ts)
            elif key == ord("e"):
                overlay_idx = (overlay_idx + 1) % len(overlay_modes)
                logger.info("overlay -> %s", overlay_modes[overlay_idx])
            elif key == ord("b"):
                backend = backends[(backends.index(backend) + 1) % len(backends)]
                logger.info("backend -> %s", backend)
            elif key == ord("m"):
                motion = motions[(motions.index(motion) + 1) % len(motions)]
                logger.info("ecc motion -> %s", motion)
    finally:
        cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================


@parser.wrap()
def wrist_calib(cfg: WristCalibConfig) -> None:
    init_logging()
    logger.info("\n%s", pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    if not isinstance(robot, PiperRobot):
        raise ValueError(
            f"当前仅支持 piper 机器人 (--robot.type=piper)，收到 {robot.__class__.__name__}"
        )
    if cfg.camera_key not in robot.cameras:
        raise ValueError(
            f"--camera_key={cfg.camera_key} 不在 robot.cameras 中（可用键: {list(robot.cameras)}）"
        )

    # 不在连接时回零；record 要保持当前可拖动姿态，verify 由我们显式下发目标位
    robot.connect(calibrate=False)
    try:
        camera = robot.cameras[cfg.camera_key]
        if cfg.mode == "record":
            _run_record(cfg, robot, camera)
        else:
            _run_verify(cfg, robot, camera)
    finally:
        try:
            robot.bus.hold_current_position()
            time.sleep(0.5)
        except Exception as e:
            logger.warning("保位失败: %s", e)
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning("disconnect 失败: %s", e)


def main() -> None:
    # 支持把 record/verify 作为首个位置参数（自动改写为 --mode=...）
    if len(sys.argv) > 1 and sys.argv[1] in ("record", "verify"):
        sub = sys.argv.pop(1)
        sys.argv.insert(1, f"--mode={sub}")
    register_third_party_plugins()
    wrist_calib()


if __name__ == "__main__":
    main()
