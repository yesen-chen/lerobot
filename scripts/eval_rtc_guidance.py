#!/usr/bin/env python
"""
RTC 引导质量评估脚本

核心问题：
  当 RTC 引导生成新 chunk 时，新 chunk 的前 inference_delay 步
  是否与旧 chunk 对应位置的动作误差足够小？

时序示意：
  时刻 0: 机器人从 chunk_0[action_index] 开始执行，同时触发推理
  时刻 +inference_delay: 推理完成，得到 chunk_1
  此时机器人正在执行 chunk_0[action_index + inference_delay]
  ─────────────────────────────────────────────
  RTC 目标: chunk_1[inference_delay] ≈ chunk_0[action_index + inference_delay]
           即: new_chunk[real_delay] ≈ prev_left_over[real_delay]

关键变量:
  prev_chunk_left_over = chunk_0[action_index:]   # get_left_over() 返回值
  RTC 权重覆盖 [0, execution_horizon)
  若 execution_horizon <= inference_delay → 权重在 inference_delay 处为 0 → BUG!

使用方法:
  python scripts/eval_rtc_guidance.py \
    --policy_path /home/zhang/robot/lerobot/outputs/train/2.14_307/cs64/fm/pretrained_model \
    --inference_delays 5 8 10 16 20 \
    --execution_horizons 8 16 24 32 \
    --max_guidance_weights 0.1 0.5 1.0 \
    --plot
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器时使用
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── 路径设置 ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(policy_path: str, device: str):
    """
    加载 FM Policy。

    绕过 draccus 的 argparse 冲突：直接从 config.json 构建配置，
    然后用 from_pretrained 加载权重。
    """
    import json
    from lerobot.policies.fm.configuration_fm import FlowMatchingConfig
    from lerobot.policies.fm.modeling_fm import FlowMatchingPolicy

    # 直接从 JSON 读取配置，绕过 draccus
    config_path = Path(policy_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict.pop("type", None)

    # 把 input_features / output_features 从原始字典转换为 PolicyFeature 对象
    from lerobot.configs.types import FeatureType
    from lerobot.configs.policies import PolicyFeature

    def _parse_features(feat_dict):
        result = {}
        for key, val in feat_dict.items():
            result[key] = PolicyFeature(
                type=FeatureType(val["type"]),
                shape=tuple(val["shape"]),
            )
        return result

    if "input_features" in config_dict:
        config_dict["input_features"] = _parse_features(config_dict["input_features"])
    if "output_features" in config_dict:
        config_dict["output_features"] = _parse_features(config_dict["output_features"])

    # 逐字段构建 FlowMatchingConfig（只传支持的字段）
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(FlowMatchingConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = FlowMatchingConfig(**filtered)

    # 加载模型权重
    policy = FlowMatchingPolicy.from_pretrained(policy_path, config=config)
    policy = policy.to(device)
    policy.eval()

    logger.info(
        f"Policy 加载完成: horizon={policy.config.horizon}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"n_obs_steps={policy.config.n_obs_steps}, "
        f"device={device}"
    )
    return policy


def build_dummy_batch(policy, device: str) -> dict[str, torch.Tensor]:
    """构造全零的 dummy 观测，用于推理测试（不需要真实观测）。"""
    cfg = policy.config
    n_obs = cfg.n_obs_steps

    # 机器人状态
    state_dim = list(cfg.input_features["observation.state"].shape)[0]
    state = torch.zeros(1, n_obs, state_dim, device=device, dtype=torch.float32)
    batch = {OBS_STATE: state}

    # 图像观测（若存在）
    if cfg.image_features:
        image_keys = list(cfg.image_features.keys())
        num_cams = len(image_keys)
        C, H, W = list(cfg.input_features[image_keys[0]].shape)
        # 形状: (B=1, n_obs, num_cams, C, H, W)
        images = torch.zeros(1, n_obs, num_cams, C, H, W, device=device, dtype=torch.float32)
        batch[OBS_IMAGES] = images

    return batch


def make_noise(policy, device: str, seed: int) -> torch.Tensor:
    """生成确定性噪声（固定种子）。"""
    cfg = policy.config
    action_dim = cfg.action_feature.shape[0]
    dtype = next(policy.flow_matching.parameters()).dtype

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(
        1, cfg.horizon, action_dim,
        dtype=dtype, device=device,
        generator=gen,
    )


def generate_chunk(policy, batch: dict, noise: torch.Tensor,
                   rtc_processor=None,
                   prev_chunk_left_over=None,
                   inference_delay: int = 0,
                   execution_horizon: int = 0) -> torch.Tensor:
    """
    调用 flow_matching.generate_actions() 直接生成动作块，绕过 predict_action_chunk
    中的模拟延迟 time.sleep()。

    返回: shape (n_action_steps, action_dim)，即 (64, 6) 对于 FM policy
    """
    fm = policy.flow_matching

    # 始终使用 torch.no_grad() 作为外层（节省显存）
    # RTC 的 denoise_step 内部会用 torch.enable_grad() 局部开启梯度
    # 注意：不能用 torch.inference_mode()，它会阻止 enable_grad() 生效
    ctx = torch.no_grad()

    pclover = None
    if prev_chunk_left_over is not None:
        # 确保有 batch 维度: (1, T, A)
        if prev_chunk_left_over.dim() == 2:
            pclover = prev_chunk_left_over.unsqueeze(0)
        else:
            pclover = prev_chunk_left_over

    with ctx:
        actions = fm.generate_actions(
            batch,
            noise=noise.clone(),
            rtc_processor=rtc_processor,
            inference_delay=inference_delay if rtc_processor is not None else None,
            prev_chunk_left_over=pclover,
            execution_horizon=execution_horizon if rtc_processor is not None else None,
        )

    return actions.squeeze(0)  # (n_action_steps, action_dim)


def compute_per_step_mae(chunk_new: torch.Tensor,
                         reference: torch.Tensor) -> list[float]:
    """计算 chunk_new[d] 与 reference[d] 的逐步 MAE。"""
    overlap = min(len(chunk_new), len(reference))
    return [
        (chunk_new[d] - reference[d]).abs().mean().item()
        for d in range(overlap)
    ]


def compute_per_step_l2(chunk_new: torch.Tensor,
                        reference: torch.Tensor) -> list[float]:
    """计算 chunk_new[d] 与 reference[d] 的逐步 L2 距离。"""
    overlap = min(len(chunk_new), len(reference))
    return [
        (chunk_new[d] - reference[d]).norm(p=2).item()
        for d in range(overlap)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 主测试逻辑
# ─────────────────────────────────────────────────────────────────────────────

def run_single_test(policy, batch, rtc_processor,
                    action_index: int,
                    inference_delay: int,
                    execution_horizon: int,
                    seed_chunk0: int = 42,
                    seed_chunk1: int = 99,
                    device: str = "cuda") -> dict:
    """
    执行单次 RTC 引导质量测试。

    步骤:
      1. 用 seed_chunk0 生成 chunk_0 (无 RTC) 作为参考
      2. 计算 prev_chunk_left_over = chunk_0[action_index:]
      3. 用 seed_chunk1 生成 chunk_1_no_rtc (无 RTC) 作为 baseline 对比
      4. 用 seed_chunk1 + RTC 生成 chunk_1_rtc (有 RTC 引导)
      5. 分别计算 chunk_1_no_rtc 和 chunk_1_rtc 与 prev_left_over 的逐步误差

    返回 dict，含所有误差和轨迹数据。
    """
    logger.info(
        f"\n{'='*60}\n"
        f"  action_index={action_index}, inference_delay={inference_delay}, "
        f"  execution_horizon={execution_horizon}\n"
        f"{'='*60}"
    )

    # Step 1: 生成参考 chunk_0
    noise_0 = make_noise(policy, device, seed_chunk0)
    chunk_0 = generate_chunk(policy, batch, noise_0)  # (64, 6)
    logger.info(f"chunk_0 stats: min={chunk_0.min():.3f}, max={chunk_0.max():.3f}, "
                f"mean={chunk_0.mean():.3f}")

    # Step 2: prev_chunk_left_over = chunk_0[action_index:]
    prev_left_over = chunk_0[action_index:].clone()  # (64-action_index, 6)
    logger.info(f"prev_left_over 长度: {len(prev_left_over)} steps")

    # Step 3: 生成 chunk_1_no_rtc（同种子，无引导）
    noise_1 = make_noise(policy, device, seed_chunk1)
    chunk_1_no_rtc = generate_chunk(policy, batch, noise_1)  # (64, 6)

    # Step 4: 生成 chunk_1_rtc（同种子，有 RTC 引导）
    noise_1_rtc = make_noise(policy, device, seed_chunk1)  # 相同种子
    chunk_1_rtc = generate_chunk(
        policy, batch, noise_1_rtc,
        rtc_processor=rtc_processor,
        prev_chunk_left_over=prev_left_over,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
    )

    # Step 5: 计算误差
    mae_no_rtc = compute_per_step_mae(chunk_1_no_rtc, prev_left_over)
    mae_rtc = compute_per_step_mae(chunk_1_rtc, prev_left_over)
    l2_no_rtc = compute_per_step_l2(chunk_1_no_rtc, prev_left_over)
    l2_rtc = compute_per_step_l2(chunk_1_rtc, prev_left_over)

    # 关键点：inference_delay 处的误差（切换点）
    d = min(inference_delay, len(prev_left_over) - 1)
    err_no_rtc_at_d = mae_no_rtc[d] if d < len(mae_no_rtc) else float("nan")
    err_rtc_at_d = mae_rtc[d] if d < len(mae_rtc) else float("nan")

    improvement = err_no_rtc_at_d - err_rtc_at_d
    rel_improvement = improvement / max(err_no_rtc_at_d, 1e-8) * 100

    # RTC 权重
    rtc_proc_local = RTCProcessor(rtc_processor.rtc_config)
    weights = rtc_proc_local.get_prefix_weights(
        inference_delay, execution_horizon, policy.config.horizon
    ).numpy()
    weight_at_delay = float(weights[min(d, len(weights) - 1)])

    msg = (
        f"\n  关键切换点 (step={d}) 误差:\n"
        f"    无 RTC: {err_no_rtc_at_d:.4f}\n"
        f"    有 RTC: {err_rtc_at_d:.4f}\n"
        f"    改善:   {improvement:.4f} ({rel_improvement:.1f}%)\n"
        f"    RTC 权重 @ step {d}: {weight_at_delay:.4f}\n"
        f"  {'[OK] RTC 有效改善切换连续性' if improvement > 0 else '[FAIL] RTC 未改善切换连续性 (检查参数!)'}\n"
        f"{'  [BUG] execution_horizon <= inference_delay, 权重为0!' if execution_horizon <= inference_delay else ''}"
    )
    logger.info(msg)
    print(msg, flush=True)

    return {
        "action_index": action_index,
        "inference_delay": inference_delay,
        "execution_horizon": execution_horizon,
        "weight_at_delay": weight_at_delay,
        "mae_no_rtc": mae_no_rtc,
        "mae_rtc": mae_rtc,
        "l2_no_rtc": l2_no_rtc,
        "l2_rtc": l2_rtc,
        "err_no_rtc_at_d": err_no_rtc_at_d,
        "err_rtc_at_d": err_rtc_at_d,
        "improvement": improvement,
        "rel_improvement": rel_improvement,
        "chunk_0": chunk_0.cpu().float().numpy(),
        "chunk_1_no_rtc": chunk_1_no_rtc.cpu().float().numpy(),
        "chunk_1_rtc": chunk_1_rtc.cpu().float().numpy(),
        "prev_left_over": prev_left_over.cpu().float().numpy(),
        "weights": weights,
    }


def run_sweep(policy, batch, device: str, args) -> list[dict]:
    """扫描所有参数组合，收集测试结果。"""
    all_results = []

    for exec_h in args.execution_horizons:
        for max_gw in args.max_guidance_weights:
            rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=exec_h,
                max_guidance_weight=max_gw,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            rtc_processor = RTCProcessor(rtc_config)

            for action_idx in args.action_indices:
                for infer_d in args.inference_delays:
                    result = run_single_test(
                        policy=policy,
                        batch=batch,
                        rtc_processor=rtc_processor,
                        action_index=action_idx,
                        inference_delay=infer_d,
                        execution_horizon=exec_h,
                        seed_chunk0=args.seed,
                        seed_chunk1=args.seed + 1,
                        device=device,
                    )
                    result["max_guidance_weight"] = max_gw
                    all_results.append(result)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 汇总输出
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    """以表格形式打印关键指标汇总。"""
    header = (
        f"{'exec_h':>8} {'max_gw':>8} {'delay':>6} {'ai':>4} "
        f"{'w@d':>6} {'no_rtc':>8} {'rtc':>8} {'improve%':>9} {'ok?':>5}"
    )
    sep = "=" * 75
    lines = [
        "\n" + sep,
        "RTC Guidance Quality Table  (MAE at inference_delay step)",
        "exec_h=execution_horizon, max_gw=max_guidance_weight",
        "w@d=RTC weight at delay, ok=RTC improved continuity",
        sep,
        header,
        "-" * 75,
    ]
    for r in results:
        ok = "[OK]" if r["improvement"] > 0 and r["weight_at_delay"] > 0.01 else "[FAIL]"
        bug = " BUG!" if r["execution_horizon"] <= r["inference_delay"] else ""
        lines.append(
            f"{r['execution_horizon']:>8} "
            f"{r['max_guidance_weight']:>8.2f} "
            f"{r['inference_delay']:>6} "
            f"{r['action_index']:>4} "
            f"{r['weight_at_delay']:>6.3f} "
            f"{r['err_no_rtc_at_d']:>8.4f} "
            f"{r['err_rtc_at_d']:>8.4f} "
            f"{r['rel_improvement']:>8.1f}% "
            f"{ok:>7}{bug}"
        )
    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str, flush=True)
    logger.info(table_str)


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], output_path: str = "rtc_diagnostic.png"):
    """
    生成可视化图表：
      列1 - 轨迹对比 (joint 0)
      列2 - 逐步 MAE 误差曲线
      列3 - RTC 权重分布
    """
    # 对每个 (exec_h, max_gw) 组合各绘一行
    unique_configs = sorted(set(
        (r["execution_horizon"], r["max_guidance_weight"], r["inference_delay"])
        for r in results
    ))

    n_rows = len(unique_configs)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
    fig.suptitle("RTC 引导质量评估\nnew_chunk[inference_delay] vs prev_left_over[inference_delay]",
                 fontsize=14, y=1.01)

    for row_idx, (exec_h, max_gw, infer_d) in enumerate(unique_configs):
        # 找到对应结果（取第一个 action_index）
        candidates = [
            r for r in results
            if r["execution_horizon"] == exec_h
            and r["max_guidance_weight"] == max_gw
            and r["inference_delay"] == infer_d
        ]
        if not candidates:
            continue
        r = candidates[0]
        ai = r["action_index"]

        horizon = len(r["chunk_0"])
        left_len = len(r["prev_left_over"])

        # ─── 列1: 轨迹对比 ───
        ax = axes[row_idx, 0]
        ax.plot(r["chunk_0"][:, 0], "b-", alpha=0.5, label="chunk_0 (旧)")
        ax.plot(range(ai, ai + left_len), r["prev_left_over"][:, 0],
                "g--", alpha=0.8, label=f"target (old[{ai}:])")
        ax.plot(r["chunk_1_no_rtc"][:, 0], "r-", alpha=0.6, label="chunk_1 无RTC")
        ax.plot(r["chunk_1_rtc"][:, 0], "m-", alpha=0.8, label="chunk_1 有RTC")
        ax.axvline(infer_d, color="k", linestyle=":", lw=2, label=f"delay={infer_d}")
        ax.axvline(exec_h, color="orange", linestyle=":", lw=2, label=f"exec_h={exec_h}")
        title = f"轨迹 joint0 | exec_h={exec_h}, max_gw={max_gw:.2f}, delay={infer_d}"
        if exec_h <= infer_d:
            title += "\n⚠ BUG: exec_h<=delay, 引导无效!"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("新 chunk 中的步骤")
        ax.set_ylabel("归一化动作值")

        # ─── 列2: 逐步误差 ───
        ax = axes[row_idx, 1]
        steps_no = range(len(r["mae_no_rtc"]))
        steps_rt = range(len(r["mae_rtc"]))
        ax.plot(list(steps_no), r["mae_no_rtc"], "r-o", ms=3, alpha=0.7, label="MAE 无RTC")
        ax.plot(list(steps_rt), r["mae_rtc"], "m-o", ms=3, alpha=0.9, label="MAE 有RTC")
        ax.axvline(infer_d, color="k", linestyle=":", lw=2, label=f"delay={infer_d}")
        ax.axvline(exec_h, color="orange", linestyle=":", lw=2, label=f"exec_h={exec_h}")
        ax.axvspan(0, min(infer_d, left_len), alpha=0.05, color="green",
                   label="引导覆盖区(理想)")

        # 标注切换点
        d = min(infer_d, len(r["mae_rtc"]) - 1)
        ax.annotate(
            f"切换点\nno_rtc={r['err_no_rtc_at_d']:.3f}\nrtc={r['err_rtc_at_d']:.3f}",
            xy=(d, r["err_rtc_at_d"]),
            xytext=(d + 2, r["err_rtc_at_d"] + 0.05),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        ax.set_title(
            f"逐步 MAE (越低越好)\n"
            f"@delay: 无RTC={r['err_no_rtc_at_d']:.4f}, 有RTC={r['err_rtc_at_d']:.4f} "
            f"({r['rel_improvement']:.1f}%改善)",
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.set_xlabel("步骤 d")
        ax.set_ylabel("MAE")
        ax.set_ylim(bottom=0)

        # ─── 列3: RTC 权重 ───
        ax = axes[row_idx, 2]
        ax.fill_between(range(len(r["weights"])), r["weights"],
                        alpha=0.3, color="purple")
        ax.plot(r["weights"], "purple", lw=2, label="RTC 前缀权重")
        ax.axvline(infer_d, color="k", linestyle=":", lw=2, label=f"delay={infer_d}")
        ax.axvline(exec_h, color="orange", linestyle=":", lw=2, label=f"exec_h={exec_h}")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(
            f"RTC 权重分布\n"
            f"weight@delay={r['weight_at_delay']:.4f} "
            f"({'✓有效' if r['weight_at_delay'] > 0.01 else '✗零权重=BUG'})",
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.set_xlabel("步骤")
        ax.set_ylabel("权重")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    logger.info(f"\n✓ 图表已保存至: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 权重参数影响分析（固定 inference_delay，扫描 execution_horizon & max_gw）
# ─────────────────────────────────────────────────────────────────────────────

def plot_parameter_heatmap(results: list[dict], output_path: str = "rtc_heatmap.png"):
    """绘制 execution_horizon × inference_delay 的误差热力图。"""
    # 按 max_gw 分组
    max_gws = sorted(set(r["max_guidance_weight"] for r in results))

    for mgw in max_gws:
        sub = [r for r in results if r["max_guidance_weight"] == mgw]
        exec_hs = sorted(set(r["execution_horizon"] for r in sub))
        delays = sorted(set(r["inference_delay"] for r in sub))

        if not exec_hs or not delays:
            continue

        mat_no_rtc = np.full((len(exec_hs), len(delays)), np.nan)
        mat_rtc = np.full((len(exec_hs), len(delays)), np.nan)
        mat_improve = np.full((len(exec_hs), len(delays)), np.nan)

        for r in sub:
            ei = exec_hs.index(r["execution_horizon"])
            di = delays.index(r["inference_delay"])
            mat_no_rtc[ei, di] = r["err_no_rtc_at_d"]
            mat_rtc[ei, di] = r["err_rtc_at_d"]
            mat_improve[ei, di] = r["rel_improvement"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"参数热力图 (max_guidance_weight={mgw})\n"
                     f"行=execution_horizon, 列=inference_delay, action_index=0",
                     fontsize=12)

        def _heatmap(ax, data, title, cmap, fmt=".3f", vmin=None, vmax=None):
            im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(delays)))
            ax.set_xticklabels(delays)
            ax.set_yticks(range(len(exec_hs)))
            ax.set_yticklabels(exec_hs)
            ax.set_xlabel("inference_delay")
            ax.set_ylabel("execution_horizon")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            for i in range(len(exec_hs)):
                for j in range(len(delays)):
                    v = data[i, j]
                    if not np.isnan(v):
                        color = "white" if abs(v) > 0.5 * (np.nanmax(data) or 1) else "black"
                        ax.text(j, i, format(v, fmt), ha="center", va="center",
                                fontsize=8, color=color)
            # 标记 exec_h <= delay 的非法区域
            for i, eh in enumerate(exec_hs):
                for j, d in enumerate(delays):
                    if eh <= d:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="red", lw=2
                        ))

        _heatmap(axes[0], mat_no_rtc, "无 RTC 误差@delay", "Reds")
        _heatmap(axes[1], mat_rtc, "有 RTC 误差@delay", "Reds")
        _heatmap(axes[2], mat_improve, "改善百分比(%)", "RdYlGn", fmt=".1f",
                 vmin=-100, vmax=100)
        axes[2].set_title(axes[2].get_title() + "\n(红框=exec_h<=delay BUG区)")

        plt.tight_layout()
        fname = output_path.replace(".png", f"_mgw{mgw}.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        logger.info(f"✓ 热力图已保存: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 在所有 lerobot 导入之后重新配置 root logger，以免被覆盖
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    parser = argparse.ArgumentParser(
        description="评估 RTC 引导生成的动作连续性"
    )
    parser.add_argument(
        "--policy_path",
        required=True,
        help="预训练 FM policy 路径，例如 outputs/train/xxx/pretrained_model",
    )
    parser.add_argument(
        "--inference_delays",
        type=int,
        nargs="+",
        default=[5, 8, 10, 16, 20],
        help="要测试的推理延迟步数列表（默认 fps=10, 1步=100ms）",
    )
    parser.add_argument(
        "--execution_horizons",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32],
        help="要测试的执行视野列表",
    )
    parser.add_argument(
        "--action_indices",
        type=int,
        nargs="+",
        default=[0],
        help="触发推理时已执行的步数列表",
    )
    parser.add_argument(
        "--max_guidance_weights",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="最大引导权重列表",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（chunk_0 用 seed，chunk_1 用 seed+1）",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成可视化图表",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="图表输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    logger.info(f"使用设备: {args.device}")
    logger.info(f"policy 路径: {args.policy_path}")

    # 加载 policy
    policy = load_policy(args.policy_path, args.device)

    # 构造 dummy 观测
    batch = build_dummy_batch(policy, args.device)

    # 运行全量扫描
    logger.info("\n开始 RTC 引导质量扫描...")
    all_results = run_sweep(policy, batch, args.device, args)

    # 打印汇总表
    print_summary_table(all_results)

    # 生成图表
    if args.plot:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_results(
            all_results,
            output_path=str(out_dir / "rtc_diagnostic.png"),
        )
        plot_parameter_heatmap(
            all_results,
            output_path=str(out_dir / "rtc_heatmap.png"),
        )

    # 核心结论输出
    logger.info("\n" + "=" * 75)
    logger.info("核心诊断结论:")
    logger.info("=" * 75)

    good_configs = [
        r for r in all_results
        if r["improvement"] > 0 and r["weight_at_delay"] > 0.01
    ]
    bad_configs = [
        r for r in all_results
        if r["execution_horizon"] <= r["inference_delay"]
    ]
    zero_weight = [
        r for r in all_results
        if r["weight_at_delay"] < 0.01 and r["execution_horizon"] > r["inference_delay"]
    ]

    logger.info(
        f"\n  总测试数: {len(all_results)}\n"
        f"  RTC 有效改善切换点: {len(good_configs)} / {len(all_results)}\n"
        f"  BUG (exec_h <= delay, 权重=0): {len(bad_configs)} 个\n"
        f"  权重接近0但无 BUG: {len(zero_weight)} 个\n"
    )

    if bad_configs:
        logger.warning(
            "⚠ 检测到 BUG 配置! 以下 execution_horizon <= inference_delay:\n" +
            "\n".join(
                f"    exec_h={r['execution_horizon']}, delay={r['inference_delay']}"
                for r in bad_configs[:5]
            )
        )
        logger.warning(
            "  → RTC 权重在切换点为 0，引导完全失效，直接导致动作不连续/回抽！\n"
            "  → 修复: 必须设置 execution_horizon > inference_delay"
        )

    # 找最优配置
    if good_configs:
        best = max(good_configs, key=lambda r: r["rel_improvement"])
        logger.info(
            f"\n  最优配置:\n"
            f"    execution_horizon={best['execution_horizon']}\n"
            f"    inference_delay={best['inference_delay']}\n"
            f"    max_guidance_weight={best['max_guidance_weight']}\n"
            f"    切换点误差改善: {best['rel_improvement']:.1f}%\n"
            f"    (无RTC={best['err_no_rtc_at_d']:.4f} → 有RTC={best['err_rtc_at_d']:.4f})"
        )

    return all_results


if __name__ == "__main__":
    main()
