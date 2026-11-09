# 电动自行车跟踪主链路

从仓库根目录运行，使用已安装本项目依赖的 Python 环境。缓存回放和合成检查无需权重或训练。

## 回放检测缓存

```powershell
python -m tracker.track_ebike --cache tracker/example_detections.jsonl --output tracks.jsonl
```

版本 1 检测缓存是 UTF-8 JSONL。首行包含 `schema_version`、`source_id`、`fps` 和原图 `width`、`height`。其后每行是 `frame_id`、`timestamp_s` 和 `detections`。帧号从 0 连续增长，时间戳从 0 开始并与固定帧率对齐。每个检测是 `[x1, y1, x2, y2, score, class_id]`。框为原图像素，空帧写 `[]`。带框 MP4 用于查看结果，回放需要对应的结构化缓存。

## 原始视频推理和跟踪

```powershell
python -m tracker.track_ebike --video input.mp4 --weights best.pt --device cpu --output video-tracks.jsonl --save-cache detections.jsonl
```

输入为一段可通过解码时间戳验证的固定帧率原始视频和第一部分的本地 NMS-Free 权重。每帧输出来源、帧号、时间戳、当帧有实测支持的已确认轨迹及统计。轨迹记录包括 `track_id`、`class_id`、滤波后的像素框 `bbox_xyxy` 和指向该帧原始检测列表的零基 `detection_index`。启用 `--debug-predictions` 时，丢失轨迹的预测放在独立的 `predictions` 中。各输出文件使用新路径。输入中途无效时命令返回失败，已写出的有效前缀可能保留。

主链路按 `--min-score` 筛选候选，并按置信度及输入顺序截断到 `--max-detections`。新增轨迹受 `--new-track-score` 和 `--max-tracks` 限制。已确认和未到期丢失轨迹优先分配，通过 `--min-iou` 和类别门控。新轨迹连续命中 `--min-hits` 后确认，未匹配的待确认轨迹当帧移除。最近实测距当前帧超过 `--max-age` 的轨迹在关联前移除。用 `--help` 查看全部参数。

报告中的关联行列数对应真实候选矩阵。发生竞争时，求解器在最多 `N × (M + N)` 的空间内分配，虚拟列允许轨迹不匹配，其中 `N` 和 `M` 分别是本轮实际轨迹和检测的数量。批量卡尔曼校正保留与逐轨校正相同的状态方程。

## 观测中心关联

```powershell
python -m tracker.track_ebike --cache tracker/example_detections.jsonl --ocm adaptive --ocr --output associated.jsonl
```

OCM 和 OCR 默认均关闭，输出保持主链路格式。启用任一项后，轨迹增加 `match_source` 字段，值为 `primary`、`ocr` 或 `new`。每帧的 `association` 记录有效方向参与的轨迹数、OCR 执行与跳过次数、矩阵行列数和截断数。

- OCM 只在通过 IoU 和类别门控的竞争配对上计算方向代价。默认保留 6 次真实检测中心，优先使用距最近实测 3 帧的参考点，目标间隔不足时用较短的有效间隔。参考点或最近实测过旧、位移不足 2 像素时，沿用几何评分。分别用 `--ocm-history-size`、`--ocm-interval`、`--ocm-max-age` 和 `--ocm-min-displacement` 调整。固定权重默认 0.2。自适应模式按位移和历史方向一致性降低同一上限下的权重。
- OCR 只对主匹配后的剩余检测和可找回轨迹使用最近一次真实框。默认窗口为 6 帧，且受主链路存活期限制。候选默认最多 64 条轨迹和 128 个检测。超限时优先保留较新的轨迹和较高分的检测，同等条件按 ID 和输入顺序处理。可用 `--ocr-max-age`、`--ocr-min-iou`、`--ocr-max-tracks` 和 `--ocr-max-detections` 调整。

方向一致性与最近观测恢复参考 [OC-SORT](https://arxiv.org/html/2203.14360)。本实现保留主链路的 XYAH 滤波和一对一关联。

## 性能测量

```powershell
python -m tracker.benchmark --output tracking-benchmark.json
python -m tracker.benchmark --association --frames 270 --warmup 27 --repeat 3 --output association-benchmark.json
python -m tracker.track_ebike --video input.mp4 --weights best.pt --device 0 --output measured-tracks.jsonl --report measured-video.json --warmup 30 --target-fps 30
```

合成基准用普通、拥挤和超容量框序列测量跟踪核心。默认每个场景预热 30 帧，测 240 帧，独立重复三次。核心计时从检测记录就绪开始，到轨迹结果就绪结束。缓存回放的 `--report` 也仅报告跟踪核心。两者的 `end_to_end_latency_ms` 为 `null`。

视频模式使用 `--report` 时，按 `--target-fps` 或视频 FPS 的节奏安排帧到达。端到端计时从计划到达开始，到轨迹可用结束，含等待、解码、检测和适配。JSONL 写出与最后缓冲刷出分别统计，前一帧的同步写出可能影响后续等待。报告包含 P50/P95/P99、吞吐、起处理时积压、超期、工作量与内存。分位数使用预热后前 `--profile-frames` 个样本，整段仍逐帧处理并汇总负载。视频成功完整解码才出具报告。进程 RSS 峰值含运行时和已加载依赖，分场景 RSS 另按固定间隔采样。

[本机合成基线](local_synthetic_baseline.json)使用单线程 BLAS。普通、密集和超容量场景三次运行的 P95 中位数分别为 0.66 ms、2.79 ms 和 3.17 ms。密集场景含 128 个检测和 128 条轨迹。目标硬件的端到端实时性需使用真实视频和固定权重另行验收。

在[关联消融报告](local_association_baseline.json)中，六种配置使用相同的合成检测序列，单列首次重现帧的 P95/P99 和 OCR 工作量。128 目标的持续密集场景中，关闭增强、固定 OCM 加 OCR、自适应 OCM 加 OCR 的跟踪 P95 中位数分别为 3.03 ms、4.07 ms 和 4.34 ms。重现序列中 OCR 处理的最大真实候选矩阵为 64 × 128。

## 本地非训练检查

```powershell
python -m unittest tests.test_ebike_tracking -v
python -m unittest tests.test_observation_association -v
```

本地验证使用 Python 3.9 与 PyTorch 2.5.1 CPU。17 项跟踪合成检查和 7 项第一阶段合成回归通过。另用 4 帧合成视频及未训练的端到端权重完成视频入口、检测缓存回放和按帧节奏计时的联调，直接推理和缓存回放的轨迹文件逐字节一致。
