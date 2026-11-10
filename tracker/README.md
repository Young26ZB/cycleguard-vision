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

## 遮挡滤波回放

```powershell
python -m tracker.track_ebike --cache tracker/example_detections.jsonl --ocr --oru --output replayed.jsonl
```

ORU 默认关闭，可与主关联或 OCR 搭配。每条轨迹保留一个最近真实观测及其更新后的滤波快照。丢失轨迹重新匹配后，从快照逐帧回放遮挡区间的 XYAH 线性插值，再使用现有批量更新融合重现帧的真实检测。虚拟观测用于滤波状态修复，已发布的结果和真实观测历史保持原样。

- `--oru-max-missing` 默认 6，限制单轨可回放的缺失帧数。
- `--oru-max-steps` 默认 16，限制整帧的虚拟更新尝试步数。先处理缺失帧较少的轨迹，同长度按轨迹 ID 排序。以每轨缺失 4 帧为例，默认预算最多回放 4 条轨迹。
- `--oru-noise-scale` 默认 4.0。虚拟观测使用放大后的基础观测协方差，每维方差有正下限，独立于检测置信度。

预算不足、快照缺失或回放状态无效时，沿用普通预测和已匹配的 ID，当帧完成真实更新。失败时已尝试的步骤计入预算。开启后的帧级 `oru` 记录请求、成功、尝试步数及各类回退。

## 置信度自适应观测更新

```powershell
python -m tracker.track_ebike --cache tracker/example_detections.jsonl --ocm adaptive --ocr --oru --nsa --output adaptive-tracks.jsonl
```

NSA 默认关闭。启用后，对已匹配真实框按 `max(min_scale, 1 - score)` 缩放基础观测噪声，`--nsa-min-scale` 默认 0.1，各维方差另有正下限。计算按匹配批量执行，ORU 虚拟观测仍使用独立的倍率。遇到数值异常，受影响的轨迹用普通噪声重试。普通更新仍无效时，保留有限预测并按当帧未匹配处理。帧级 `nsa` 和报告记录普通更新回退次数、重试后仍失败次数。报告中的 `real_updates` 统计匹配后成功提交的真实更新。

## 离线质量消融与部署筛选

`tracker.evaluate` 按同一批检测缓存运行八种配置。包括基线 `baseline`、仅 NSA `nsa_only`、全开 `full`、逐项关闭 `without_ocm`、`without_ocr`、`without_oru`、`without_nsa` 和全开但使用固定方向权重的 `fixed_ocm`。

清单提供 `weights_sha256` 和 `sequences`。每条序列提供 `id`、`split`（`validation` 或 `test`）、`cache` 和 `truth`。路径相对于清单文件。每段原始视频只归入一个划分。真值 JSONL 与缓存逐帧对齐，示例如下。

```json
{"frame_id":0,"visible":[[15,26,40,74,138,0]],"occluded":[12],"partial":[15]}
```

`visible` 每项为 `[跨帧ID, x1, y1, x2, y2, class_id]`，`occluded` 为有证据确认全遮挡的 ID，`partial` 为仍有真值框的部分遮挡 ID。一次事件从可见帧开始，经连续显式全遮挡后在首次可见帧结算。报告按遮挡帧数分组，并分别统计全部事件与重现帧存在合格检测的子集。合格检测须通过分数、类别及 IoU 门槛，和真值一对一匹配。把遮挡前的轨迹 ID 接到其他目标，或接回曾属于其他目标的 ID，均计为错连。分数校准按清晰、部分遮挡和重现分别统计匹配样本的平均 `1 - IoU` 及检测漏失数。无样本的比率为 `null`。

协议 JSON 必须给出 `iou`、遮挡帧数分组上界 `occlusion_bins`、`target_fps`、`tracking_p95_ms` 和 `quality_thresholds`。门槛字段为 `min_idf1`、`min_assa`、`min_same_id_rate`、`max_wrong_reconnect_rate`（均为 0 至 1 的小数）及 `max_id_switches`。可选 `tracking`、`ocm`、`ocr`、`oru` 和 `nsa_min_scale` 使用对应配置类的参数名。先用验证视频确定参数与门槛，再保存最终验证报告。测试命令检查协议、缓存、标注、权重标识和各配置的快照是否与验证报告一致。

```powershell
python -m tracker.evaluate --manifest sequences.json --protocol protocol.json --split validation --output validation.json
python -m tracker.evaluate --manifest sequences.json --protocol protocol.json --split test --validation-report validation.json --output test.json
```

标准 IDF1、AssA 和 ID 切换由 [TrackEval](https://github.com/JonathonLuiten/TrackEval) 计算。验证和测试使用同一个兼容 NumPy 的 TrackEval 环境。两条命令均追加 `--events-only` 可单独检查事件与检测分桶，适合本地合成联调。

部署筛选在测试命令中追加 `--timing-manifest timings.json`。先在同一目标设备和线程设置下用原始视频逐配置运行 `tracker.track_ebike --report`，启用参数须与冻结快照一致。每次使用同一个目标帧率、预热帧数及采样上限。计时清单填写报告中的 `environment`、`device_argument`、`warmup_frames`、`sample_limit` 和 `profiles`，其中 `profiles` 的格式为 `{"配置名":{"序列ID":"计时报告路径"}}`。报告路径相对于计时清单。筛选要求每条测试视频完整处理、无掉帧和结尾积压、实际吞吐达到目标，并同时满足跟踪 P95 与端到端 P95 预算。对通过质量和时延门槛的配置，以各自最慢测试视频的端到端 P95 排序。

## 性能测量

```powershell
python -m tracker.benchmark --output tracking-benchmark.json
python -m tracker.benchmark --association --frames 270 --warmup 27 --repeat 3 --output association-benchmark.json
python -m tracker.benchmark --oru --frames 270 --warmup 27 --repeat 3 --output oru-benchmark.json
python -m tracker.benchmark --nsa --frames 270 --warmup 27 --repeat 3 --output adaptive-benchmark.json
python -m tracker.track_ebike --video input.mp4 --weights best.pt --device 0 --output measured-tracks.jsonl --report measured-video.json --warmup 30 --target-fps 30
```

合成基准用普通、拥挤和超容量框序列测量跟踪核心。默认每个场景预热 30 帧，测 240 帧，独立重复三次。核心计时从检测记录就绪开始，到轨迹结果就绪结束。缓存回放的 `--report` 也仅报告跟踪核心。两者的 `end_to_end_latency_ms` 为 `null`。

视频模式使用 `--report` 时，按 `--target-fps` 或视频 FPS 的节奏安排帧到达。端到端计时从计划到达开始，到轨迹可用结束，含等待、解码、检测和适配。JSONL 写出与最后缓冲刷出分别统计，前一帧的同步写出可能影响后续等待。报告包含 P50/P95/P99、吞吐、起处理时积压、超期、工作量与内存。分位数使用预热后前 `--profile-frames` 个样本，整段仍逐帧处理并汇总负载。视频成功完整解码才出具报告。进程 RSS 峰值含运行时和已加载依赖，分场景 RSS 另按固定间隔采样。

[本机合成基线](local_synthetic_baseline.json)使用单线程 BLAS。普通、密集和超容量场景三次运行的 P95 中位数分别为 0.66 ms、2.79 ms 和 3.17 ms。密集场景含 128 个检测和 128 条轨迹。目标硬件的端到端实时性需使用真实视频和固定权重另行验收。

在[关联消融报告](local_association_baseline.json)中，六种配置使用相同的合成检测序列，单列首次重现帧的 P95/P99 和 OCR 工作量。128 目标的持续密集场景中，关闭增强、固定 OCM 加 OCR、自适应 OCM 加 OCR 的跟踪 P95 中位数分别为 3.03 ms、4.07 ms 和 4.34 ms。重现序列中 OCR 处理的最大真实候选矩阵为 64 × 128。

[ORU 消融报告](local_oru_baseline.json)在相同的 128 目标合成检测序列和 OCM/OCR 配置下对比开关。三次运行中，全序列 P95 中位数从 3.74 ms 变为 4.91 ms，重现帧 P95 中位数从 2.89 ms 变为 5.55 ms。默认预算下，每次运行的 1280 个请求有 70 个完成回放，1210 个因整帧预算走普通更新。可使用 `--oru-max-steps` 复测其他预算。

NSA 消融在本机 CPU 上复用同一份内存检测缓存，对比 128 目标持续密集和周期同步重现。每组预热 27 帧后测量 270 帧，独立重复三次。持续密集场景下，基线和仅 NSA 的跟踪 P95 中位数分别为 3.67 ms 和 3.84 ms。同步重现场景下，全开配置和关闭 NSA 的重现帧 P95 中位数分别为 5.99 ms 和 5.76 ms。各组未出现 NSA 数值回退。报告保留各次 P50/P95/P99、重现帧延迟、跟踪吞吐、回退计数与 RSS。[本机合成报告](local_adaptive_baseline.json)

## 本地非训练检查

```powershell
python -m unittest tests.test_ebike_tracking -v
python -m unittest tests.test_observation_association -v
python -m unittest tests.test_occlusion_filter_replay -v
python -m unittest tests.test_confidence_adaptive_tracking tests.test_tracking_evaluation -v
```

本地验证使用 Python 3.9 与 PyTorch 2.5.1 CPU。17 项跟踪合成检查和 7 项第一阶段合成回归通过。另用 4 帧合成视频及未训练的端到端权重完成视频入口、检测缓存回放和按帧节奏计时的联调，直接推理和缓存回放的轨迹文件逐字节一致。
