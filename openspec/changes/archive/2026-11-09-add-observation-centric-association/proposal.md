# Proposal

## Why

第二部分的预测框 IoU 主链路在停走、交叉和短遮挡时可能因预测漂移而丢失原 ID。需要独立检验真实观测对关联的帮助及额外耗时。

## What Changes

- 加入 OCM 方向代价和 OCR 二次关联，分别可关闭。用合成案例验证方向有效、方向无效和预测框漂移时的行为。
- 限制方向历史、候选规模及 OCR 工作量，分别测量功能开启前后的尾延迟。
- 本变更是实施顺序第 2 项，依赖 `add-ebike-tracking-core` 的输入输出、轨迹池和性能基准。匹配结果供后续 ORU 使用。低分框救援、外观特征和转弯动力学另行评估。

## Capabilities

### New Capabilities

- `observation-centric-association`：基于真实观测的方向辅助关联与短时未匹配轨迹恢复。

### Modified Capabilities

## Impact

扩展主链路的关联和轨迹历史。沿用同一输入输出契约和检测缓存。复杂路口中的 ID 改善幅度由真实视频评估确定。
