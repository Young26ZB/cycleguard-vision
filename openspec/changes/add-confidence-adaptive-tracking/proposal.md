# Proposal

## Why

第二部分接收的检测框质量会随遮挡变化。需要验证置信度自适应更新能否改善定位，并按统一协议选择满足精度与时延约束的跟踪配置。

## What Changes

- 为真实检测增加可关闭的 NSA 观测噪声更新。验证数值稳定性、分数与定位误差关系，以及与 ORU 虚拟观测的隔离。
- 使用独立路口视频和相同检测缓存完成逐项消融。依据统一预算选择可部署配置，记录超过预算和缺少实测证据的情形。
- 本变更是实施顺序第 4 项，依赖 `add-ebike-tracking-core`、`add-observation-centric-association` 和 `add-occlusion-filter-replay`。验收覆盖跟踪核心和端到端视频链路。

## Capabilities

### New Capabilities

- `confidence-adaptive-tracking`：真实观测的自适应滤波与跟踪组合配置的时延和质量验收。

### Modified Capabilities

## Impact

扩展独立跟踪器的观测更新，增加组合评估配置与报告。第一部分权重和检测候选在对照间固定。正式结论依赖目标硬件和独立标注视频。
