# Proposal

## Why

第二部分在遮挡后可能已找回旧 ID，但滤波状态仍受此前连续预测影响，下一帧容易再次失配。需要独立检验基于观测的滤波回放。

## What Changes

- 在旧轨迹成功重关联后，用两端真实观测构造缺失帧的线性虚拟观测，回放滤波状态并融合当帧实测。
- 为单轨与整帧回放设置硬上限，验证集体重现时可按确定规则回退到普通更新。
- 本变更是实施顺序第 3 项，依赖 `add-ebike-tracking-core` 和 `add-observation-centric-association`。用合成缺帧验证回放触发、同 ID 和单次真实更新。线性插值用于修复滤波状态。

## Capabilities

### New Capabilities

- `occlusion-filter-replay`：失而复得轨迹的有界在线滤波回放。

### Modified Capabilities

## Impact

扩展滤波更新与轨迹状态，增加每轨一个可替换快照。使用相同检测缓存比较 ORU 开关对重现后效果及尾延迟的影响。
