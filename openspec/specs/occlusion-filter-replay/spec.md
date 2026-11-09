# occlusion-filter-replay Specification

## Purpose

对遮挡后已成功找回身份的电动自行车轨迹执行在线滤波回放，限制连续缺测造成的状态漂移，并控制多车同时重现造成的单帧额外工作量。

## Requirements

### Requirement: Replay only after identity recovery

启用 ORU 时，跟踪器 SHALL 仅在丢失轨迹已匹配当前真实检测且存在中间缺测帧时回放。它 SHALL 从上次真实更新后的状态与协方差恢复，按帧顺序处理两次原图像素观测之间的线性虚拟观测，再融合当前真实检测一次。

#### Scenario: A track is recovered after three missing frames

- **WHEN** 第 20 帧最后实测，第 21 至 23 帧无匹配，第 24 帧由主关联或 OCR 找回原轨迹且预算允许
- **THEN** ORU 只回放第 21 至 23 帧的虚拟观测，第 24 帧仅做一次真实观测更新，输出原 ID

#### Scenario: There is no newly recovered track

- **WHEN** 轨迹仍失配、逐帧连续命中，或者 ORU 被关闭
- **THEN** 不启动回放，已匹配轨迹沿用普通更新路径

### Requirement: Preserve observation provenance

虚拟观测 SHALL 仅用于滤波回放。它们 MUST 使用独立于真实检测置信度的噪声规则，不得写入真实观测历史、替代真实检测或改写已发布的帧结果。

#### Scenario: A replay completes successfully

- **WHEN** ORU 使用虚拟观测恢复当前滤波状态
- **THEN** 已发布的缺测帧结果保持原样。当前输出引用重现帧的真实检测，后续方向历史只包含真实观测

### Requirement: Bound replay cost and fall back atomically

ORU SHALL 配置单轨最大缺帧数及整帧虚拟更新步数上限，每条轨迹只保留一个最近真实更新后的快照。超限、快照缺失或回放产生无效状态时，MUST 放弃该轨迹的回放结果，使用当帧正常预测及真实检测完成更新，保持已匹配的 ID。

#### Scenario: Multiple tracks reappear in a crowded frame

- **WHEN** 多条已找回轨迹请求的虚拟更新步数之和超过整帧上限
- **THEN** 系统按确定顺序分配回放额度，尝试的虚拟更新总步数不超过上限。无法完整回放的轨迹在当前帧走普通更新，跳过原因计入统计

#### Scenario: A replay fails midway

- **WHEN** 某次回放在部分虚拟更新后产生无效状态
- **THEN** 丢弃该次临时状态，已用步数仍计入整帧上限。该轨迹以普通更新输出，不留下部分回放的状态

#### Scenario: Benchmark simultaneous recoveries

- **WHEN** 同一批同步重现输入以相同硬件分别运行 ORU 开与关
- **THEN** 报告重现帧和全序列的 P95/P99、回放步数与回退次数。超过主链路预算的配置不得标记为实时配置
