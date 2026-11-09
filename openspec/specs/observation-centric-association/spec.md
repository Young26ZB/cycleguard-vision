# observation-centric-association Specification

## Purpose

在第二部分的跟踪主链路中加入基于真实观测的方向关联与短时找回，用于处理停车起步、交叉和预测框漂移，并限制密集场景下额外的关联耗时。

## Requirements

### Requirement: Observation direction in primary association

启用 OCM 时，主关联 SHALL 用历史真实检测框的中心位移方向与候选方向的一致性辅助预测框 IoU，并支持固定权重和按有效位移与历史方向稳定性调权的配置。坐标使用原图像素，方向项 MUST 在分配前计算，且不得使未通过主几何门控的配对变为有效。历史不足、位移过小或历史过旧时 SHALL 省略方向项。

#### Scenario: Direction separates otherwise similar candidates

- **WHEN** 两个已通过几何门控的候选具有相同 IoU，其中一个与有效观测方向更一致
- **THEN** 方向一致的候选获得更优的关联代价，已匹配检测不进入其他轨迹的后续分配

#### Scenario: A stopped track has insufficient displacement

- **WHEN** 两次历史实测的中心位移低于配置门槛或方向不可定义
- **THEN** 该轨迹按主链路几何代价参与匹配，方向项不产生非有限值，也不阻止其继续匹配

### Requirement: Last-observation recovery

启用 OCR 时，跟踪器 SHALL 在主匹配后使用最近真实观测框，对未匹配的已确认或丢失轨迹与剩余合格检测进行一次有门控的一对一恢复。恢复窗口以距最近真实观测的帧数计，MUST 受主链路丢失保留期约束。

#### Scenario: A short occlusion causes prediction drift

- **WHEN** 主匹配未找到候选，但窗口内的一条未匹配轨迹的最近真实框与一个剩余检测通过恢复门控并成功分配
- **THEN** 该检测沿用原 ID，并标明匹配来自 OCR，原来已成功的匹配保持有效

#### Scenario: An expired or already matched track is excluded

- **WHEN** 某轨迹已匹配、已移除、仍待确认或超过 OCR 窗口
- **THEN** OCR 不用该轨迹竞争剩余检测

### Requirement: Bounded association overhead

OCM 和 OCR SHALL 可独立关闭。系统 SHALL 限制真实观测历史长度及 OCR 的候选行列数，使用主链路候选上限约束主关联。OCR MUST 只对剩余有效候选执行一次，按可复现规则处理超限并统计被跳过数量。

#### Scenario: Primary association consumes all candidates

- **WHEN** 主关联后任一侧剩余候选为空
- **THEN** 跳过 OCR 的代价矩阵与求解

#### Scenario: Dense recovery exceeds its configured bounds

- **WHEN** 剩余轨迹和检测数超过 OCR 容量
- **THEN** 二次矩阵不超过配置上限，跳过项保持未匹配，并保留容量截断计数

#### Scenario: Compare association profiles

- **WHEN** 在同一缓存、硬件和基准配置上分别测量关闭、单独开启和同时开启 OCM/OCR
- **THEN** 关闭两项时结果与主链路一致。报告各组的 P95/P99、候选规模与 OCR 调用次数，超出跟踪预算的配置不能标为实时配置
