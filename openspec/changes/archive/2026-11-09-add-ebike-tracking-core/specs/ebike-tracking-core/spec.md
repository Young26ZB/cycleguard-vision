# Spec Delta

## Purpose

建立单路电动自行车视频的在线跟踪基础能力，将 NMS-Free 逐帧检测转为带身份的轨迹结果，并提供可重复的时延与容量基线供遮挡恢复等后续能力使用。

## ADDED Requirements

### Requirement: Frame-aligned detections and replay

跟踪入口 SHALL 将原始视频推理与结构化检测缓存统一为同一逐帧输入。每帧包含来源标识、帧号、视频时间戳（秒）和有效的原图像素 `xyxy` 框、置信度及类别。首版 SHALL 支持固定帧率、连续帧、单路视频，包含零检测帧。

#### Scenario: Replay produces the same tracks as direct inference

- **WHEN** 同一视频的直接推理适配器和缓存回放按相同次序提供相同检测记录及配置
- **THEN** 两种入口产生相同的逐帧轨迹与检测对应关系，缓存回放无需加载模型

#### Scenario: Reject an unaligned input

- **WHEN** 提供仅渲染了检测框的视频作为检测缓存，或者同一来源的缓存帧号重复、缺帧、时间戳非递增
- **THEN** 入口明确指出输入问题，在该帧更新轨迹前停止。合法空帧须显式记录为空列表

### Requirement: Online track assignment and lifecycle

跟踪器 SHALL 逐帧预测并以预测框的 IoU 作门控的一对一关联。每个检测与轨迹在该帧最多匹配一次，并允许拒绝所有候选。轨迹 SHALL 经历待确认、已确认、丢失和移除，视频切换时清空旧轨迹池。

#### Scenario: Confirm and recover an identity

- **WHEN** 某轨迹达到连续确认命中阈值，随后出现若干空检测帧，且在丢失保留窗口内与新检测成功匹配
- **THEN** 跟踪器确认并沿用该 ID，空帧仅推进预测与丢失计时

#### Scenario: Expired tracks cannot claim a detection

- **WHEN** 已确认轨迹距最近实测的帧数超过配置上限，或待确认轨迹首次失配
- **THEN** 跟踪器在相应时机移除该轨迹。超时轨迹不得参与当帧匹配，后续高分检测可初始化新 ID

### Requirement: Observed frame output

标准输出 SHALL 每帧提供来源、帧号、时间戳及当帧有真实检测支持的已确认轨迹。每条结果包含 ID、类别、原图像素框及其匹配检测索引。调试输出若包含预测，MUST 明确标识其来源。

#### Scenario: No observation is present

- **WHEN** 已确认轨迹在当前帧没有匹配检测
- **THEN** 该轨迹不出现在当帧标准结果中。若当帧无其他已确认匹配，仍输出一条空轨迹列表的帧记录

### Requirement: Bounded work per frame

跟踪器 SHALL 具有可配置的检测候选与活跃轨迹池容量，并在关联前限制矩阵行列。超过容量时 MUST 按可复现规则取舍并记录数量。回放和在线入口 SHALL 逐帧推进，避免积累完整视频、检测结果或轨迹历史。

#### Scenario: A dense frame exceeds configured capacities

- **WHEN** 检测数或新轨迹请求超过配置容量
- **THEN** 关联矩阵尺寸和轨迹池大小均未超过上限。报告区分候选被截断与新轨迹因容量未创建，且相同输入产生相同取舍

#### Scenario: A long stream reaches steady state

- **WHEN** 相同负载的视频运行长度增加
- **THEN** 轨迹状态、历史及待处理缓冲的占用受配置上限约束，不随已处理帧数线性增长

### Requirement: Reproducible tracking latency baseline

基准 SHALL 在固定硬件和相同输入上分别测量跟踪核心及端到端链路的 P50、P95、P99 时延，记录吞吐、帧间积压或超期、输入规模、容量截断、峰值内存和配置。跟踪核心从当帧数据已可用计至轨迹产生。端到端从帧到达计至轨迹可供下游读取，包含解码、检测和适配。可视化和文件写出开销 SHALL 单列。

#### Scenario: Benchmark ordinary and dense cached inputs

- **WHEN** 回放同一普通负载和密集负载，并运行本地不训练的合成基准
- **THEN** 两组均产生可重跑的跟踪时延统计与工作量记录。缓存回放不得作为检测加跟踪的端到端实测

#### Scenario: Assess the target real-time profile

- **WHEN** 使用声明的目标设备、持续负载与目标帧率进行完整链路验收
- **THEN** 跟踪 P95 须在事先登记的预算内，端到端 P95 须不超过一个目标帧周期，且持续吞吐须达到目标帧率。报告 P99、丢帧与积压。跳帧、持续积压、超预算或缺少目标环境测量时，不得报告该配置通过完整帧实时性验收
