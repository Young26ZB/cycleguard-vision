# Spec Delta

## Purpose

为电动自行车检测提供可追溯的训练标注审计和诊断评估。统一全图人车标注、整图待复核判定和无人骑行车辆的计分豁免，使同一批图片上的检测结果可以按一致规则复查。

## ADDED Requirements

### Requirement: Full-image rider and bicycle annotation
诊断标注 SHALL 检查整张图片。每辆正在骑行的电动自行车与驾驶者、所有同乘者、固定尾箱、外卖箱和挡风被共用一个 `ebike` 框。部分遮挡时应框到可可靠推断的完整范围，画面边缘的可识别实例保留并裁到图像边界。

#### Scenario: A bicycle with passengers and accessories
- **WHEN** 人行道上一辆电动自行车载有乘客，并装有固定外卖箱
- **THEN** 该实例参与标注，一个框包含车辆、驾驶者、同乘者和外卖箱

#### Scenario: An identifiable bicycle at the image edge
- **WHEN** 人车组合仍可明确识别但部分出画
- **THEN** 保留该目标，框截在原图边界

#### Scenario: A verified vehicle outside the target class
- **WHEN** 全图复核确认某实例是普通自行车、电动摩托车或快递电动三轮车
- **THEN** 不将其标为骑行目标或无人骑行电动车的忽略区域

### Requirement: Whole-image review status
诊断清单 SHALL 为每张图片保存复核状态和待复核原因。车型不确定、完整人车范围不确定或实例归属不确定时，整张图 MUST 暂停计分，直至所有原因均有复核结论。

#### Scenario: Ambiguous class or ownership
- **WHEN** 夜间无法区分普通自行车与电动自行车，或遮挡导致无法判定两个实例的人员归属
- **THEN** 记录相应原因及位置，整张图的目标和预测均不参与量化计分

#### Scenario: Unreliable full bounds
- **WHEN** 可确认是电动自行车但无法可靠确定完整人车范围
- **THEN** 记录范围不确定，整张图暂停计分

#### Scenario: Resolving a pending frame with evidence
- **WHEN** 复核者利用当前帧及可对应到同一实例的相邻帧解决车型或归属疑问，并能依据当前帧确定框范围
- **THEN** 记录复核证据和处理结论，所有原因均解除后该图可参与计分

### Requirement: Traceable training label audit
训练审计 SHALL 按数据根目录内的完整相对图片路径记录异常、证据和复核结论。缺标签图片 MUST 区分确认背景、确认漏标和待复核。零宽框 MUST 经人工复核后才能重画或移除，派生版本保留来源和已核对的同图标签。

#### Scenario: Zero-width row in an otherwise labeled image
- **WHEN** 已有标签中出现零宽框
- **THEN** 记录具体文件和行号，经复核后重画有效目标、移除误标行或保留待复核状态，其他已核对的行保留

#### Scenario: A missing training label with a negative filename
- **WHEN** 图片没有对应标签，即使文件名带 `negative_` 前缀
- **THEN** 在人工确认前保留待复核状态。确认无目标后输出显式空标签，确认漏标后补录目标框

#### Scenario: A person pushing a bicycle
- **WHEN** 训练图中人站在车旁推行电动自行车
- **THEN** 该实例按背景处理，图中的其他骑行目标继续标注

#### Scenario: Preserving the source dataset
- **WHEN** 发布经复核的训练派生版本
- **THEN** 发布检查拒绝任何仍含待复核异常、未确认的缺标签或退化框的版本。保留各集合原有成员与来源映射，原始图片和标注保持可追溯

### Requirement: Diagnostic matching with ignore regions
诊断计分 SHALL 使用原图像素 `xyxy` 框和固定的置信度、匹配 IoU、忽略 IoA 阈值，按图匹配 `ebike` 预测。先按置信度降序一对一匹配骑行目标，再识别重复检出，最后按单个已确认无人骑行电动车的忽略框处理剩余预测。IoA 的分母为预测框面积。

#### Scenario: A rider and a parked bicycle in the same image
- **WHEN** 一个预测满足骑行目标的匹配条件，另一个仅达到无人骑行车辆忽略条件
- **THEN** 骑行目标记为命中，无人骑行车辆的预测记为忽略并从精确率与召回率的计数中排除

#### Scenario: A duplicate overlaps a nearby ignore region
- **WHEN** 剩余预测达到一个已匹配骑行目标的 IoU 门槛，同时达到一个忽略框的 IoA 门槛
- **THEN** 该预测记为重复检出并计入误检

#### Scenario: An unmatched prediction or target
- **WHEN** 预测达到置信度门槛但既未匹配骑行目标也未达到忽略条件，或骑行目标没有被匹配
- **THEN** 分别计为误检或漏检

### Requirement: Reproducible diagnostic reporting
显式启用诊断模式时，评估入口 SHALL 输出精确率、召回率、重复检出次数及其占计分预测的比例、已匹配框的 `1 − IoU` 中位数及配对数。结果同时记录数据版本、权重、阈值、计分与待复核图片数、目标数、忽略预测数和逐图明细。

#### Scenario: A verified image has no targets
- **WHEN** 已复核图确认为无骑行目标
- **THEN** 未满足忽略条件的计分预测仍记为误检，汇总指标分母为零或没有定位配对时标为不适用

#### Scenario: Evaluating the current 13 images
- **WHEN** 使用当前与训练集完全重复的 13 张图运行诊断模式
- **THEN** 报告标明样本与训练集重叠，结果只用于诊断，并列出被排除的待复核图片及原因

#### Scenario: Inspecting diagnostic scene coverage
- **WHEN** 审查已复核可计分图片的场景覆盖
- **THEN** 可查询逐图的光照、真实目标数和最大真实框两两 IoU，以及逐目标的相对面积和遮挡状态，缺少证据时显式标为未知或不适用

#### Scenario: Running existing evaluation commands
- **WHEN** 用户未启用诊断模式
- **THEN** 现有验证和测试命令沿用原有计分及参数语义
