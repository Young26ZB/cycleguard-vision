# Development Workflow

## Purpose

为检测、非线性跟踪和神经符号违章推理三部分研究建立可复现的规范开发流程。通过可核查的需求和场景，明确区分代码实现、本地非训练检查及阿里云实验，避免将计划中的能力或未执行的评估当作已完成成果。

## Requirements

### Requirement: Project-local OpenSpec tooling

项目 MUST 使用锁定版本的 OpenSpec 开发依赖，并提供项目级命令入口。

#### Scenario: Restore development tooling

- **WHEN** 开发者在已安装兼容 Node.js 的机器上执行 npm ci
- **THEN** 根据 package-lock.json 恢复 OpenSpec，并可通过 npm run openspec -- --version 检查版本

### Requirement: Separate local checks from cloud training

开发流程 MUST 遵守用户的本机不训练约束，并区分检查结果和训练实验结果。

#### Scenario: Run local verification

- **WHEN** 本地验证检测代码且用户没有更新约束
- **THEN** 只执行静态检查或无优化器更新的合成数据检查，真实数据训练留待阿里云

#### Scenario: Report pending cloud verification

- **WHEN** 某项 CUDA 检查或真实数据实验尚未执行
- **THEN** 任务与报告明确标记待验证，不声明其通过或带来性能提升

### Requirement: Traceable research changes

研究功能变更 SHALL 关联可核查的需求场景、技术设计和实施任务。

#### Scenario: Plan a research capability

- **WHEN** 创建检测、跟踪或违章推理的新功能变更
- **THEN** 指明所属研究部分，记录范围、验收依据和待验证假设
