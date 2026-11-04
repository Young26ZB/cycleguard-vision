# 项目开发约定

本仓库后续研究功能与算法变更使用 OpenSpec。先阅读 `openspec/config.yaml`、相关 `openspec/specs/` 和 `docs/detection/STAGE1.md`，按用户请求选择适用的 OpenSpec 工作流；只读分析、工具安装和微小维护不必创建功能提案。

- 官方 Codex 工作流在 `.agents/skills/openspec-*/SKILL.md`，项目使用本地 CLI。工作流中的 `openspec ...` 在此仓库执行为 `npm run openspec -- ...`；PowerShell 若阻止 npm.ps1，使用 `npm.cmd run openspec -- ...`。安装依赖用 `npm ci`。
- 用中文编写项目说明和规划；规范结构标题、SHALL/MUST、WHEN/THEN 保持英文以便校验。
- 当前重点是第一部分 NMS-Free 检测。第二部分非线性跟踪、第三部分神经符号违章推理的设计文档不代表功能已实现。
- 用户要求本机不训练。可以执行静态检查和无优化器更新的合成张量检查；真实数据训练与 CUDA 验证留给阿里云，除非用户更新约束。
- 不自动修改数据标注、删除历史权重或覆盖实验。不得把历史基础模型成绩当作新损失或增强模块的效果证明。
- 项目非训练检查：`python -m unittest tests.test_stage1 -v`。依赖有效的 PyTorch 环境；不要使用缺少依赖的默认解释器推断代码有错。
- OpenSpec 校验：`npm run openspec:check`。仅在有对应证据时勾选完成任务；云端未执行的任务保持未完成。

快速入门和命令示例见 `docs/development/OPENSPEC_GUIDE.md`。执行范围以用户的明确授权为准。
