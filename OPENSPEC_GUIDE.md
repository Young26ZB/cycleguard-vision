# OpenSpec 项目使用说明

已安装项目本地 OpenSpec 1.16.0，使用官方 spec-driven schema，生成 Codex 的六个核心技能。Node.js 要求 >=22.12.0；开发依赖不会加入 Python 运行或模型推理依赖。

## 安装与校验

从项目根目录执行：

```bash
npm ci
npm run openspec -- --version
npm run openspec:check
npm run openspec -- list --specs
```

Windows PowerShell 如果限制 npm.ps1，使用 npm.cmd 替代 npm。无需安装全局 openspec；官方技能中的 openspec 命令，在本项目写成 npm run openspec -- 后跟原参数。

## 在 Codex 中使用

重新打开项目或重启应用，让新增技能被发现。在桌面版的 Skills 中选择相应技能；在支持技能文本调用的界面中可用下面的形式。

| 技能 | 用途 | 例子 |
|---|---|---|
| openspec-explore | 调查代码并讨论方案 | `$openspec-explore 分析高密度检测测试集应如何构建` |
| openspec-propose | 生成提案、规格、设计和任务 | `$openspec-propose 建立电动自行车拥堵场景评估集` |
| openspec-apply-change | 实施一个已规划的变更 | `$openspec-apply-change <change-name>` |
| openspec-update-change | 根据新信息调整计划 | `$openspec-update-change <change-name>` |
| openspec-sync-specs | 将变更规范同步到主规范 | `$openspec-sync-specs <change-name>` |
| openspec-archive-change | 归档已完成的变更 | `$openspec-archive-change <change-name>` |

propose 工作流只产出规划，不在同一流程中自动实现。希望执行时，再明确使用 apply。具体可用的技能显示形式以当前界面为准；也可直接用自然语言点名相应工作流。

## 项目结构

```text
AGENTS.md                         仓库开发约定与本地命令入口
package.json / package-lock.json  锁定 OpenSpec 开发工具版本
.agents/skills/openspec-*/        官方生成的 Codex 工作流
openspec/config.yaml              中文输出、三部分研究背景与约束
openspec/specs/                   当前规范基线
  development-workflow/spec.md    开发流程与验证约束
  nms-free-detection/spec.md       第一阶段已实现的检测行为
openspec/changes/                 后续独立变更
  archive/                        完成后归档
```

本次只建立工具和现有能力基线，没有创建新的功能提案或替你启动下一研究阶段。跟踪和认知推理尚未作为已实现能力写入规范。旧 README 和阶段文档中的历史状态，须结合代码、STAGE1.md 及实际实验记录判断。

## 常用终端命令

```bash
npm run openspec -- list
npm run openspec -- list --specs
npm run openspec -- doctor
npm run openspec -- status --change <change-name>
npm run openspec -- validate <change-name> --strict --no-interactive
```

上传阿里云时一并保留 package.json、package-lock.json、AGENTS.md、openspec/ 和 .agents/。node_modules/ 不上传，确实需要在云端使用 OpenSpec 时安装兼容 Node.js 并运行 npm ci；只做 Python 训练不需要 Node.js。

## 更新

需要升级时，在项目目录显式执行并审查锁文件和生成文件变化：

```bash
npm install --save-dev --save-exact @fission-ai/openspec@latest
npm run openspec -- update
npm run openspec:check
```

不要在官方生成技能文件中长期维护项目要求；将项目要求写入 openspec/config.yaml 或 AGENTS.md。参阅 [OpenSpec 官方说明](https://github.com/Fission-AI/OpenSpec)。
