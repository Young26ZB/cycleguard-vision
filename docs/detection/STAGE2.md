# 第二阶段数据审计与诊断评估

诊断清单覆盖当前测试目录中的 13 张图。现有图与训练图重叠，因此诊断报告只能用于分析流程与错误案例。

截至 2026-11-05，审计、派生数据发布工具和诊断计分已实现并通过本地非训练检查。人工确认和实际派生版本发布、云端真实数据训练及诊断评估待完成，详见[归档任务记录](../../openspec/changes/archive/2026-11-05-stage2-ebike-data-diagnostics/tasks.md)。

## 诊断清单

清单使用 UTF-8 JSON，顶层包含 `schema_version: 1`、`dataset_version` 和 `images`。图片以数据根下的完整相对路径为键，统一用 `/` 并保留扩展名。每张图包含以下字段。

| 字段 | 含义 |
| --- | --- |
| `width`、`height`、`sha256` | 原图宽高及文件内容 SHA-256，加载时核对 |
| `training_overlap` | 是否已知与训练集重叠 |
| `lighting` | `day`、`night` 或 `unknown` |
| `status` | `ready` 表示可计分，`pending` 表示整图待复核 |
| `targets` | 骑行中的真实目标列表。每项包含 `box` 和 `occlusion`，遮挡取 `none`、`partial` 或 `unknown` |
| `ignore` | 已确认无人骑行电动车的框列表 |
| `review` | 复核项列表，每项包含 `reason`、`location`、`evidence` 和 `conclusion`，未决结论填 `null` |

各框为原图像素 `[x1, y1, x2, y2]`，在图像内且宽高为正。三个列表均需显式提供，无项时填 `[]`。原因可用 `uncertain_class`（车型）、`uncertain_bounds`（范围）或 `uncertain_ownership`（实例归属）。任一原因未解除时，整图保持 `pending`。

实际逐图复核记录见 [13 张图的诊断清单](stage2/diagnostic-13.json)。

## 训练数据审计与派生版本

在仓库根目录执行以下命令。审计报告必须位于原数据目录之外，文件名须为新文件。

```powershell
python audit_ebike.py audit --data-root ebike --split both --output runs/stage2/audit.json
```

当前扫描结果见 [训练审计报告](stage2/training-audit.json)，逐路径初筛见 [复核草稿](stage2/training-review-draft.json)。草稿中的建议与证据供人工复核使用。发布前需另存处置清单，填写已确认的状态、复核者和标签行。带 `negative_` 前缀的图片中已有疑似漏标案例，文件名不能用来确认背景。

按报告中的完整图片路径填写 UTF-8 JSON 处置清单。顶层为 `schema_version: 1` 和 `reviews`。每项包含 `status`、`reviewed_by`、`evidence`、审计时的 `image_sha256` 与 `label_sha256`，以及完整的 YOLO `labels` 行列表。确认背景用 `background` 和空列表，确认或修正目标用 `annotated` 和有效标签行。零宽框的处置保留同图其他有效目标。未决项留待复核，发布入口会拒绝不完整、未确认或过期的处置。

```powershell
python audit_ebike.py publish --data-root ebike --review runs/stage2/review.json --output runs/stage2/derived-v1
```

派生版本保存逐图来源映射，并在通过校验后生成 `ebike.yaml`。后续云端训练可将它交给现有 `--data` 参数。生成过程保留源图和源标签，派生目录必须是新目录。

## 诊断计分

真实图像评估留待阿里云。将权重、原图和诊断清单放到可访问路径后运行以下命令。每次指定新的运行名。

```powershell
python evaluate_model.py --weights best.pt --data-root ebike --diagnostic-manifest docs/detection/stage2/diagnostic-13.json --name stage2_diagnostic_v1
```

默认诊断置信度、匹配 IoU 和忽略 IoA 分别为 `0.25`、`0.50` 和 `0.80`，可用 `--diagnostic-conf`、`--diagnostic-match-iou` 和 `--diagnostic-ignore-ioa` 覆盖。模型对照使用相同参数、图片和 `--diagnostic-max-det`。`--iou` 仍用于原生 NMS，常规评估继续使用原有命令。

报告为运行目录中的 `diagnostic-metrics.json`。预测依次判定为命中、重复、忽略或普通误检，重复检出计入误检。定位偏差为配对框 `1 − IoU` 的中位数。无定义值写为 `null`，待复核图整图跳过并在报告中说明。
