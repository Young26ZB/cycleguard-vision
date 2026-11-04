# 第一阶段实现与云端验证说明

本阶段修复检测训练链路和算法正确性，不开展性能训练，不修改数据标注。基础版与增强版均已通过 CPU 合成张量检查；实际数据训练、CUDA FP16 验证和性能评估留给云端。

## 已完成的修复

- 默认选择 `yolo11-ebike.yaml`，配置拼写错误立即报错，不再回退到标准 YOLO11。`--model-scale` 通过 Ultralytics 规模别名真正影响构建与训练器重建。支持 n/s/m/l/x；YAML 文件名使用 `yolo11[-variant].yaml` 约定。`--model-scale` 优先于文件名中的规模。
- 注册 `use_repulsion`、`repulsion_weight`，支持 `--no-repulsion`、`--repulsion-weight` 和 `--no-amp`。
- 先构建指定架构，再加载本地预训练权重。标准检测头没有 one-to-one 分支时，以已经迁移的 one-to-many 分支初始化该分支。增强版结构层号不同，权重只能部分迁移，应检查启动时的迁移数量。
- 固定每个检测头实例的 end2end 状态，避免后续构造标准模型改变先前端到端模型的模式。
- 排斥损失使用真正的全部有效 GT 和分配器返回的 GT 索引；RepGT 排除本目标，RepBox 排除同一 GT 的预测对。各尺度预测统一转换为图像坐标，逐图计算，无跨图排斥。
- 排斥项在 float32 中计算，使用平滑对数和范围约束；无正样本时正常返回零；外部 repulsion_weight 只应用一次。当前内部权重为 RepGT 0.5、RepBox 1.0，两分支均启用。
- DySample 基础网格与 align_corners=False 对齐，修复可选 DyScope 广播维度，并使用 float32 网格采样支持混合精度。本仓库的 DySample 仍是“先插值、再动态采样”的自定义变体，不应在论文中描述成原论文的逐行复现。
- 数据根路径按源 YAML 所在目录解析，或由 `--data-root` 覆盖；在 project/_configs/ 写入含绝对路径的副本，不修改源 YAML。训练和评估脚本均使用此方式，避免依赖全局 datasets_dir。直接调用 Ultralytics API 时，请使用生成的副本。

## 本地验证范围

使用现有环境 Python 3.10 / PyTorch 2.5.1 CPU，运行：

```bash
python -m unittest tests.test_stage1 -v
```

六项检查覆盖规模配置、参数注册、数据路径、DySample 零偏移与 DyScope、排斥匹配和坐标变换、基础版和增强版空/单/多目标前向反向、CPU BF16 autocast、权重保存加载、标准权重迁移。全部使用随机合成张量，无优化器、无参数更新、无数据集训练。保存会量化到 FP16，所以保存加载一致性比较有空间顺序的原始分支输出，避免未训练模型 TopK 分数近似相等时的排序变化。

## 阿里云上运行

将完整仓库及 ebike/ 上传到同一目录。先准备与云端 GPU 驱动兼容的 PyTorch/torchvision 环境，再从项目根目录安装本项目。建议 Python 3.10 或更新版本。

```bash
pip install -e .
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available()); print(ultralytics.__file__)"
```

确认 ultralytics 指向上传的仓库。先做无训练的配置检查：

```bash
python train_ebike.py --model yolo11-ebike.yaml --model-scale n --data-root /data/ebike --no-repulsion --check-only
python train_ebike.py --model yolo11-nms-free.yaml --model-scale n --data-root /data/ebike --check-only
```

把 `/data/ebike` 替换为云端实际路径，目录中应包含 images/train 和 images/val。云端 GPU 合成数据检查也不启动训练：

```bash
STAGE1_DEVICE=cuda python -m unittest tests.test_stage1 -v
```

CPU 检查已通过；上述 CUDA FP16 检查尚未执行，需云端确认。

## 后续云端训练命令

下列命令仅供后续执行，本阶段没有运行。先用少量 epoch 检查真实数据链路，再开展正式对比。初步检查可以关闭 AMP，随后移除 --no-amp 比较混合精度。

```bash
# 真实数据短轮次检查，不代表正式实验成绩
python train_ebike.py --model yolo11-ebike.yaml --model-scale n --data-root /data/ebike --epochs 2 --batch 4 --workers 2 --device 0 --no-repulsion --no-amp --name stage1_data_check

# 基础 NMS-Free 对照，不使用排斥损失
python train_ebike.py --model yolo11-ebike.yaml --model-scale n --data-root /data/ebike --epochs 300 --batch 16 --workers 4 --device 0 --no-repulsion --name e2e_baseline

# DySample + SimAM + 修正后的排斥损失
python train_ebike.py --model yolo11-nms-free.yaml --model-scale n --data-root /data/ebike --epochs 300 --batch 16 --workers 4 --device 0 --repulsion-weight 0.5 --name e2e_enhanced

# 可选加载本地权重：在上述命令追加 --pretrained /path/to/yolo11n.pt
# 恢复新实验：--model runs/train/实验名/weights/last.pt --resume

python evaluate_model.py --weights runs/train/e2e_baseline/weights/best.pt --data-root /data/ebike --device 0 --save-plots
```

默认启用排斥损失；做纯端到端基线必须显式加 --no-repulsion。修改损失后应建立新实验，不以历史检查点续训作为旧实验的等价延续。--resume 保留检查点的原架构和训练配置，不用于切换模型或损失消融。

## 尚待云端完成

- 短轮次真实数据训练、验证和中断恢复；实际 GPU FP16 检查。
- 数据质量清理和有标注的拥堵测试集；现有零宽框、负样本和 classes.txt 等未在本阶段修改。
- BiFPN 实际组网、各模块消融、正式训练及延迟测试。当前增强版配置包含 DySample 和 SimAM，不包含 BiFPN。
- 对当前 topk=1 任务对齐分配进行实验评估；实现不是全局二分匹配，不能声称严格保证每个 GT 始终获得唯一正确预测。

现有历史训练成绩不能作为这些修复后的性能验证，本阶段只确认实现与合成数据链路。
