"""只读审计 YOLO 数据，按人工复核清单发布派生版本。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from ebike_diagnostics import rooted_image, sha256_file


# 与本仓库检测加载器的图片后缀一致，审计无需导入 PyTorch。
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
BOUND_TOLERANCE = 1e-6


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _label_row(line):
    fields = line.split()
    if len(fields) != 5:
        return None, "invalid_format"
    try:
        values = tuple(map(float, fields))
    except ValueError:
        return None, "invalid_format"
    cls, cx, cy, width, height = values
    if not all(map(math.isfinite, values)):
        return None, "nonfinite"
    if cls != 0:
        return None, "invalid_class"
    if width == 0:
        return None, "zero_width"
    if height == 0:
        return None, "zero_height"
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None, "invalid_geometry"
    # 允许小数序列化舍入误差。
    if min(cx - width / 2, cy - height / 2) < -BOUND_TOLERANCE or max(
        cx + width / 2, cy + height / 2
    ) > 1 + BOUND_TOLERANCE:
        return None, "out_of_bounds"
    return values, None


def _read_label(path):
    if not path.is_file():
        return None, [{"code": "missing_label"}]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError:
        return [], [{"code": "invalid_encoding"}]
    issues = []
    for number, line in enumerate(lines, 1):
        if line.strip():
            _, code = _label_row(line)
            if code:
                issues.append({"code": code, "line": number, "text": line})
    return lines, issues


def scan_dataset(data_root, splits=("train",)):
    """返回逐路径数据快照和可公开的审计报告。"""
    root = Path(data_root).resolve()
    records, report = {}, {"schema_version": 1, "splits": {}, "issues": {}, "orphan_labels": []}
    for split in splits:
        base = root / "images" / split
        paths = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"找不到 {split} 图片 {base}")
        label_base = root / "labels" / split
        pairs = {}
        for image in paths:
            relative = image.relative_to(root).as_posix()
            rooted_image(root, relative)
            label = label_base / image.relative_to(base).with_suffix(".txt")
            if root not in label.resolve().parents:
                raise ValueError(f"标签超出数据根目录 {label}")
            pairs[relative] = label
        collisions = Counter(p.relative_to(root).as_posix().casefold() for p in pairs.values())
        counts = Counter(images=len(paths))
        for relative, label in pairs.items():
            label_name = label.relative_to(root).as_posix()
            lines, issues = _read_label(label)
            if lines is not None and not any(line.strip() for line in lines) and not issues:
                counts["empty_labels"] += 1
            if collisions[label_name.casefold()] > 1:
                issues.append({"code": "shared_label_path"})
            record = {"split": split, "label": label_name, "lines": lines, "issues": issues}
            records[relative] = record
            if issues:
                for issue in issues:
                    counts[issue["code"]] += 1
                report["issues"][relative] = {
                    "label": label_name, "issues": issues,
                    "image_sha256": sha256_file(root / relative),
                    "label_sha256": sha256_file(label) if lines is not None else None,
                    "status": "pending",
                }
        expected = set(pairs.values())
        report["orphan_labels"].extend(
            p.relative_to(root).as_posix()
            for p in sorted(label_base.rglob("*.txt"))
            if p.is_file() and p.name.casefold() != "classes.txt" and p not in expected
        )
        report["splits"][split] = dict(counts)
    return records, report


def _approved_labels(name, approval, snapshot, old_lines):
    if not isinstance(approval, dict):
        raise ValueError(f"缺少已确认处置 {name}")
    status = approval.get("status")
    if status not in ("background", "annotated") or not all(
        isinstance(approval.get(key), str) and approval[key].strip() for key in ("reviewed_by", "evidence")
    ):
        raise ValueError(f"处置尚未确认 {name}")
    if any(key not in approval or approval[key] != snapshot[key] for key in ("image_sha256", "label_sha256")):
        raise ValueError(f"复核后来源已变化 {name}")
    lines = approval.get("labels")
    if not isinstance(lines, list) or any(not isinstance(row, str) or _label_row(row)[1] for row in lines):
        raise ValueError(f"复核标签格式无效 {name}")
    if (status == "background") != (not lines):
        raise ValueError(f"处置类型与标签内容不符 {name}")
    codes = {item["code"] for item in snapshot["issues"]}
    # 零宽框处置只替换异常行，保留原文件的其他有效目标。
    if "zero_width" in codes and "shared_label_path" not in codes:
        valid_original = Counter(_label_row(row)[0] for row in old_lines if row.strip() and not _label_row(row)[1])
        if valid_original - Counter(_label_row(row)[0] for row in lines):
            raise ValueError(f"零宽框处置丢失同图其他有效标签 {name}")
    return lines


def publish_dataset(data_root, output, review_path):
    """只发布完整的 train/val 派生版本，输出目录必须是新目录。"""
    root, output = Path(data_root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("派生目录必须与原数据目录分离")
    if output.exists():
        raise FileExistsError(f"派生目录已存在 {output}")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("schema_version") != 1 or not isinstance(review.get("reviews"), dict):
        raise ValueError("处置清单须包含 schema_version: 1 和 reviews 对象")
    approvals = review["reviews"]
    records, audit = scan_dataset(root, ("train", "val"))
    issues = audit["issues"]
    if audit["orphan_labels"]:
        raise ValueError(f"有未配对的标签 {audit['orphan_labels']}")
    missing = issues.keys() - approvals.keys()
    extra = approvals.keys() - issues.keys()
    if missing or extra:
        raise ValueError(f"处置清单不完整或已过期，缺 {len(missing)}，多 {len(extra)}")
    labels = {
        name: _approved_labels(name, approvals[name], issues[name], record["lines"])
        if name in issues else record["lines"]
        for name, record in records.items()
    }

    output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    try:
        for index, (name, record) in enumerate(records.items(), 1):
            stem = f"{index:06d}"
            image = Path("images") / record["split"] / (stem + Path(name).suffix.lower())
            label = Path("labels") / record["split"] / (stem + ".txt")
            (output / image).parent.mkdir(parents=True, exist_ok=True)
            (output / label).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, output / image)
            digest = sha256_file(output / image)
            if name in issues and digest != issues[name]["image_sha256"]:
                raise ValueError(f"复核后图片已变化 {name}")
            lines = [row.strip() for row in labels[name] if row.strip()]
            (output / label).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            mapping[name] = {"image": image.as_posix(), "label": label.as_posix(), "sha256": digest,
                             "source_label": record["label"], "review": approvals.get(name)}
        derived, check = scan_dataset(output, ("train", "val"))
        if check["issues"] or check["orphan_labels"] or len(derived) != len(records):
            raise ValueError("派生数据的标签或成员校验失败")
        write_json(output / "source-map.json", {"schema_version": 1, "images": mapping})
        # 发布 YAML 最后写入。相对路径由现有训练入口按 YAML 所在目录解析。
        config = output / "ebike.yaml"
        config.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ebike\n", encoding="utf-8")
    except Exception:
        if output.resolve() == output:
            shutil.rmtree(output)
        raise
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="只读扫描并生成异常报告")
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--split", choices=("train", "val", "both"), default="train")
    audit.add_argument("--output", type=Path, required=True)
    release = commands.add_parser("publish", help="生成新的派生版本")
    release.add_argument("--data-root", type=Path, required=True)
    release.add_argument("--review", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            root, output = args.data_root.resolve(), args.output.resolve()
            if output == root or root in output.parents:
                raise ValueError("审计报告必须保存在原数据目录之外")
            splits = ("train", "val") if args.split == "both" else (args.split,)
            _, report = scan_dataset(root, splits)
            write_json(output, report)
            print(json.dumps(report["splits"], ensure_ascii=False))
        else:
            print(publish_dataset(args.data_root, args.output, args.review))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
