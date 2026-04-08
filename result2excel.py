from pathlib import Path
import json
import pandas as pd
from numbers import Number


def flatten_dict(d, parent_key="", sep=".", only_numeric=False, skip_prefixes=None):
    """
    递归展开嵌套字典:
    {"a": {"b": 1}} -> {"a.b": 1}

    参数:
    - only_numeric: True 时，只保留数值型叶子节点
    - skip_prefixes: 跳过指定前缀的字段，比如 ["part.per_class_raw_counts"]
    """
    if skip_prefixes is None:
        skip_prefixes = []

    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)

        # 跳过指定前缀
        if any(new_key.startswith(prefix) for prefix in skip_prefixes):
            continue

        if isinstance(v, dict):
            items.extend(
                flatten_dict(
                    v,
                    parent_key=new_key,
                    sep=sep,
                    only_numeric=only_numeric,
                    skip_prefixes=skip_prefixes
                ).items()
            )
        else:
            if only_numeric:
                if isinstance(v, Number):
                    items.append((new_key, v))
            else:
                items.append((new_key, v))

    return dict(items)


def collect_results_by_method(
    dataset_dirs,
    json_name="summary_all_methods.json",
    only_numeric=True,
    include_raw_counts=False
):
    """
    读取多个数据集文件夹中的 summary_all_methods.json，
    按 method 汇总：
    {
        "citygml_clip": [row1, row2, ...],
        "langsplat": [row1, row2, ...],
        ...
    }

    新 summary 支持格式:
    {
        "citygml_clip": {
            "whole": {...},
            "part": {...}
        },
        ...
    }
    """
    method_rows = {}

    skip_prefixes = []
    if not include_raw_counts:
        skip_prefixes.append("part.per_class_raw_counts")

    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        json_path = dataset_dir / json_name

        if not json_path.exists():
            print(f"[跳过] 未找到文件: {json_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[跳过] 读取失败: {json_path}, 错误: {e}")
            continue

        # 第一层 key 是 method 名
        for method_name, method_result in data.items():
            row = {
                "dataset_name": dataset_dir.name,
                "dataset_path": str(dataset_dir.resolve()),
            }

            flat_result = flatten_dict(
                method_result,
                only_numeric=only_numeric,
                skip_prefixes=skip_prefixes
            )
            row.update(flat_result)

            if method_name not in method_rows:
                method_rows[method_name] = []
            method_rows[method_name].append(row)

    return method_rows


def export_to_excel_by_method(
    dataset_dirs,
    output_excel="evaluation_summary_by_method.xlsx",
    blank_rows=3,
    only_numeric=True,
    include_raw_counts=False
):
    """
    把不同 method 的结果写到同一个 sheet 中，
    每个 method 一个独立表格，中间空 blank_rows 行。
    """
    method_rows = collect_results_by_method(
        dataset_dirs,
        only_numeric=only_numeric,
        include_raw_counts=include_raw_counts
    )

    if not method_rows:
        print("没有成功读取到任何结果。")
        return

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        sheet_name = "summary"
        current_row = 0

        # 按方法名排序，输出更稳定
        for method_name in sorted(method_rows.keys()):
            rows = method_rows[method_name]
            df = pd.DataFrame(rows)

            # 列顺序：dataset_name, dataset_path 放前面，其余排序
            front_cols = ["dataset_name", "dataset_path"]
            other_cols = sorted([c for c in df.columns if c not in front_cols])
            df = df[front_cols + other_cols]

            # 写方法标题
            pd.DataFrame([[method_name]]).to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=current_row,
                startcol=0,
                index=False,
                header=False
            )

            # 写表格
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=current_row + 1,
                startcol=0,
                index=False
            )

            # 下一张表起始行
            current_row += len(df) + 1 + 1 + blank_rows
            # 数据行 + 标题行 + 表头行 + 空白行

    print(f"已导出到: {output_excel}")


if __name__ == "__main__":
    dataset_dirs = [
        "/workspace/zaha_eval/all_eval_results_1",
        "/workspace/zaha_eval/all_eval_results_44",
        "/workspace/zaha_eval/all_eval_results_55",
        "/workspace/zaha_eval/all_eval_results_66",
        "/workspace/zaha_eval/all_eval_results_88",
        "/workspace/zaha_eval/all_eval_results_goldcoast6"
    ]

    export_to_excel_by_method(
        dataset_dirs,
        output_excel="evaluation_summary_by_method.xlsx",
        blank_rows=3,
        only_numeric=True,         # 只导出数值
        include_raw_counts=True   # 不导出 tp/fp/fn/tn；要的话改成 True
    )