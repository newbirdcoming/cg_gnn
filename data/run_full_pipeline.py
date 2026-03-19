from pathlib import Path
import subprocess
import sys


def run_module(module: str) -> None:
    """
    在子进程中运行一个 Python 模块（例如 'data.data_preprocess.preview_clean_excel'），
    并将 stdout/stderr 直接透传到当前终端。
    """
    cmd = [sys.executable, "-m", module]
    print(f"[run_full_pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed with exit code {result.returncode}: {module}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    raw_excel = data_dir / "combinnation.xlsx"
    cleaned_excel = data_dir / "cleaned_combinnation.xlsx"

    if not raw_excel.exists():
        raise FileNotFoundError(
            f"原始 Excel 文件不存在: {raw_excel}\n"
            "请将原始人工整理表格保存为该路径后再运行本脚本。"
        )

    print(f"[run_full_pipeline] 项目根目录: {project_root}")
    print(f"[run_full_pipeline] 原始 Excel: {raw_excel}")

    # Step 1: 从 combinnation.xlsx 生成 cleaned_combinnation.xlsx
    run_module("data.data_preprocess.preview_clean_excel")

    if not cleaned_excel.exists():
        raise FileNotFoundError(
            f"清洗后的 Excel 未找到: {cleaned_excel}\n"
            "请检查 preview_clean_excel 是否成功运行。"
        )

    # Step 2: 从 cleaned_combinnation.xlsx 生成 triples.csv + processed 数据
    run_module("data.data_preprocess.build_triples_and_splits")

    processed_dir = data_dir / "processed"
    raw_triples = data_dir / "raw" / "triples.csv"

    print("\n[run_full_pipeline] 全部步骤完成。关键产物：")
    print(f"- 清洗后宽表: {cleaned_excel}")
    print(f"- 原始三元组 CSV: {raw_triples}")
    print(f"- processed 目录: {processed_dir}")


if __name__ == "__main__":
    main()

from pathlib import Path
import subprocess
import sys


def run_module(module: str) -> None:
    """
    Run a Python module (e.g. 'data.data_preprocess.preview_clean_excel')
    in a subprocess, forwarding stdout/stderr.
    """
    cmd = [sys.executable, "-m", module]
    print(f"[run_full_pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {result.returncode}: {module}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    raw_excel = data_dir / "combinnation.xlsx"
    cleaned_excel = data_dir / "cleaned_combinnation.xlsx"

    if not raw_excel.exists():
        raise FileNotFoundError(
            f"原始 Excel 文件不存在: {raw_excel}\n"
            "请将原始人工整理表格保存为该路径后再运行本脚本。"
        )

    print(f"[run_full_pipeline] 项目根目录: {project_root}")
    print(f"[run_full_pipeline] 原始 Excel: {raw_excel}")

    # Step 1: 从 combinnation.xlsx 生成 cleaned_combinnation.xlsx
    run_module("data.data_preprocess.preview_clean_excel")

    if not cleaned_excel.exists():
        raise FileNotFoundError(
            f"清洗后的 Excel 未找到: {cleaned_excel}\n"
            "请检查 preview_clean_excel 是否成功运行。"
        )

    # Step 2: 从 cleaned_combinnation.xlsx 生成 triples.csv + processed 数据
    run_module("data.data_preprocess.build_triples_and_splits")

    processed_dir = data_dir / "processed"
    raw_triples = data_dir / "raw" / "triples.csv"

    print("\n[run_full_pipeline] 全部步骤完成。关键产物：")
    print(f"- 清洗后宽表: {cleaned_excel}")
    print(f"- 原始三元组 CSV: {raw_triples}")
    print(f"- processed 目录: {processed_dir}")


if __name__ == "__main__":
    main()

