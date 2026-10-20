"""数据快照读取和完整性辅助函数。

这里不做指标重算；panel 脚本只对行数、关键列和输入文件存在性做轻量检查。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def require_file(path: Path) -> Path:
    """在读取前确认文件存在且非空。"""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required panel input is missing or empty: {path}")
    return path


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """读取 CSV，并保留显式路径用于 QA 记录。"""

    return pd.read_csv(require_file(path), **kwargs)


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    """读取制表符文件；原始 TSV 不转换回 CSV。"""

    return pd.read_csv(require_file(path), sep="\t", **kwargs)


def sha256(path: Path) -> str:
    """计算输入或输出文件的 SHA256。"""

    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    """检查 panel 数据所需字段，失败时给出可定位错误。"""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
