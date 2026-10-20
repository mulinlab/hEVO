#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成分面 PCA 散点图 PDF (每个 Group 一个面板, 共享同一坐标轴)

用法:
  python generate_pca_facet_pdf.py -i input.txt -o output.pdf
  python generate_pca_facet_pdf.py -i input.txt -o output.pdf --pc-x PC1 --pc-y PC2
  python generate_pca_facet_pdf.py -i input.txt -o output.pdf --max-points 20000
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "UKB": "UK Biobank",
    "EAS": "East Asian cohort",
    "AFR": "African cohort",
    "HGDP": "HGDP",
    "UK10K": "UK10K",
    "SG10K": "Singapore 10K",
    "TopMed": "TOPMed",
    "1kgenome": "1000 Genomes",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="分面 PCA 散点图 (每个 Group 一个面板, 共享坐标轴)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="输入 eigenvec 文件 (列: #IID PC1 PC2 ... Group)",
    )
    parser.add_argument("-o", "--output", required=True, help="输出 PDF 文件路径")
    parser.add_argument("--pc-x", default="PC1", help="X 轴主成分 (默认: PC1)")
    parser.add_argument("--pc-y", default="PC2", help="Y 轴主成分 (默认: PC2)")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="每组最大点数 (默认: None=全部样本)",
    )
    parser.add_argument(
        "--point-size", type=float, default=3.0, help="散点大小 (默认: 3.0)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="散点透明度 0-1 (默认: 1.0 不透明)"
    )
    parser.add_argument(
        "--pad", type=float, default=0.05, help="坐标轴向外扩展比例 (默认: 0.05)"
    )
    parser.add_argument("--ncol", type=int, default=4, help="每行面板数 (默认: 4)")
    parser.add_argument(
        "--seed", type=int, default=42, help="降采样随机种子 (默认: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the data.
    df = pd.read_csv(args.input, sep="\t", comment=None, header=0)
    df.columns = [c.lstrip("#") for c in df.columns]
    print(f"输入: {args.input}")
    print(f"总样本数: {len(df):,}")

    # Use a fixed order, either by first appearance or a custom sort order.
    order = list(df["Group"].value_counts().index)
    df["Group"] = pd.Categorical(df["Group"], categories=order, ordered=True)

    # Assign colors automatically.
    cmap = plt.cm.tab10
    colors = {grp: cmap(i % 10) for i, grp in enumerate(order)}

    # Downsample; use all samples when max_points is None.
    if args.max_points is not None:
        np.random.seed(args.seed)
        sampled = []
        for grp, g in df.groupby("Group"):
            n = min(len(g), args.max_points)
            sampled.append(g.sample(n, random_state=args.seed))
        df = pd.concat(sampled, ignore_index=True)
        print(f"降采样后: {len(df):,} (每组最多 {args.max_points:,})")
    else:
        print("使用全部样本 (无降采样)")

    # Share axis limits and add padding around the plotted data.
    x_min, x_max = df[args.pc_x].min(), df[args.pc_x].max()
    y_min, y_max = df[args.pc_y].min(), df[args.pc_y].max()
    x_pad = (x_max - x_min) * args.pad
    y_pad = (y_max - y_min) * args.pad
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # Layout: calculate the number of rows from the number of groups.
    n_groups = len(order)
    ncol = min(args.ncol, n_groups)
    nrow = int(np.ceil(n_groups / ncol))

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4 * ncol, 4 * nrow), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, grp in zip(axes, order):
        sub = df[df["Group"] == grp]
        ax.scatter(
            sub[args.pc_x],
            sub[args.pc_y],
            s=args.point_size,
            c=[colors[grp]],
            alpha=args.alpha,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(
            DISPLAY_NAMES.get(str(grp), str(grp)), fontsize=11, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused empty panels.
    for ax in axes[n_groups:]:
        ax.set_visible(False)

    # Share the axes.
    for ax in axes[:n_groups]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(False)

    fig.suptitle(
        f"PCA: {args.pc_x} vs {args.pc_y} (faceted by cohort)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.02, args.pc_x, ha="center", fontsize=12)
    fig.text(0.02, 0.5, args.pc_y, va="center", rotation="vertical", fontsize=12)

    plt.tight_layout(rect=[0.03, 0.04, 1, 0.95])
    fig.savefig(args.output, dpi=150)
    print(f"已输出: {args.output}")


if __name__ == "__main__":
    main()
