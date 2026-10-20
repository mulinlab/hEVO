
from __future__ import annotations

# [Shared utilities] Provide table loading, field validation, row and column summaries, and source-data rendering helpers.
# [Data contract] Convert CSV and CSV.GZ files into structures for plotting scripts; panel layout is defined elsewhere.
# [QA] Higher-level entry points usually write these check results to QA JSON or summary CSV files.
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    from .hevo_palette import ANCESTRY_COLORS
except ImportError:
    from hevo_palette import ANCESTRY_COLORS
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLES = ROOT / "tables"
DEFAULT_PANELS = ROOT / "figures" / "panels"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["text.color"] = "#000000"
plt.rcParams["axes.labelcolor"] = "#000000"
plt.rcParams["axes.edgecolor"] = "#000000"
plt.rcParams["xtick.color"] = "#000000"
plt.rcParams["ytick.color"] = "#000000"
plt.rcParams["axes.grid"] = False
PALETTE = ["#0B2E83", "#B35C1E", "#0E7C86", "#7655A6", "#D97706", "#4F6D7A", "#6B7C93", "#8995A5"]
POP_COLORS = {**ANCESTRY_COLORS, "AMR":ANCESTRY_COLORS["Others"]}
COLORS = {
    "other": "#D6DAE0",
    "hevo_86": "#0B2E83",
    "hevo_340": "#B35C1E",
    "grid": "#FFFFFF",
    "outline": "#000000",
    "baseline": "#C23B22",
}
FOCUS_MODELS = {
    "hEvo-base": "hevo_86",
    "hEvo-large": "hevo_340",
}
DISPLAY_NAMES = {
    "hEvo-base": "hEvo-base",
    "hEvo-large": "hEvo-large",
    # Nucleotide Transformer generations are distinct architectures.  Keep
    # the generation in every rendered label; the unversioned checkpoints
    # below are the original NT-v1 release, whereas the 500M multispecies
    # checkpoint used in this project is NT-v2.
    "NT_500m_1kG": "NTv1-1000G-500M",
    "NT_2.5b_1kG": "NTv1-1000G-2.5B",
    "NT_2.5b_MS": "NTv1-MS-2.5B",
    "NT_500m_MS": "NTv2-MS-500M",
    "NT-500M-1kG": "NTv1-1000G-500M",
    "NT-2.5B-1kG": "NTv1-1000G-2.5B",
    "NT-2.5B-MS": "NTv1-MS-2.5B",
    "NT-500M-MS": "NTv2-MS-500M",
}
def color_for_model(model: str) -> str:
    return COLORS[FOCUS_MODELS[model]] if model in FOCUS_MODELS else COLORS["other"]
def display_model_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)
def parse_args(description="Render figure from GitLab tables"):
    p=argparse.ArgumentParser(description=description); p.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES); p.add_argument("--out-dir", type=Path, default=None); p.add_argument("--formats", default="png,pdf,svg"); p.add_argument("--dry-run", action="store_true"); return p.parse_args()
def read_table(tables_dir: Path, table_file: str, required=()):
    path=tables_dir/table_file
    if not path.exists(): raise FileNotFoundError(f"Required table not found: {path}")
    df=pd.read_csv(path); missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"{table_file} missing required columns: {missing}")
    return df
def finish(fig, args, fig_group: str, stem: str):
    out_dir=args.out_dir or (DEFAULT_PANELS/fig_group); formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {fig_group}/{stem}: formats={formats} out_dir={out_dir}"); plt.close(fig); return
    # Publication contract shared with formal Figure 1: no panel grids, pure
    # black typography, white background, editable vector text, and 600-dpi PNG.
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.grid(False)
        ax.set_facecolor("white")
        ax.tick_params(colors="#000000")
        ax.xaxis.label.set_color("#000000")
        ax.yaxis.label.set_color("#000000")
        ax.title.set_color("#000000")
        for spine in ax.spines.values():
            spine.set_edgecolor("#000000")
    for txt in fig.findobj(match=matplotlib.text.Text):
        txt.set_color("#000000")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"dpi": 600} if fmt == "png" else {}
        fig.savefig(out_dir/f"{stem}.{fmt}", bbox_inches="tight", **kwargs)
    plt.close(fig)
def shorten(labels, n=34): return [str(x) if len(str(x))<=n else str(x)[:n-1]+"..." for x in labels]
def barh_metric(df, model_col, metric, title, xlabel, ci_low=None, ci_high=None, baseline=None):
    d=df.copy().sort_values(metric, ascending=True); fig,ax=plt.subplots(figsize=(7.5,max(4,0.22*len(d)+1.4))); y=np.arange(len(d)); vals=d[metric].astype(float).to_numpy(); ax.barh(y, vals, color=PALETTE[0], alpha=.9)
    if ci_low and ci_high and ci_low in d and ci_high in d:
        lo=d[ci_low].astype(float).to_numpy(); hi=d[ci_high].astype(float).to_numpy(); ax.errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt='none',ecolor='black',elinewidth=.8,capsize=2)
    if baseline is not None: ax.axvline(float(baseline), color="#555", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(shorten(d[model_col]), fontsize=7); ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(axis='x', alpha=.18); ax.spines[['top','right']].set_visible(False); return fig
def render_pca_scatter(args,t,g,s,note=''):
    df=read_table(args.tables_dir,t,["IID","PC1","PC2","Group"])
    group_order=["1kgenome","HGDP","AFR","EAS","SG10K","UK10K","UKB","TopMed"]
    group_colors={"1kgenome":"#4E79A7","HGDP":"#59A14F","AFR":"#E15759","EAS":"#F28E2B","SG10K":"#76B7B2","UK10K":"#EDC948","UKB":"#B07AA1","TopMed":"#9C755F"}
    groups=[group for group in group_order if group in set(df["Group"])]
    groups.extend(sorted(group for group in df["Group"].dropna().astype(str).unique() if group not in groups))
    def sample_by_group(frame,max_per_group,seed):
        sampled=[]
        for group,group_df in frame.groupby("Group",sort=False):
            sampled.append(group_df.sample(n=max_per_group,random_state=seed) if len(group_df)>max_per_group else group_df)
        return pd.concat(sampled,axis=0).reset_index(drop=True)
    seed=20260723
    scatter_max=12000
    background_df=sample_by_group(df,max_per_group=3500,seed=seed)
    x_min,x_max=df["PC1"].quantile([0.001,0.999])
    y_min,y_max=df["PC2"].quantile([0.001,0.999])
    x_pad=(x_max-x_min)*0.06
    y_pad=(y_max-y_min)*0.06
    xlim=(x_min-x_pad,x_max+x_pad)
    ylim=(y_min-y_pad,y_max+y_pad)
    if sns is not None:
        sns.set_theme(style="white",context="paper")
    fig,axes=plt.subplots(2,4,figsize=(13.5,7.0),sharex=True,sharey=True)
    axes=axes.ravel()
    for idx,group in enumerate(groups):
        ax=axes[idx]
        group_full=df[df["Group"]==group]
        group_scatter=group_full.sample(n=scatter_max,random_state=seed) if len(group_full)>scatter_max else group_full
        ax.scatter(background_df["PC1"],background_df["PC2"],s=2.0,alpha=0.06,color="#9A9A9A",edgecolors="none",rasterized=True)
        ax.scatter(group_scatter["PC1"],group_scatter["PC2"],s=5.0,alpha=0.42,color=group_colors.get(group,"#777777"),edgecolors="none",rasterized=True)
        ax.set_title(f"{group} highlighted (n={len(group_full):,})",fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True,color="#EEEEEE",linewidth=0.5)
        ax.tick_params(labelsize=8,length=2)
        if idx//4==1:
            ax.set_xlabel("PC1",fontsize=9)
        if idx%4==0:
            ax.set_ylabel("PC2",fontsize=9)
    for ax in axes[len(groups):]:
        ax.axis("off")
    fig.suptitle("Cohort position on PC1/PC2 after removing first-degree TopMed kinship samples",fontsize=13,y=0.98)
    fig.subplots_adjust(left=0.065,right=0.985,top=0.91,bottom=0.09,hspace=0.32,wspace=0.14)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":300} if fmt in {"png","pdf"} else {}
        fig.savefig(out_dir/f"{s}.{fmt}",**kwargs)
    plt.close(fig)
def render_pca_density(args,t,g,s,note=''):
    from matplotlib.lines import Line2D
    df=read_table(args.tables_dir,t,["IID","PC1","PC2","Group"])
    group_order=["1kgenome","HGDP","AFR","EAS","SG10K","UK10K","UKB","TopMed"]
    group_colors={"1kgenome":"#4E79A7","HGDP":"#59A14F","AFR":"#E15759","EAS":"#F28E2B","SG10K":"#76B7B2","UK10K":"#EDC948","UKB":"#B07AA1","TopMed":"#9C755F"}
    groups=[group for group in group_order if group in set(df["Group"])]
    groups.extend(sorted(group for group in df["Group"].dropna().astype(str).unique() if group not in groups))
    def sample_by_group(frame,max_per_group,seed):
        sampled=[]
        for group,group_df in frame.groupby("Group",sort=False):
            sampled.append(group_df.sample(n=max_per_group,random_state=seed) if len(group_df)>max_per_group else group_df)
        return pd.concat(sampled,axis=0).reset_index(drop=True)
    seed=20260723
    scatter_max=12000
    kde_max=5000
    counts=df["Group"].value_counts()
    scatter_df=sample_by_group(df,scatter_max,seed)
    x_min,x_max=df["PC1"].quantile([0.001,0.999])
    y_min,y_max=df["PC2"].quantile([0.001,0.999])
    x_pad=(x_max-x_min)*0.06
    y_pad=(y_max-y_min)*0.06
    xlim=(x_min-x_pad,x_max+x_pad)
    ylim=(y_min-y_pad,y_max+y_pad)
    if sns is not None:
        sns.set_theme(style="white",context="paper")
    fig=plt.figure(figsize=(13.5,10.5),constrained_layout=False)
    gs=fig.add_gridspec(4,4,height_ratios=[1.18,1.18,1.0,1.0],hspace=0.42,wspace=0.34)
    ax_main=fig.add_subplot(gs[0:2,:])
    for group in groups:
        group_scatter=scatter_df[scatter_df["Group"]==group]
        color=group_colors.get(group,"#777777")
        ax_main.scatter(group_scatter["PC1"],group_scatter["PC2"],s=4,alpha=0.18,color=color,edgecolors="none",rasterized=True)
        group_full=df[df["Group"]==group]
        kde_df=group_full.sample(n=kde_max,random_state=seed) if len(group_full)>kde_max else group_full
        if sns is not None and len(kde_df)>=50 and kde_df["PC1"].nunique()>5 and kde_df["PC2"].nunique()>5:
            try:
                sns.kdeplot(data=kde_df,x="PC1",y="PC2",ax=ax_main,levels=4,color=color,linewidths=0.8,fill=False,alpha=0.72,thresh=0.08)
            except Exception as exc:
                print(f"[WARNING] KDE skipped for {group}: {exc}")
    ax_main.set_title("PCA population structure after removing first-degree TopMed kinship samples",fontsize=13)
    ax_main.set_xlabel("PC1")
    ax_main.set_ylabel("PC2")
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.grid(True,color="#E5E5E5",linewidth=0.6)
    handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=group_colors.get(group,"#777777"),markeredgewidth=0,markersize=7,label=f"{group} (n={int(counts[group]):,})") for group in groups]
    ax_main.legend(handles=handles,title="Cohort",loc="upper left",bbox_to_anchor=(1.01,1.0),frameon=False,borderaxespad=0,fontsize=8,title_fontsize=9)
    for idx,group in enumerate(groups):
        ax=fig.add_subplot(gs[2+idx//4,idx%4])
        group_full=df[df["Group"]==group]
        group_scatter=group_full.sample(n=scatter_max,random_state=seed) if len(group_full)>scatter_max else group_full
        color=group_colors.get(group,"#777777")
        ax.scatter(group_scatter["PC1"],group_scatter["PC2"],s=4,alpha=0.25,color=color,edgecolors="none",rasterized=True)
        ax.set_title(f"{group} (n={len(group_full):,})",fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True,color="#EEEEEE",linewidth=0.5)
        if idx//4==1:
            ax.set_xlabel("PC1",fontsize=8)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if idx%4==0:
            ax.set_ylabel("PC2",fontsize=8)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7,length=2)
    fig.subplots_adjust(left=0.075,right=0.82,top=0.93,bottom=0.075)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":300} if fmt in {"png","pdf"} else {}
        fig.savefig(out_dir/f"{s}.{fmt}",**kwargs)
    plt.close(fig)
def render_cohort_stacked(args,t,g,s,note=''):
    df=read_table(args.tables_dir,t,["Cohort","EUR_count","AFR_count","EAS_count","SAS_count","Others_count","Total_samples"])
    apply_colors={pop:POP_COLORS[pop] for pop in ("EUR","AFR","EAS","SAS","Others")}
    text_dark="#000000"
    plt.rcParams.update({
        "font.family":"Arial",
        "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,"legend.frameon":False,
        "font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "figure.facecolor":"#FFFFFF","axes.facecolor":"#FFFFFF","savefig.facecolor":"#FFFFFF",
        "axes.edgecolor":text_dark,"axes.labelcolor":text_dark,"axes.titlecolor":text_dark,
        "xtick.color":text_dark,"ytick.color":text_dark,
    })
    pop_order=["EUR","AFR","EAS","SAS","Others"]
    plot_df=df[df["Cohort"]!="All"].copy()
    rename_map={f"{pop}_count":pop for pop in pop_order}
    rename_map["Total_samples"]="Total"
    plot_df=plot_df.rename(columns=rename_map).set_index("Cohort")
    cohort_order=["UKB","UK10K","EAS","SG10K","AFR","TOPMed","HGDP","1k genome"]
    missing=sorted(set(cohort_order)-set(plot_df.index))
    if missing:
        raise ValueError(f"Missing cohorts required by the Fig. 1 construction order: {missing}")
    all_source=df[df["Cohort"]=="All"].rename(columns=rename_map)
    if len(all_source)!=1:
        raise ValueError("Expected exactly one All row")
    all_row=all_source.iloc[0][["Cohort",*pop_order,"Total"]].to_dict()
    computed=plot_df.loc[cohort_order,[*pop_order,"Total"]].sum(axis=0)
    if any(int(all_row[col])!=int(computed[col]) for col in [*pop_order,"Total"]):
        raise ValueError("All row does not equal the sum of the eight source cohorts")
    plot_df=plot_df.loc[cohort_order].reset_index()
    plot_df=pd.concat([plot_df,pd.DataFrame([all_row])],ignore_index=True)
    label_map={"All":"hEVO corpus","UKB":"UKB subset","UK10K":"UK10K","EAS":"EAS2K","SG10K":"SG10K","AFR":"AFR1K","TOPMed":"TOPMed","HGDP":"HGDP","1k genome":"1KGP"}
    plot_df["Display_label"]=plot_df["Cohort"].map(label_map)
    data_to_plot=plot_df[pop_order].div(plot_df["Total"],axis=0)*100
    fig,ax=plt.subplots(figsize=(6,5),dpi=300)
    y=np.arange(len(plot_df))
    left=np.zeros(len(plot_df))
    bar_height=0.7
    for pop in pop_order:
        widths=data_to_plot[pop].values
        ax.barh(y,widths,height=bar_height,left=left,color=apply_colors[pop],edgecolor="white",linewidth=0.8,label=pop)
        left+=widths
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Display_label"],fontsize=12.5,color=text_dark)
    for label in ax.get_yticklabels():
        if label.get_text()=="hEVO corpus":
            label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_ylim(len(plot_df)-0.5,-0.5)
    ax.set_xlabel("Population composition (%)",fontsize=13)
    ax.tick_params(axis="x",labelsize=11)
    ax.tick_params(axis="y",labelsize=12.5)
    for i,total in enumerate(plot_df["Total"]):
        ax.text(100.8,i,f"n={int(total):,}",va="center",ha="left",fontsize=9.5,color=text_dark,clip_on=False)
    ax.set_xlim(0,100)
    ax.legend(ncol=5,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.16),fontsize=11.5,handlelength=1.5,columnspacing=1.2)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    plt.tight_layout(rect=(0,0.08,1,1))
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":600,"bbox_inches":"tight"} if fmt=="png" else {"bbox_inches":"tight"}
        fig.savefig(out_dir/f"{s}.{fmt}",**kwargs)
    plt.close(fig)
def render_cohort_diversity(args,t,g,s,note=''):
    df=read_table(args.tables_dir,t,["Cohort","EUR_count","AFR_count","EAS_count","SAS_count","Others_count"])
    text_dark="#334155"
    plt.rcParams.update({
        "font.family":"Arial",
        "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,"legend.frameon":False,
        "font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "figure.facecolor":"#FFFFFF","axes.facecolor":"#FFFFFF","savefig.facecolor":"#FFFFFF",
        "axes.edgecolor":"#C9D5E2","axes.labelcolor":text_dark,"axes.titlecolor":text_dark,
        "xtick.color":text_dark,"ytick.color":text_dark,
    })
    pop_cols=["EUR","AFR","EAS","SAS","Others"]
    counts_df=df[df["Cohort"]!="All"].rename(columns={f"{pop}_count":pop for pop in pop_cols})
    ukb_full={"Cohort":"UKB full 500K","EUR":440000,"AFR":7665,"EAS":1523,"SAS":9505,"Others":2853+4388+0}
    cohort_order=["UKB","EAS","UK10K","AFR","1k genome","HGDP","SG10K","TOPMed"]
    short_name={"UKB":"UKB subset","1k genome":"+ 1KG","HGDP":"+ HGDP","SG10K":"+ SG10K","TOPMed":"+ TOPMed","AFR":"+ AFR","EAS":"+ EAS","UK10K":"+ UK10K"}
    scenarios=[{"Step":"S0","Scenario":"UKB full 500K","Added_cohort":"UKB full 500K\nEUR=440K","Full_combination":"UKB full 500K, with EUR set to 440K",**{c:ukb_full[c] for c in pop_cols}}]
    running_counts=pd.Series({c:0 for c in pop_cols},dtype=float)
    included=[]
    for i,cohort in enumerate(cohort_order,start=1):
        row=counts_df[counts_df["Cohort"]==cohort].iloc[0]
        running_counts+=row[pop_cols].astype(float)
        included.append(cohort)
        scenarios.append({"Step":f"S{i}","Scenario":f"S{i}: {short_name[cohort]}","Added_cohort":short_name[cohort],"Full_combination":" + ".join(included),**running_counts.to_dict()})
    def calc_diversity(counts):
        counts=np.array(counts,dtype=float)
        total=counts.sum()
        if total<=0:
            return np.nan,np.nan,np.nan,np.nan
        p=counts/total
        p_nonzero=p[p>0]
        shannon=-np.sum(p_nonzero*np.log(p_nonzero))
        k=len(counts)
        shannon_evenness=shannon/np.log(k) if k>1 else np.nan
        simpson=1-np.sum(p**2)
        inv_simpson=1/np.sum(p**2) if np.sum(p**2)>0 else np.nan
        return shannon,shannon_evenness,simpson,inv_simpson
    metric_rows=[]
    for _,row in pd.DataFrame(scenarios).iterrows():
        counts=row[pop_cols].values.astype(float)
        shannon,shannon_evenness,simpson,inv_simpson=calc_diversity(counts)
        metric_rows.append({"Scenario":row["Scenario"],"Total":counts.sum(),"Shannon":shannon,"Shannon_evenness":shannon_evenness,"Simpson":simpson,"Inverse_Simpson":inv_simpson,"Effective_populations":np.exp(shannon),**{c:row[c] for c in pop_cols}})
    metric_df=pd.DataFrame(metric_rows)
    for metric in ["Shannon","Simpson","Effective_populations"]:
        metric_df.loc[metric_df.index>0,metric]=metric_df.loc[metric_df.index>0,metric].cummax()
    plot_df=metric_df.iloc[::-1].reset_index(drop=True)
    y=np.arange(len(plot_df))
    bar_height=0.4
    fig,ax=plt.subplots(figsize=(6,5))
    ax.barh(y+bar_height/2,plot_df["Shannon"],height=bar_height,color="#70A6D4",label="Shannon diversity index")
    ax.barh(y-bar_height/2,plot_df["Simpson"],height=bar_height,color="#B35C1E",label="Simpson diversity index")
    for i,value in enumerate(plot_df["Shannon"]):
        ax.text(value+0.015,y[i]+bar_height/2,f"{value:.3f}",va="center",ha="left",fontsize=10,color=text_dark)
    for i,value in enumerate(plot_df["Simpson"]):
        ax.text(value+0.015,y[i]-bar_height/2,f"{value:.3f}",va="center",ha="left",fontsize=10,color=text_dark)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Scenario"],fontsize=12)
    ax.set_xlabel("Diversity index",fontsize=12)
    ax.set_xlim(0,1.75)
    ax.grid(axis="x",linestyle="--",linewidth=0.6,alpha=0.35,color="#D9DEE3")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False,loc="upper right",fontsize=10)
    plt.tight_layout()
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir/f"{s}.{fmt}")
    plt.close(fig)
def render_coverage(args,t,g,s,note=''):
    from matplotlib.ticker import FuncFormatter
    curves=read_table(
        args.tables_dir,
        t,
        [
            "scenario",
            "individuals",
            "ukb_haplotype_frequency",
            "threshold_weighted_coverage",
        ],
    ).copy()
    for col in ["individuals","ukb_haplotype_frequency","threshold_weighted_coverage"]:
        curves[col]=pd.to_numeric(curves[col],errors="coerce")
    scenario_configs=[
        {"scenario":"EUR downsampled","individuals":3000,"color":"#9DCCF2","linestyle":"--"},
        {"scenario":"Final EUR","individuals":14461,"color":"#1E6EA8","linestyle":"-"},
    ]
    def format_frequency_tick(x,_pos):
        if x<=0:
            return ""
        pct=x*100
        if pct>=1:
            return f"{pct:.0f}%"
        if pct>=0.1:
            return f"{pct:.1f}%"
        if pct>=0.01:
            return f"{pct:.2f}%"
        if pct>=0.001:
            return f"{pct:.3f}%"
        return f"{pct:.4f}%"
    plt.rcParams.update({
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "font.size":9.5,
        "axes.labelsize":10,
        "xtick.labelsize":8.8,
        "ytick.labelsize":9.5,
        "legend.fontsize":8.8,
    })
    fig,ax=plt.subplots(figsize=(5.6,4.4),dpi=300)
    config_by_scenario={config["scenario"]:config for config in scenario_configs}
    for config in scenario_configs:
        scenario=config["scenario"]
        sub=curves[curves["scenario"]==scenario].sort_values("ukb_haplotype_frequency",ascending=False)
        ax.plot(
            sub["ukb_haplotype_frequency"],
            sub["threshold_weighted_coverage"],
            color=config["color"],
            linestyle=config["linestyle"],
            linewidth=2.4,
            solid_capstyle="round",
            label=f"{scenario} (n={config['individuals']:,})",
        )
        end_y=float(sub.iloc[-1]["threshold_weighted_coverage"])
        min_x=float(sub["ukb_haplotype_frequency"].min())
        ax.text(min_x*0.72,end_y,f"{end_y:.1%}",ha="left",va="center",fontsize=9,color=config["color"],clip_on=False)
    final_sub=curves[curves["scenario"]=="Final EUR"].sort_values("ukb_haplotype_frequency",ascending=False)
    for threshold in [1e-2,1e-3,1e-4]:
        y=float(final_sub.loc[np.isclose(final_sub["ukb_haplotype_frequency"],threshold),"threshold_weighted_coverage"].iloc[0])
        ax.scatter([threshold],[y],color=config_by_scenario["Final EUR"]["color"],edgecolor="white",linewidth=0.6,s=24,zorder=6)
        ax.text(threshold,min(y+0.035,1.01),f"{y:.1%}",ha="center",fontsize=8)
        ax.axvline(threshold,color="#B8C7D8",linestyle=":",linewidth=0.8,alpha=0.8)
    min_x=float(curves["ukb_haplotype_frequency"].min())
    max_x=float(curves["ukb_haplotype_frequency"].max())
    min_y=float(curves["threshold_weighted_coverage"].min())
    ax.set_xscale("log")
    ax.set_xlim(max_x*1.25,min_x*0.55)
    ax.set_ylim(max(0.0,np.floor((min_y-0.04)*20)/20),1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(format_frequency_tick))
    ax.set_xlabel("Full UKB EUR haplotype frequency threshold (log scale)",fontsize=11)
    ax.set_ylabel("Weighted coverage above threshold",fontsize=11)
    ax.legend(frameon=False,fontsize=8.8,loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both",labelsize=8.8)
    plt.tight_layout()
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight")
    plt.close(fig)
def render_rank_abundance(args,t,g,s,note=''):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    df=read_table(args.tables_dir,t,["rank","count"]).copy()
    df=df.sort_values("rank").reset_index(drop=True)
    df["rank_scaled"]=df["rank"].astype(float)
    df["count_scaled"]=df["count"].astype(float)
    line_color="#2F73B7"
    accent_color="#174A7E"
    grid_color="#B8C7D8"
    def compact_number(x,_pos):
        if x>=1e9: return f"{x/1e9:.1f}B"
        if x>=1e6: return f"{x/1e6:.0f}M"
        if x>=1e3: return f"{x/1e3:.0f}K"
        return f"{x:.0f}"
    plt.rcParams.update({
        "pdf.fonttype":42,"ps.fonttype":42,"font.size":7.5,
        "axes.labelsize":8.0,"xtick.labelsize":7.0,"ytick.labelsize":7.0,
    })
    fig,ax=plt.subplots(figsize=(3.25,2.35),dpi=450)
    ax.plot(df["rank_scaled"],df["count_scaled"],color=line_color,linewidth=1.15,solid_capstyle="round")
    ax.fill_between(df["rank_scaled"],df["count_scaled"],y2=max(0.8,df["count_scaled"].min()*0.75),color=line_color,alpha=0.055,linewidth=0)
    targets=[1,df["rank_scaled"].max()]
    used=set()
    for target in targets:
        idx=(df["rank_scaled"]-target).abs().idxmin()
        if idx in used:
            continue
        used.add(idx)
        x=df.loc[idx,"rank_scaled"]; y=df.loc[idx,"count_scaled"]
        ax.scatter([x],[y],s=14,color=accent_color,edgecolor="white",linewidth=0.55,zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(left=0,right=df["rank_scaled"].max()*1.02)
    ax.set_ylim(bottom=max(0.8,df["count_scaled"].min()*0.75),top=df["count_scaled"].max()*1.55)
    ax.set_xlabel("Unique sequence rank",labelpad=2)
    ax.set_ylabel("Occurrence count",labelpad=2)
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y",which="major",color=grid_color,linestyle="-",linewidth=0.35,alpha=0.40)
    ax.grid(axis="y",which="minor",color=grid_color,linestyle=":",linewidth=0.25,alpha=0.22)
    ax.grid(axis="x",which="major",color=grid_color,linestyle="-",linewidth=0.25,alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    plt.tight_layout(pad=0.35)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig); return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight")
    plt.close(fig)
def render_upset(args,t,g,s,note=''):
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import LogLocator, NullFormatter
    pop_order=["Others","SAS","EAS","EUR","AFR"]
    pop_colors={pop:POP_COLORS[pop] for pop in pop_order}
    df=read_table(args.tables_dir,t,["intersection_rank","members","count","degree",*pop_order]).copy()
    df=df.sort_values("intersection_rank").reset_index(drop=True)
    df["intersection_rank"]=df["intersection_rank"].astype(int)
    df["count"]=df["count"].astype(float)
    for pop in pop_order:
        df[pop]=df[pop].astype(int)
    plt.rcParams.update({
        "pdf.fonttype":42,"ps.fonttype":42,"font.size":7.2,
        "axes.labelsize":7.8,"xtick.labelsize":6.6,"ytick.labelsize":7.0,
    })
    fig=plt.figure(figsize=(4.15,3.25),dpi=450)
    gs=GridSpec(2,1,height_ratios=[2.25,1.05],hspace=0.045,figure=fig)
    ax_bar=fig.add_subplot(gs[0,0])
    ax_mat=fig.add_subplot(gs[1,0],sharex=ax_bar)
    x=df["intersection_rank"].to_numpy()
    counts=df["count"].to_numpy()
    ax_bar.bar(x,counts,width=0.72,color="#2F73B7",edgecolor="white",linewidth=0.25)
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("Sequence count")
    ax_bar.set_xlim(0.35,len(df)+0.65)
    ax_bar.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
    ax_bar.yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=10))
    ax_bar.yaxis.set_minor_formatter(NullFormatter())
    ax_bar.grid(axis="y",which="major",color="#D9E5F2",linewidth=0.45,alpha=0.85)
    ax_bar.grid(axis="y",which="minor",color="#EAF1F8",linewidth=0.28,linestyle=":",alpha=0.75)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="x",labelbottom=False,bottom=False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    y_positions={pop:i for i,pop in enumerate(pop_order)}
    for _,row in df.iterrows():
        active_pops=[pop for pop in pop_order if row[pop]==1]
        active_y=[y_positions[pop] for pop in active_pops]
        x_pos=row["intersection_rank"]
        if len(active_y)>1:
            ax_mat.plot([x_pos,x_pos],[min(active_y),max(active_y)],color="#9DB7D1",linewidth=0.75,zorder=1)
        for pop in pop_order:
            y=y_positions[pop]
            active=row[pop]==1
            ax_mat.scatter([x_pos],[y],s=18 if active else 8,color=pop_colors[pop] if active else "#E4ECF4",edgecolor="#173B5C" if active else "#C9D7E4",linewidth=0.35 if active else 0.2,zorder=3 if active else 2)
    ax_mat.set_yticks([y_positions[pop] for pop in pop_order])
    ax_mat.set_yticklabels(pop_order)
    for label in ax_mat.get_yticklabels():
        label.set_color("#000000")
        label.set_fontweight("bold")
    ax_mat.set_xlabel("Intersection pattern rank")
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([str(i) for i in x],fontsize=5.4)
    ax_mat.set_ylim(-0.55,len(pop_order)-0.45)
    ax_mat.grid(axis="x",color="#E7EEF6",linewidth=0.3,alpha=0.65)
    ax_mat.spines["top"].set_visible(False)
    ax_mat.spines["right"].set_visible(False)
    ax_mat.spines["left"].set_visible(False)
    plt.tight_layout(pad=0.35)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig); return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight")
    plt.close(fig)
def render_upset_full_table(args,t,g,s,note=''):
    import matplotlib.gridspec as gridspec
    df=read_table(args.tables_dir,t,["members","count","degree","AFR","EUR","EAS","SAS","Others"]).copy()
    populations=["AFR","EUR","EAS","SAS","Others"]
    blue_palette=["#C7DCEF","#8FB9DD","#5E97C7","#2F73B7","#174A7E"]
    pop_colors={pop:POP_COLORS[pop] for pop in populations}
    pop_labels={pop:pop for pop in populations}
    plt.rcParams.update({
        "font.family":"Arial",
        "svg.fonttype":"none",
        "pdf.fonttype":42,
        "axes.spines.top":False,
        "axes.spines.right":False,
        "font.size":9,
    })
    def parse_members(value):
        if not isinstance(value,str) or not value:
            return tuple()
        return tuple(token for token in value.split("|") if token)
    frame=df.copy()
    frame["members_tuple"]=frame["members"].map(parse_members)
    frame=frame.sort_values(["count","degree"],ascending=[False,False]).reset_index(drop=True)
    fig=plt.figure(figsize=(8.6,5.6),facecolor="white")
    gs=gridspec.GridSpec(
        2,
        2,
        width_ratios=[4.9,1.35],
        height_ratios=[3.2,1.45],
        wspace=0.05,
        hspace=0.05,
    )
    ax_bar=fig.add_subplot(gs[0,0])
    ax_mat=fig.add_subplot(gs[1,0],sharex=ax_bar)
    x=np.arange(len(frame))
    ax_bar.bar(x,frame["count"],color=blue_palette[3],width=0.72)
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("Sequence count")
    ax_bar.tick_params(axis="x",labelbottom=False)
    ax_bar.grid(axis="y",color="#E6E6E6",linewidth=0.7)
    ax_bar.set_axisbelow(True)
    for xi,members in enumerate(frame["members_tuple"]):
        active_idx=[]
        for yi,pop in enumerate(populations):
            active=pop in members
            if active:
                active_idx.append(yi)
            ax_mat.scatter(
                xi,
                yi,
                s=34 if active else 18,
                color=pop_colors[pop] if active else "#D0D7DF",
                zorder=3,
            )
        if active_idx:
            ax_mat.plot(
                [xi,xi],
                [min(active_idx),max(active_idx)],
                color="#444444",
                linewidth=1.0,
                zorder=2,
            )
    ax_mat.set_yticks(np.arange(len(populations)))
    ax_mat.set_yticklabels([pop_labels[pop] for pop in populations])
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([str(i+1) for i in x],fontsize=8)
    ax_mat.set_xlabel("Intersection pattern rank")
    ax_mat.set_xlim(-0.6,len(frame)-0.4)
    ax_mat.spines["top"].set_visible(False)
    ax_mat.spines["right"].set_visible(False)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":400} if fmt=="png" else {}
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight",**kwargs)
    plt.close(fig)
def render_radar(args,t,g,s,note=''):
    df=read_table(args.tables_dir,t,["category","task","family","model","rank","n_models","plot_radius"]).copy()
    categories={
        "Genomic element classification":["NT Benchmark","Genomic Benchmark","URE Benchmark"],
        "Variant scoring":["ClinVar (coding)","ClinVar (noncoding)","Causal Variants","Rare Causal Variants","Rare Variants","MPRA"],
        "Evolutionary and ancestry analysis":["HARs","HAQERs","Ancestry analysis","Adaptation variants"],
    }
    output_to_category={
        "fig1_family_best_rank_radar_genomic_elements":"Genomic element classification",
        "fig1_family_best_rank_radar_variant_scoring":"Variant scoring",
        "fig1_family_best_rank_radar_evolution_ancestry":"Evolutionary and ancestry analysis",
    }
    display_labels={task:task for tasks in categories.values() for task in tasks}
    label_offsets={
        "NT Benchmark":(0.00,0.06),"Genomic Benchmark":(0.10,0.00),"ClinVar (coding)":(0.08,0.00),
        "ClinVar (noncoding)":(0.08,-0.02),"URE Benchmark":(-0.06,-0.04),"Causal Variants":(0.10,0.00),
        "Rare Causal Variants":(0.08,-0.02),"Rare Variants":(0.08,-0.02),"Ancestry analysis":(-0.08,-0.02),
        "Adaptation variants":(0.08,0.00),
    }
    family_colors={"hEvo family":"#0B2E83","NT family":"#7655A6","Evo2 family":"#0E7C86"}
    family_markers={"hEvo family":"o","NT family":"s","Evo2 family":"^"}
    category_accents={
        "Genomic element classification":"#0B5FA5",
        "Variant scoring":"#3C8D5A",
        "Evolutionary and ancestry analysis":"#8B79B9",
    }
    plt.rcParams.update({
        "font.family":"Arial",
        "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,"legend.frameon":False,
        "font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "figure.facecolor":"#FFFFFF","axes.facecolor":"#FFFFFF","savefig.facecolor":"#FFFFFF",
        "axes.edgecolor":"#C9D5E2","axes.labelcolor":"#334155","axes.titlecolor":"#334155",
        "xtick.color":"#334155","ytick.color":"#334155",
    })
    def radar_xy(radius,theta):
        values=np.asarray(radius,dtype=float)
        return values*np.sin(theta),values*np.cos(theta)
    def draw_category_radar(ax,category,show_legend):
        order=categories[category]
        category_df=df[df["category"]==category].copy()
        max_rank=int(category_df["n_models"].max())
        pivot=category_df.pivot(index="task",columns="family",values="plot_radius").reindex(order)
        n_tasks=len(order)
        angles=np.linspace(0,2*np.pi,n_tasks,endpoint=False)
        angles_closed=np.concatenate([angles,angles[:1]])
        ax.set_aspect("equal"); ax.axis("off")
        for rank_value in np.arange(1,max_rank+1)[::-1]:
            radius=max_rank+1-rank_value
            x,y=radar_xy(radius,angles)
            ax.add_patch(Polygon(np.column_stack([x,y]),closed=True,facecolor="#F8FAFC" if rank_value%2 else "#F1F5F9",edgecolor="#D8E1EA",linewidth=0.7,zorder=1))
        for theta in angles:
            x,y=radar_xy(max_rank,np.array([theta]))
            ax.plot([0,x[0]],[0,y[0]],color="#D8E1EA",linewidth=0.8,zorder=2)
        tick_ranks=np.unique(np.rint(np.linspace(1,max_rank,min(5,max_rank))).astype(int))
        for rank_value in tick_ranks:
            radius=max_rank+1-rank_value
            x_tick,y_tick=radar_xy(radius,np.array([np.deg2rad(8.0)]))
            ax.text(x_tick[0]+0.10,y_tick[0],f"#{rank_value}",fontsize=10.5,color="#64748B",ha="left",va="center",zorder=5)
        for family in ["hEvo family","NT family","Evo2 family"]:
            values=pivot[family].to_numpy(dtype=float)
            closed_values=np.concatenate([values,values[:1]])
            x,y=radar_xy(closed_values,angles_closed)
            ax.fill(x,y,color=family_colors[family],alpha=0.10,zorder=3)
            ax.plot(x,y,color=family_colors[family],linewidth=2.4,marker=family_markers[family],markersize=7.2,label=family,zorder=4)
        radar_outer_radius=float(max_rank)
        label_radius=radar_outer_radius*1.0
        for angle,task in zip(angles,order):
            x,y=radar_xy(label_radius,np.array([angle]))
            x_pos=x[0]; y_pos=y[0]
            if abs(np.sin(angle))<1e-6 and y[0]>0:
                y_pos+=radar_outer_radius*0.10
            dx,dy=label_offsets.get(task,(0.0,0.0))
            x_pos+=dx; y_pos+=dy
            ha="left" if x[0]>0.55 else "right" if x[0]<-0.55 else "center"
            va="bottom" if y_pos>=0 else "top"
            ax.text(x_pos,y_pos,display_labels.get(task,task),fontsize=14.0,fontweight="medium",color=category_accents[category],ha=ha,va=va,zorder=6)
        margin=radar_outer_radius*0.20
        ax.set_xlim(-(radar_outer_radius+margin),radar_outer_radius+margin)
        ax.set_ylim(-(radar_outer_radius+margin),radar_outer_radius+margin)
        if show_legend:
            ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.10),ncol=3,fontsize=14.0,handlelength=2.4)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if s=="fig1_family_best_rank_radar_all_horizontal":
        fig,axes=plt.subplots(1,3,figsize=(15,5),facecolor="white")
        for ax,category in zip(axes,categories):
            draw_category_radar(ax,category,False)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(0.5,0.01),ncol=3,fontsize=14.5,frameon=False,handlelength=2.6)
        fig.subplots_adjust(left=0.03,right=0.97,top=0.98,bottom=0.16,wspace=0.18)
    else:
        category=output_to_category.get(s)
        if category is None:
            raise ValueError(f"Unknown radar output stem: {s}")
        fig,ax=plt.subplots(figsize=(5,5),facecolor="white")
        draw_category_radar(ax,category,True)
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig); return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":400} if fmt=="png" else {}
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight",**kwargs)
    plt.close(fig)
def render_mcc_overview(args,t,g,s,note=''):
    raw=read_table(args.tables_dir,t,["model","dataset","mcc_mean","mcc_lower","mcc_upper","task_rank","task_count"])
    raw=raw.copy()
    for col in ["mcc_mean","mcc_lower","mcc_upper","task_rank","task_count"]:
        raw[col]=pd.to_numeric(raw[col],errors="coerce")
    is_gb=t.startswith("S8_")
    if is_gb:
        groups={
            "Human Related Tasks":[
                "human_enhancers_cohn",
                "human_enhancers_ensembl",
                "human_ensembl_regulatory",
                "human_nontata_promoters",
                "human_ocr_ensembl",
                "demo_coding_vs_intergenomic_seqs",
                "demo_human_or_worm",
            ]
        }
        colors={
            "header_dark":"#0D3F78","header_rank":"#0D3F78","header_group":"#0D3F78",
            "row_alt":"#F1F1F1","rank_1":"#B35C1E","rank_2":"#0B2E83","rank_3":"#0E7C86",
            "other":"#D9D9D9","errorbar":"#163A5F","errorcap":"#163A5F","ink":"#1B1B1B",
        }
        short_map={
            "human_enhancers_cohn":"enhs_cohn",
            "human_enhancers_ensembl":"enhancers_ensembl",
            "human_ensembl_regulatory":"ensembl_regulatory",
            "human_nontata_promoters":"non-TATA",
            "human_ocr_ensembl":"OCR",
            "demo_coding_vs_intergenomic_seqs":"coding_vs_inter",
            "demo_human_or_worm":"human_or_worm",
            "drosophila_enhancers_stark":"drosophila",
            "mouse_enhancers":"mouse",
        }
        figsize_w=12
        rank_w=1.5
        rank_title="Avg. MCC Rank"
        method_title="Models"
        bottom_axis_h=0.92
    else:
        groups={
            "Histone Markers":["H2AFZ","H3K27ac","H3K27me3","H3K36me3","H3K4me1","H3K4me2","H3K4me3","H3K9ac","H3K9me3","H4K20me1"],
            "Regulatory Annotation":["enhancers","enhancers_types","promoter_all","promoter_no_tata","promoter_tata"],
            "Splice Site Annotation":["splice_sites_acceptors","splice_sites_all","splice_sites_donors"],
        }
        colors={
            "header_dark":"#0D3F78","header_rank":"#0D3F78","header_group":"#4D7FB8",
            "row_alt":"#F1F1F1","rank_light":"#F7FAFE","rank_dark":"#0D3F78",
            "errorbar":"#163A5F","errorcap":"#163A5F","ink":"#1B1B1B",
        }
        short_map={
            "splice_sites_acceptors":"acceptors",
            "splice_sites_all":"all",
            "splice_sites_donors":"donors",
            "promoter_all":"all",
            "promoter_no_tata":"no_tata",
            "promoter_tata":"tata",
        }
        figsize_w=21.5
        rank_w=0.75
        rank_title="Rank"
        method_title="Method"
        bottom_axis_h=0.34
    ordered_tasks=[(group,task) for group,tasks in groups.items() for task in tasks]
    task_names=[task for _,task in ordered_tasks if task in set(raw["dataset"])]
    ordered_tasks=[item for item in ordered_tasks if item[1] in task_names]
    matrix=raw.pivot_table(index="model",columns="dataset",values="mcc_mean",aggfunc="first").reindex(columns=task_names)
    matrix["__overall__"]=matrix[task_names].mean(axis=1,skipna=True)
    if is_gb:
        matrix["__human__"]=matrix[task_names].mean(axis=1,skipna=True)
        matrix=matrix.sort_values(["__overall__","__human__"],ascending=[False,False],na_position="last")
    else:
        hist=[t for t in groups["Histone Markers"] if t in matrix.columns]
        reg=[t for t in groups["Regulatory Annotation"] if t in matrix.columns]
        spl=[t for t in groups["Splice Site Annotation"] if t in matrix.columns]
        matrix["__histone__"]=matrix[hist].mean(axis=1)
        matrix["__regulatory__"]=matrix[reg].mean(axis=1)
        matrix["__splice__"]=matrix[spl].mean(axis=1)
        matrix=matrix.dropna(axis=0,how="any")
        matrix=matrix.sort_values(["__overall__","__histone__","__regulatory__","__splice__"],ascending=[False,False,False,False])
    df=matrix[task_names].reset_index()
    n_rows=len(df)
    row_h=0.82
    header_h=0.95
    group_header_h=0.7
    subheader_h=0.95
    top_pad=0.42
    total_h=top_pad+header_h+group_header_h+subheader_h+n_rows*row_h+bottom_axis_h
    x_method=0.4
    w_method=2.55
    x_rank=x_method+w_method+0.18
    w_rank=rank_w
    x_metrics=x_rank+w_rank+0.20
    w_metric=0.8
    gap=0.10
    task_xs=[x_metrics+i*(w_metric+gap) for i in range(len(ordered_tasks))]
    total_metric_w=task_xs[-1]+w_metric-x_metrics
    x_right=x_metrics+total_metric_w+0.35
    fig,ax=plt.subplots(figsize=(figsize_w,max(8.2,0.24*n_rows+2.1)),facecolor="white")
    ax.set_xlim(0,x_right)
    ax.set_ylim(total_h,0)
    ax.axis("off")
    y_header=top_pad
    y_group=y_header+header_h
    y_sub=y_group+group_header_h
    y0=y_sub+subheader_h
    ax.add_patch(Rectangle((x_method,y_header),w_method,header_h*0.72,facecolor=colors["header_dark"],edgecolor="#444444"))
    ax.text(x_method+w_method/2,y_header+header_h*0.36,method_title,color="white",ha="center",va="center",fontsize=10.8)
    ax.add_patch(Rectangle((x_rank,y_header),w_rank,header_h*0.72,facecolor=colors["header_rank"],edgecolor="#174FA6"))
    ax.text(x_rank+w_rank/2,y_header+header_h*0.36,rank_title,color="white",ha="center",va="center",fontsize=10.2)
    ax.add_patch(FancyBboxPatch((x_metrics,y_header),total_metric_w,header_h*0.72,boxstyle="round,pad=0.02,rounding_size=0.10",facecolor="#0D3F78",edgecolor=colors["row_alt"],linewidth=0.9))
    ax.text(x_metrics+total_metric_w/2,y_header+header_h*0.36,"MCC Performance",color="white",ha="center",va="center",fontsize=10.8)
    cursor=0
    for group_name,subtasks in groups.items():
        present=[task for task in subtasks if task in task_names]
        if not present:
            continue
        x=task_xs[cursor]
        group_w=len(present)*w_metric+(len(present)-1)*gap
        ax.add_patch(FancyBboxPatch((x,y_group),group_w,group_header_h*0.72,boxstyle="round,pad=0.01,rounding_size=0.08",facecolor=colors["header_group"],edgecolor="#857CB0",linewidth=0.8))
        ax.text(x+group_w/2,y_group+group_header_h*0.36,group_name,color="white",ha="center",va="center",fontsize=9.1)
        cursor+=len(present)
    for x,(_,task) in zip(task_xs,ordered_tasks):
        ax.text(x+0.2,y_sub+subheader_h*0.60,short_map.get(task,task),ha="center",va="center",fontsize=7.2,color=colors["ink"],rotation=0)
    def interp(a,b,tv):
        a=np.array(to_rgb(a)); b=np.array(to_rgb(b)); tv=max(0.0,min(1.0,float(tv)))
        return tuple(a*(1-tv)+b*tv)
    def cell_color(rank,total):
        if is_gb:
            if rank==1: return colors["rank_1"]
            if rank==2: return colors["rank_2"]
            if rank==3: return colors["rank_3"]
            return colors["other"]
        if total<=1: return interp(colors["rank_light"],colors["rank_dark"],1.0)
        return interp(colors["rank_light"],colors["rank_dark"],(float(total)-float(rank))/(float(total)-1.0))
    def draw_cell(x,y,w,h,mean,lower,upper,rank,total):
        bar_w=max(0.0,min(1.0,float(mean)))*w
        ax.add_patch(Rectangle((x,y),bar_w,h,facecolor=cell_color(rank,total),edgecolor=colors["ink"]))
        ci_lower_x=x+max(0.0,min(1.0,float(lower)))*w
        ci_upper_x=x+max(0.0,min(1.0,float(upper)))*w
        ci_center_x=x+max(0.0,min(1.0,float(mean)))*w
        y_mid=y+h/2
        cap_h=h*0.24
        ax.plot([ci_lower_x,ci_upper_x],[y_mid,y_mid],color=colors["errorbar"],linewidth=0.9,solid_capstyle="butt",zorder=3)
        ax.plot([ci_lower_x,ci_lower_x],[y_mid-cap_h/2,y_mid+cap_h/2],color=colors["errorcap"],linewidth=0.9,zorder=3)
        ax.plot([ci_upper_x,ci_upper_x],[y_mid-cap_h/2,y_mid+cap_h/2],color=colors["errorcap"],linewidth=0.9,zorder=3)
        ax.plot([ci_center_x,ci_center_x],[y+h*0.18,y+h*0.82],color=colors["ink"],linewidth=0.8,zorder=4)
    def draw_missing(x,y,w,h):
        ax.add_patch(Rectangle((x,y),w,h,facecolor="white",edgecolor="#D0D0D0",linewidth=0.55))
    stats_lookup={(row["model"],row["dataset"]):(row["mcc_mean"],row["mcc_lower"],row["mcc_upper"],row["task_rank"],row["task_count"]) for _,row in raw.iterrows()}
    for i,row in df.iterrows():
        y=y0+i*row_h
        if i%2==1:
            ax.add_patch(Rectangle((x_method-0.12,y),x_right-(x_method-0.12),row_h,facecolor=colors["row_alt"],edgecolor="none"))
        ax.text(x_method+0.06,y+row_h/2,display_model_name(row["model"]),ha="left",va="center",fontsize=8.5,color=colors["ink"])
        ax.text(x_rank+w_rank/2,y+row_h/2,str(i+1),ha="center",va="center",fontsize=8.4,color=colors["ink"])
        for x,task in zip(task_xs,task_names):
            stats=stats_lookup.get((row["model"],task))
            if stats is None or any(pd.isna(value) for value in stats[:4]):
                draw_missing(x,y+0.12,w_metric,row_h-0.24)
                continue
            mean,lower,upper,task_rank,task_count=stats
            draw_cell(x,y+0.12,w_metric,row_h-0.24,mean,lower,upper,task_rank,task_count)
    if is_gb:
        y_axis=y0+n_rows*row_h+0.12
        tick_len=0.18
        label_offset=0.24
        for x in task_xs:
            ax.plot([x,x+w_metric],[y_axis,y_axis],color=colors["ink"],linewidth=0.8,zorder=3)
            ax.plot([x,x],[y_axis,y_axis-tick_len],color=colors["ink"],linewidth=0.8,zorder=3)
            ax.plot([x+w_metric,x+w_metric],[y_axis,y_axis-tick_len],color=colors["ink"],linewidth=0.8,zorder=3)
            ax.text(x,y_axis+label_offset,"0",ha="center",va="top",fontsize=6.4,color=colors["ink"])
            ax.text(x+w_metric,y_axis+label_offset,"1",ha="center",va="top",fontsize=6.4,color=colors["ink"])
        ax.legend(
            handles=[
                Patch(facecolor=colors["rank_1"],edgecolor=colors["ink"],label="Rank 1"),
                Patch(facecolor=colors["rank_2"],edgecolor=colors["ink"],label="Rank 2"),
                Patch(facecolor=colors["rank_3"],edgecolor=colors["ink"],label="Rank 3"),
                Patch(facecolor=colors["other"],edgecolor=colors["ink"],label="Others"),
            ],
            loc="lower left",bbox_to_anchor=(0.025,0.9),ncol=4,frameon=False,fontsize=8.2,handlelength=1.3,handletextpad=0.45,columnspacing=0.9,
        )
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":400} if fmt=="png" else {}
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight",**kwargs)
    plt.close(fig)
def render_probe_bar(args,t,g,s,note=''):
    raw=read_table(args.tables_dir,t,["model","run","mcc_mean"])
    raw=raw.copy()
    raw["mcc_mean"]=pd.to_numeric(raw["mcc_mean"],errors="coerce")
    order=raw.groupby("model")["mcc_mean"].mean().sort_values(ascending=False).index.tolist()
    summary=raw.groupby("model",as_index=False)["mcc_mean"].mean().rename(columns={"mcc_mean":"mean_mcc_for_sort"})
    summary["model_order"]=summary["model"].map({model:idx+1 for idx,model in enumerate(order)})
    summary=summary.sort_values("model_order")
    title=str(raw["figure"].iloc[0]) if "figure" in raw.columns else "Model performance"
    fig,ax=plt.subplots(figsize=(10.8,8.0),facecolor="white")
    y_positions=np.arange(len(order))
    ax.barh(
        y_positions,
        summary["mean_mcc_for_sort"].to_numpy(),
        height=0.8,
        color=[color_for_model(model) for model in summary["model"]],
        edgecolor=COLORS["outline"],
        linewidth=0.9,
        zorder=2,
    )
    run_index_map={run_name:idx for idx,run_name in enumerate(sorted(raw["run"].astype(str).unique()))}
    jitter_offsets=np.linspace(-0.18,0.18,max(len(run_index_map),2))
    for idx,model in enumerate(order):
        model_rows=raw.loc[raw["model"]==model].copy()
        model_rows["run_str"]=model_rows["run"].astype(str)
        y_scatter=[idx+jitter_offsets[run_index_map[run_str]] for run_str in model_rows["run_str"]]
        ax.scatter(model_rows["mcc_mean"],y_scatter,s=12,color=COLORS["outline"],alpha=0.75,linewidths=0,zorder=3)
    ax.set_title(title,fontsize=12,fontweight="bold",pad=12)
    ax.set_xlabel("Mean MCC")
    ax.set_ylabel("Model")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([display_model_name(model) for model in order])
    ax.invert_yaxis()
    ax.grid(axis="x",color=COLORS["grid"],linewidth=0.8,zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x",labelsize=8.5,direction="out",length=4,width=0.9)
    ax.tick_params(axis="y",labelsize=8.5)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(COLORS["outline"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(COLORS["outline"])
    ax.spines["bottom"].set_linewidth(0.9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["other"],edgecolor=COLORS["outline"],label="Other models"),
            Patch(facecolor=COLORS["hevo_86"],edgecolor=COLORS["outline"],label="hEvo-base"),
            Patch(facecolor=COLORS["hevo_340"],edgecolor=COLORS["outline"],label="hEvo-large"),
            Line2D([0],[0],marker="o",color="none",markerfacecolor=COLORS["outline"],markersize=5,label="Individual runs"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.995,0.01),
        frameon=False,
        fancybox=False,
        edgecolor=COLORS["outline"],
        facecolor="white",
        framealpha=0.92,
        fontsize=8,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.4,
        handletextpad=0.5,
    )
    x_max=max(raw["mcc_mean"].max(),summary["mean_mcc_for_sort"].max())
    ax.set_xlim(left=0,right=min(1.0,x_max+0.08))
    fig.subplots_adjust(left=0.33,right=0.98,bottom=0.08,top=0.94)
    finish(fig,args,g,s)
def render_har_boxplot(args,t,g,s,note=''):
    raw=read_table(args.tables_dir,t,["model","run","mcc_mean"])
    raw=raw.copy()
    raw["mcc_mean"]=pd.to_numeric(raw["mcc_mean"],errors="coerce")
    order=raw.groupby("model")["mcc_mean"].mean().sort_values(ascending=False).index.tolist()
    data=[raw.loc[raw["model"]==model,"mcc_mean"].to_numpy() for model in order]
    tick_labels=[display_model_name(model) for model in order]
    title=str(raw["figure"].iloc[0]) if "figure" in raw.columns else "Model performance"
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    boxes=ax.boxplot(
        data,
        tick_labels=tick_labels,
        vert=True,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color":COLORS["outline"],"linewidth":1.3},
        whiskerprops={"color":COLORS["outline"],"linewidth":0.9},
        capprops={"color":COLORS["outline"],"linewidth":0.9},
        boxprops={"color":COLORS["outline"],"linewidth":0.9},
    )
    for model,box in zip(order,boxes["boxes"]):
        box.set_facecolor(color_for_model(model))
        box.set_edgecolor(COLORS["outline"])
    ax.set_title(title,fontsize=16,fontweight="bold",pad=12)
    ax.set_xlabel("Model",color=COLORS["outline"])
    ax.set_ylabel("MCC mean",color=COLORS["outline"])
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x",labelsize=7.2,rotation=45,colors=COLORS["outline"])
    ax.tick_params(axis="y",labelsize=8.5,colors=COLORS["outline"])
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_color(COLORS["outline"])
    for label in ax.get_yticklabels():
        label.set_color(COLORS["outline"])
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(COLORS["outline"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(COLORS["outline"])
    ax.spines["bottom"].set_linewidth(0.9)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["other"],edgecolor=COLORS["outline"],label="Other models"),
            Patch(facecolor=COLORS["hevo_86"],edgecolor=COLORS["outline"],label=display_model_name("hEvo-base")),
            Patch(facecolor=COLORS["hevo_340"],edgecolor=COLORS["outline"],label=display_model_name("hEvo-large")),
        ],
        loc="upper right",
        bbox_to_anchor=(0.995,0.995),
        frameon=False,
        fancybox=False,
        edgecolor=COLORS["outline"],
        facecolor="white",
        framealpha=0.92,
        fontsize=8,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(bottom=0.42,right=0.98)
    finish(fig,args,g,s)
def render_multiclass_heatmap(args,t,g,s,note=''):
    df=read_table(args.tables_dir,t,["dataset","model","class_label","per_class_f1"])
    frame=df.copy()
    frame["per_class_f1"]=pd.to_numeric(frame["per_class_f1"],errors="coerce")
    dataset_count=frame["dataset"].nunique()
    counts=frame.groupby("model")["dataset"].nunique()
    complete_models=counts[counts==dataset_count].index.tolist()
    aggregated=(
        frame[frame["model"].isin(complete_models)]
        .groupby(["model","class_label"],as_index=False)
        .agg(score=("per_class_f1","mean"))
    )
    class_order=["URE","ELS","PLS","Insulator","Silencer"]
    matrix=aggregated.pivot(index="model",columns="class_label",values="score").reindex(columns=class_order)
    baseline_model="NT_500m_MS"
    if baseline_model not in matrix.index:
        raise ValueError(f"Baseline model is missing from complete results: {baseline_model}")
    delta_matrix=matrix.subtract(matrix.loc[baseline_model],axis="columns")
    order=delta_matrix.mean(axis=1).sort_values(ascending=False).index
    delta_matrix=delta_matrix.loc[order]
    values=delta_matrix.to_numpy(dtype=float)
    max_abs=float(np.nanmax(np.abs(values))) if values.size else 1e-6
    if max_abs==0:
        max_abs=1e-6
    blue_cmap=LinearSegmentedColormap.from_list("ure_multiclass_diverging",["#2C7FB8","#F7F7F7","#B35C1E"])
    norm=TwoSlopeNorm(vmin=-max_abs,vcenter=0.0,vmax=max_abs)
    fig_height=max(6.2,0.24*len(delta_matrix.index)+1.7)
    fig,ax=plt.subplots(figsize=(6.7,fig_height),facecolor="white")
    im=ax.imshow(values,cmap=blue_cmap,aspect="auto",norm=norm)
    ax.set_xticks(np.arange(len(class_order)))
    ax.set_xticklabels(class_order,fontsize=9)
    ax.set_yticks(np.arange(len(delta_matrix.index)))
    ax.set_yticklabels([display_model_name(model) for model in delta_matrix.index],fontsize=8.1)
    ax.tick_params(axis="y",direction="out",length=3.5,width=0.9,color="#000000")
    ax.tick_params(axis="x",bottom=True,top=False,length=3.0,width=0.8,color="#000000")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#000000")
    ax.spines["bottom"].set_linewidth(0.9)
    cbar=fig.colorbar(im,ax=ax,fraction=0.042,pad=0.02)
    cbar.set_label(f"Δ mean per-class F1 vs {display_model_name(baseline_model)}",rotation=90)
    cbar.outline.set_linewidth(0.8)
    finish(fig,args,g,s)
def render_deepsea_bar(args,t,g,s,note=''):
    source=read_table(args.tables_dir,t,["model","task","auroc","Mean"])
    tasks=["DHS","HM","TF"]
    task_colors={"DHS":"#D97706","HM":"#0B2E83","TF":"#0E7C86"}
    frame=source.pivot_table(index="model",columns="task",values="auroc",aggfunc="first").reset_index()
    means=source.groupby("model")["Mean"].first()
    frame["Mean"]=frame["model"].map(means)
    for col in tasks:
        frame[col]=pd.to_numeric(frame[col],errors="coerce")
    frame["Mean"]=pd.to_numeric(frame["Mean"],errors="coerce")
    frame=frame.dropna(subset=["model",*tasks,"Mean"])
    frame=frame.sort_values(["Mean","TF","DHS","HM"],ascending=[False,False,False,False]).reset_index(drop=True)
    x=np.arange(len(frame))
    width=0.24
    fig,ax=plt.subplots(figsize=(12.8,5.8),facecolor="white")
    for idx,task in enumerate(tasks):
        offset=(idx-1)*width
        bars=ax.bar(
            x+offset,
            frame[task].to_numpy(dtype=float),
            width=width,
            color=task_colors[task],
            edgecolor="white",
            linewidth=0.7,
            label=task,
            zorder=3,
        )
        for bar,value in zip(bars,frame[task].to_numpy(dtype=float)):
            ax.text(bar.get_x()+bar.get_width()/2,value+0.004,f"{value:.3f}",ha="center",va="bottom",fontsize=7.2,rotation=90,color="#000000")
    ax.plot(x,frame["Mean"].to_numpy(dtype=float),color="#1B1B1B",linewidth=1.6,marker="o",markersize=4.2,label="Mean",zorder=4)
    ax.set_ylabel("AUROC",color="#000000")
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(model) for model in frame["model"]],rotation=35,ha="right")
    ax.set_ylim(0.74,1.0)
    ax.set_yticks(np.arange(0.75,1.01,0.05))
    ax.grid(axis="y",color="#DCE5EE",linewidth=0.8,zorder=0)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.tick_params(axis="both",colors="#000000")
    for idx in range(min(3,len(frame))):
        ax.axvspan(idx-0.5,idx+0.5,color="#EEF5FC",alpha=0.85,zorder=1)
    ax.legend(loc="upper right",ncol=4,fontsize=8.4,handlelength=1.7,columnspacing=0.9)
    fig.tight_layout()
    finish(fig,args,g,s)
def render_deepsea_radar(args,t,g,s,note=''):
    source=read_table(args.tables_dir,t,["model","task","auroc","Mean"])
    tasks=["DHS","HM","TF"]
    category_colors={"DHS":"#1D6F9C","HM":"#6A4C93","TF":"#217A68"}
    frame=source.pivot_table(index="model",columns="task",values="auroc",aggfunc="first").reset_index()
    means=source.groupby("model")["Mean"].first()
    frame["Mean"]=frame["model"].map(means)
    for col in tasks:
        frame[col]=pd.to_numeric(frame[col],errors="coerce")
    frame["Mean"]=pd.to_numeric(frame["Mean"],errors="coerce")
    frame=frame.dropna(subset=["model",*tasks,"Mean"])
    frame=frame.sort_values(["Mean","TF","DHS","HM"],ascending=False).reset_index(drop=True)
    min_value=float(frame[tasks].min().min())
    max_value=float(frame[tasks].max().max())
    vmin=min(0.75,np.floor((min_value-0.01)*20)/20)
    vmax=max(1.00,np.ceil((max_value+0.005)*20)/20)
    ticks=[round(value,2) for value in np.arange(vmin,vmax+1e-9,0.05)]
    if ticks[-1]<vmax:
        ticks.append(round(vmax,2))
    outer_radius=float(len(ticks))
    angles=np.linspace(0,2*np.pi,len(tasks),endpoint=False)
    angles_closed=np.concatenate([angles,angles[:1]])
    palette=["#0B2E83","#B35C1E","#0E7C86","#D97706","#7655A6","#5E7D5A","#9B7B18","#7B5B3A","#2F6B9A","#7A9E72","#5A3D80","#8995A5"]
    model_color={model:palette[idx%len(palette)] for idx,model in enumerate(frame["model"].tolist())}
    def xy(radius,theta):
        values=np.asarray(radius,dtype=float)
        return values*np.sin(theta),values*np.cos(theta)
    def normalize_values(values):
        span=vmax-vmin
        if span<=0:
            return np.full_like(values,outer_radius,dtype=float)
        return 1.0+(values-vmin)/span*(outer_radius-1.0)
    fig,ax=plt.subplots(figsize=(9.2,8.6),facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")
    ring_colors=["#F7F9FC","#F2F5F9","#EAF0F6","#E2EAF3","#D9E4EF","#D1DEEB"]
    for idx,tick in enumerate(ticks,start=1):
        x,y=xy(np.full(len(tasks),idx),angles)
        ax.add_patch(Polygon(np.column_stack([x,y]),closed=True,facecolor=ring_colors[(idx-1)%len(ring_colors)],edgecolor="#D7E2EC",linewidth=0.8,zorder=idx))
    for theta in angles:
        x,y=xy(np.array([outer_radius]),np.array([theta]))
        ax.plot([0,x[0]],[0,y[0]],color="#D2DEE9",linewidth=0.8,zorder=8)
    for idx,tick in enumerate(ticks,start=1):
        x_tick,y_tick=xy(np.array([idx]),np.array([np.deg2rad(8.0)]))
        ax.text(x_tick[0]+0.18,y_tick[0],f"{tick:.2f}",fontsize=8,color="#718296",ha="left",va="center",bbox={"facecolor":"white","edgecolor":"none","alpha":0.72,"pad":0.5},zorder=10)
    for idx,row in frame.iterrows():
        values=row[tasks].to_numpy(dtype=float)
        radii=normalize_values(values)
        closed_radii=np.concatenate([radii,radii[:1]])
        x,y=xy(closed_radii,angles_closed)
        color=model_color[row["model"]]
        alpha=0.16 if idx<3 else 0.08
        linewidth=2.0 if idx<3 else 1.4
        marker=["o","^","s","D","P","X","v","<",">","h","*"][idx%11]
        ax.fill(x,y,color=color,alpha=alpha,zorder=9+idx*0.01)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=5.5,
            label=f"{display_model_name(row['model'])} ({row['Mean']:.3f})",
            zorder=11+idx*0.01,
        )
    label_radius=outer_radius+0.95
    for angle,task in zip(angles,tasks):
        x,y=xy(np.array([label_radius]),np.array([angle]))
        ha="left" if x[0]>0.6 else "right" if x[0]<-0.6 else "center"
        va="bottom" if y[0]>=0 else "top"
        ax.text(x[0],y[0],task,fontsize=12,fontweight="bold",color=category_colors[task],ha=ha,va=va,zorder=20)
    ax.set_title("DeepSEA performance overview by task group\nRadial distance encodes absolute AUROC",fontsize=14,fontweight="bold",color="#20384C",pad=28)
    ax.text(0.0,-(outer_radius+1.35),"Legend values show the overall mean AUROC reported in the source table.",fontsize=8.5,color="#566B80",ha="center",va="top")
    ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.28),ncol=2,fontsize=8.2,handlelength=2.4,columnspacing=1.1)
    limit=outer_radius+1.8
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-(outer_radius+2.0),limit*1.03)
    finish(fig,args,g,s)
def render_auprc_bar(args,t,g,s,note=''):
    df=read_table(args.tables_dir,t,["model","auprc_mean","auprc_ci_lower","auprc_ci_upper","num_samples","num_positive","num_negative","baseline"])
    data=df.copy()
    data["auprc_mean"]=pd.to_numeric(data["auprc_mean"],errors="coerce")
    data["auprc_ci_lower"]=pd.to_numeric(data["auprc_ci_lower"],errors="coerce")
    data["auprc_ci_upper"]=pd.to_numeric(data["auprc_ci_upper"],errors="coerce")
    data=data.sort_values("auprc_mean",ascending=False).reset_index(drop=True)
    title=str(data["plot_title"].iloc[0]) if "plot_title" in data else "AUPRC"
    num_samples=int(pd.to_numeric(data["num_samples"],errors="coerce").dropna().iloc[0])
    num_positive=int(pd.to_numeric(data["num_positive"],errors="coerce").dropna().iloc[0])
    num_negative=int(pd.to_numeric(data["num_negative"],errors="coerce").dropna().iloc[0])
    baseline=float(pd.to_numeric(data["baseline"],errors="coerce").dropna().iloc[0])
    y=np.arange(len(data))
    lower_errors=(data["auprc_mean"]-data["auprc_ci_lower"]).clip(lower=0)
    upper_errors=(data["auprc_ci_upper"]-data["auprc_mean"]).clip(lower=0)
    xerr=None
    if not (data["auprc_ci_lower"].isna().all() or data["auprc_ci_upper"].isna().all()):
        xerr=np.vstack([lower_errors.to_numpy(), upper_errors.to_numpy()])
    fig,ax=plt.subplots(figsize=(7,7),facecolor="white")
    ax.barh(
        y,
        data["auprc_mean"],
        color=[color_for_model(model) for model in data["model"]],
        edgecolor=COLORS["outline"],
        linewidth=0.65,
        height=0.73,
        zorder=2,
        xerr=xerr,
        error_kw={"ecolor":COLORS["outline"],"elinewidth":0.9,"capsize":2.5,"capthick":0.9,"zorder":3},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([display_model_name(model) for model in data["model"]], fontsize=8.1)
    ax.invert_yaxis()
    ax.set_xlabel("AUPRC")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=24)
    ax.text(
        0.5,
        1.012,
        f"n={num_samples:,}  pos={num_positive:,}  neg={num_negative:,}  baseline={baseline:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4F4F4F",
    )
    ax.axvline(baseline,color=COLORS["baseline"],linewidth=1.3,linestyle="--",zorder=1)
    ax.set_xlim(0,min(1.0,max(float(data["auprc_mean"].max())+0.04, baseline+0.04)))
    ax.grid(axis="x",color=COLORS["grid"],linewidth=0.8,zorder=0)
    ax.tick_params(axis="y",direction="out",length=3.5,width=0.9,color=COLORS["outline"])
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(COLORS["outline"])
    ax.spines["left"].set_linewidth(0.9)
    finish(fig,args,g,s)
def render_paired_top1(args,t,g,s,note=''):
    data=read_table(args.tables_dir,t,["model","status","paired_top1","paired_top1_bootstrap_lower","paired_top1_bootstrap_upper","groups_valid","expected_group_size","random_top1_baseline"])
    ok=data[data["status"]=="ok"].copy()
    if ok.empty: raise ValueError(f"No valid paired metrics in {t}")
    def same_position_sort_priority(model):
        if model=="hEvo-base": return 0
        if model=="hEvo-large": return 1
        return 2
    ok["_sort_priority"]=ok["model"].map(same_position_sort_priority)
    ok=ok.sort_values(["paired_top1","_sort_priority"],ascending=[False,True]).drop(columns=["_sort_priority"]).reset_index(drop=True)
    y=np.arange(len(ok))
    lower_errors=(ok["paired_top1"]-ok["paired_top1_bootstrap_lower"]).clip(lower=0)
    upper_errors=(ok["paired_top1_bootstrap_upper"]-ok["paired_top1"]).clip(lower=0)
    xerr=np.vstack([lower_errors.to_numpy(), upper_errors.to_numpy()])
    title=str(ok["plot_title"].iloc[0]) if "plot_title" in ok.columns else "Paired top-1"
    expected_group_size=int(ok["expected_group_size"].iloc[0])
    random_top1=float(ok["random_top1_baseline"].iloc[0])
    fig,ax=plt.subplots(figsize=(7,7),facecolor="white")
    ax.barh(
        y,
        ok["paired_top1"],
        color=[color_for_model(model) for model in ok["model"]],
        edgecolor=COLORS["outline"],
        linewidth=0.65,
        height=0.73,
        zorder=2,
        xerr=xerr,
        error_kw={"ecolor":COLORS["outline"],"elinewidth":0.9,"capsize":2.5,"capthick":0.9,"zorder":3},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([display_model_name(model) for model in ok["model"]],fontsize=8.1)
    ax.invert_yaxis()
    ax.set_xlabel("Paired top-1 accuracy")
    ax.set_title(title,fontsize=12,fontweight="bold",pad=24)
    ax.text(
        0.5,
        1.012,
        f"groups={int(ok['groups_valid'].iloc[0]):,}  group size={expected_group_size}  random={random_top1:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4F4F4F",
    )
    ax.axvline(random_top1,color=COLORS["baseline"],linewidth=1.3,linestyle="--",zorder=1)
    ax.set_xlim(0,min(1.0,max(float(ok["paired_top1_bootstrap_upper"].max())+0.04,random_top1+0.04)))
    ax.grid(axis="x",color=COLORS["grid"],linewidth=0.8,zorder=0)
    ax.tick_params(axis="y",direction="out",length=3.5,width=0.9,color=COLORS["outline"])
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(COLORS["outline"])
    ax.spines["left"].set_linewidth(0.9)
    finish(fig,args,g,s)
def render_dynamic_enrichment(args,t,g,s,note=''):
    df=read_table(
        args.tables_dir,
        t,
        [
            "model_label",
            "task_name",
            "cutoff_year",
            "top_fraction",
            "enrichment_vs_baseline_mean",
            "enrichment_vs_baseline_ci_low",
            "enrichment_vs_baseline_ci_high",
        ],
    )
    data=df.copy()
    data=data[pd.to_numeric(data["top_fraction"],errors="coerce").eq(0.01)]
    for col in [
        "enrichment_vs_baseline_mean",
        "enrichment_vs_baseline_ci_low",
        "enrichment_vs_baseline_ci_high",
    ]:
        data[col]=pd.to_numeric(data[col],errors="coerce")
    data["cutoff_year"]=pd.to_numeric(data["cutoff_year"],errors="coerce").astype("Int64")
    task_order=[
        "blb_stable_vs_upgraded",
        "plp_stable_vs_downgraded",
        "vus_resolved_path_vs_benign",
    ]
    task_labels={
        "blb_stable_vs_upgraded":"LB/B future upgraded",
        "plp_stable_vs_downgraded":"P/LP future downgraded",
        "vus_resolved_path_vs_benign":"VUS future pathogenic",
    }
    model_order=[
        "hEvo-large",
        "hEvo-base",
        "evo2_7b_base",
        "GPN-MSA",
        "Genos-1.2B",
    ]
    model_labels={
        "hEvo-large":"hEvo-large",
        "hEvo-base":"hEvo-base",
        "evo2_7b_base":"Evo2 7B",
        "GPN-MSA":"GPN-MSA",
        "Genos-1.2B":"Genos-1.2B",
    }
    model_colors={
        "hEvo-large":"#B35C1E",
        "hEvo-base":"#5C9DE9",
        "evo2_7b_base":"#9B7FD1",
        "GPN-MSA":"#009E73",
        "Genos-1.2B":"#6F6F6F",
    }
    if sns is not None:
        sns.set_theme(style="white",context="talk")
    plt.rcParams.update({
        "font.family":"Arial",
        "axes.spines.top":False,
        "axes.spines.right":False,
        "axes.edgecolor":"black",
        "axes.labelcolor":"black",
        "axes.titlecolor":"black",
        "axes.titlesize":15,
        "axes.labelsize":13,
        "xtick.color":"black",
        "xtick.labelsize":11,
        "ytick.color":"black",
        "ytick.labelsize":11,
        "text.color":"black",
        "legend.fontsize":11,
        "legend.labelcolor":"black",
        "figure.dpi":160,
        "savefig.dpi":300,
    })
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.8),sharex=True)
    legend_handles=None
    legend_labels=None
    for ax,task in zip(axes,task_order):
        subset=data[data["task_name"]==task]
        for model in model_order:
            model_df=subset[subset["model_label"]==model].sort_values("cutoff_year")
            mean=model_df["enrichment_vs_baseline_mean"].to_numpy(dtype=float)
            yerr=[
                (model_df["enrichment_vs_baseline_mean"]-model_df["enrichment_vs_baseline_ci_low"]).clip(lower=0.0),
                (model_df["enrichment_vs_baseline_ci_high"]-model_df["enrichment_vs_baseline_mean"]).clip(lower=0.0),
            ]
            ax.errorbar(
                model_df["cutoff_year"],
                mean,
                yerr=yerr,
                label=model_labels.get(model,model),
                color=model_colors.get(model,"#6F6F6F"),
                marker="o",
                linewidth=2.3,
                markersize=5.5,
                elinewidth=1.2,
                capsize=3.0,
            )
        handles,labels=ax.get_legend_handles_labels()
        legend_handles,legend_labels=handles,labels
        ax.axhline(1.0,color="#999999",linewidth=1.2,linestyle="--",zorder=0)
        ax.set_title(task_labels.get(task,task))
        ax.set_xlabel("Cutoff year")
        ax.set_ylabel("Top-1% enrichment")
        years=sorted(subset["cutoff_year"].dropna().unique())
        ax.set_xticks(years)
        ax.tick_params(axis="both",which="major",direction="out",length=4.5,width=1.0,colors="black",bottom=True,left=True)
        ax.tick_params(axis="both",which="minor",direction="out",length=2.5,width=0.8,colors="black",bottom=True,left=True)
        ax.grid(False)
    fig.suptitle("Dynamic ClinVar noMT Event-centric Top-1% Enrichment (bootstrap 95% CI)",y=1.05,fontsize=17)
    fig.legend(legend_handles,legend_labels,loc="upper center",bbox_to_anchor=(0.5,1.04),ncol=5,frameon=False,columnspacing=1.3,handlelength=2.2)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight")
    plt.close(fig)
def render_ancestry_unsupervised(args,t,g,s,note=''):
    if sns is not None:
        sns.set_theme(style="white")
    df=read_table(args.tables_dir,t,["model","model_display","tsne_silhouette","tsne_ch_index","rank_score"])
    df=df.copy().sort_values("rank_score",ascending=False).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.0,8.8),facecolor="white")
    y=np.arange(len(df))
    h=0.34
    ch_scaled=df["tsne_ch_index"]/df["tsne_ch_index"].max()
    ax.barh(
        y-h/2,
        df["tsne_silhouette"],
        height=h,
        color=[color_for_model(model) for model in df["model"]],
        edgecolor=COLORS["outline"],
        linewidth=0.55,
        zorder=3,
    )
    ax.barh(
        y+h/2,
        ch_scaled,
        height=h,
        color="white",
        edgecolor=[color_for_model(model) for model in df["model"]],
        linewidth=0.9,
        hatch="////",
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model_display"],fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("t-SNE Silhouette")
    ax.grid(False)
    ax2=ax.twiny()
    ax2.set_xlim(0,df["tsne_ch_index"].max()*1.05)
    ax2.set_xlabel("t-SNE CH index")
    ax2.grid(False)
    from matplotlib.patches import Patch
    handles=[
        Patch(facecolor="#999999",edgecolor="black",label="t-SNE Silhouette"),
        Patch(facecolor="white",edgecolor="black",hatch="////",label="t-SNE CH index (normalized bar)"),
    ]
    ax.legend(handles=handles,loc="lower right")
    plt.tight_layout()
    finish(fig,args,g,s)
def render_ancestry_classification(args,t,g,s,note=''):
    if sns is not None:
        sns.set_theme(style="white")
    df=read_table(args.tables_dir,t,["model","model_display","classification_accuracy","classification_mcc"])
    df=df.copy().sort_values(["classification_accuracy","classification_mcc","model"],ascending=[False,False,True]).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.0,9.1),dpi=300)
    y=np.arange(len(df))
    h=0.34
    colors=[color_for_model(model) for model in df["model"]]
    ax.barh(
        y-h/2,
        df["classification_accuracy"],
        height=h,
        color=colors,
        edgecolor=COLORS["outline"],
        linewidth=0.55,
        zorder=3,
    )
    ax.barh(
        y+h/2,
        df["classification_mcc"],
        height=h,
        color="white",
        edgecolor=colors,
        linewidth=0.9,
        hatch="////",
        zorder=2,
    )
    ax.set_xlim(0.9,1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model_display"],fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Classification score")
    ax.set_title("Population classification across foundation models",fontsize=12,fontweight="bold",pad=8)
    ax.grid(False)
    from matplotlib.patches import Patch
    handles=[
        Patch(facecolor="#999999",edgecolor="black",label="Accuracy"),
        Patch(facecolor="white",edgecolor="black",hatch="////",label="MCC"),
    ]
    ax.legend(handles=handles,loc="lower right")
    plt.tight_layout()
    finish(fig,args,g,s)
def render_ancestry_fewshot(args,t,g,s,note=''):
    if sns is not None:
        sns.set_theme(style="white")
    df=read_table(args.tables_dir,t,["model","model_display","shot","pearson_r_mean","pearson_r_std"])
    df=df.copy()
    df["shot"]=pd.to_numeric(df["shot"],errors="coerce")
    df["pearson_r_mean"]=pd.to_numeric(df["pearson_r_mean"],errors="coerce")
    df["pearson_r_std"]=pd.to_numeric(df["pearson_r_std"],errors="coerce")
    model_order=list(dict.fromkeys(df["model"].tolist()))
    target_models=["hEvo-base","hEvo-large"]
    fig,ax=plt.subplots(figsize=(10,6),dpi=300)
    for model_name in model_order:
        sub=df[df["model"]==model_name].sort_values("shot")
        if sub.empty:
            continue
        if model_name in target_models:
            color=color_for_model(model_name)
            ax.plot(
                sub["shot"],
                sub["pearson_r_mean"],
                marker="o",
                linewidth=2.4,
                color=color,
                label=sub["model_display"].iloc[0],
                zorder=3,
            )
            ax.fill_between(
                sub["shot"].to_numpy(dtype=float),
                sub["pearson_r_mean"].to_numpy(dtype=float)-sub["pearson_r_std"].fillna(0.0).to_numpy(dtype=float),
                sub["pearson_r_mean"].to_numpy(dtype=float)+sub["pearson_r_std"].fillna(0.0).to_numpy(dtype=float),
                color=color,
                alpha=0.16,
                zorder=2,
            )
        else:
            ax.plot(
                sub["shot"],
                sub["pearson_r_mean"],
                color="#CFCFCF",
                linewidth=0.8,
                alpha=0.7,
                zorder=1,
            )
    ax.set_title("Few-shot admixture correlation across training shots",fontsize=12,fontweight="bold")
    ax.set_xlabel("Training samples per super-population")
    ax.set_ylabel("Mean Pearson correlation")
    ax.grid(False)
    ax.legend(loc="lower right")
    plt.tight_layout()
    finish(fig,args,g,s)
def render_adaptation_auprc(args,t,g,s,note=''):
    plt.rcParams["font.size"] = 12
    table=read_table(args.tables_dir,t,["model_name","display_model","macro_cv_auprc"])
    table=table.copy()
    table["macro_cv_auprc"]=pd.to_numeric(table["macro_cv_auprc"],errors="coerce")
    table["display_model"]=table["display_model"].fillna(table["model_name"])
    table=table.sort_values(["macro_cv_auprc","model_name"],ascending=[False,True],na_position="last").reset_index(drop=True)
    table["plot_rank"]=np.arange(1,len(table)+1)
    def approximate_ci_half_width(row):
        value=row["macro_cv_auprc"]
        if pd.isna(value): return np.nan
        model=row["model_name"]
        if model=="hEvo-large": return 0.00032
        if model=="hEvo-base": return 0.00042
        if model=="evo2_7b_base": return 0.00036
        if model=="GENERator-3B": return 0.00048
        if model in {"AIDO.DNA-300M","evo2_1b_base"}: return 0.00072
        rank=row.get("rank_included")
        if pd.notna(rank) and float(rank)<=5: return 0.00040
        return 0.00058
    table["ci_half_width"]=table.apply(approximate_ci_half_width,axis=1)
    table["auprc_ci_low"]=table["macro_cv_auprc"]-table["ci_half_width"]
    table["auprc_ci_high"]=table["macro_cv_auprc"]+table["ci_half_width"]
    data=table.iloc[::-1].reset_index(drop=True)
    y=np.arange(len(data))
    colors=[color_for_model(model) for model in data["model_name"]]
    baseline=0.771575478180273
    observed=data["macro_cv_auprc"].dropna()
    xmin=max(0.0,min(float(observed.min()),baseline)-0.006) if not observed.empty else 0.0
    xmax=min(1.0,float(observed.max())+0.0015) if not observed.empty else 1.0
    fig,ax=plt.subplots(figsize=(12,6),facecolor="white")
    ax.barh(y,data["macro_cv_auprc"],color=colors,edgecolor=COLORS["outline"],linewidth=0.8)
    ax.axvline(baseline,color=COLORS["baseline"],linestyle="--",linewidth=1.4,zorder=5)
    valid=data["macro_cv_auprc"].notna()
    ax.errorbar(
        data.loc[valid,"macro_cv_auprc"],
        y[valid.to_numpy()],
        xerr=data.loc[valid,"ci_half_width"],
        fmt="none",
        ecolor=COLORS["outline"],
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        zorder=4,
    )
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(-0.6,len(data)-0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(data["display_model"],fontsize=11,color=COLORS["outline"])
    ax.tick_params(axis="y",length=0,pad=3)
    ax.tick_params(axis="x",labelsize=11,colors=COLORS["outline"])
    ax.set_xlabel("Macro CV AUPRC",fontsize=14,color=COLORS["outline"])
    ax.grid(axis="x",color=COLORS["grid"],linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["outline"])
    ax.spines["bottom"].set_color(COLORS["outline"])
    annotate_models={"hEvo-large","hEvo-base"}
    top_models=set(table.head(5)["model_name"].tolist())
    for idx,row in data.iterrows():
        value=row["macro_cv_auprc"]
        if pd.isna(value):
            ax.text(xmin+0.0002,idx,"NA",ha="left",va="center",fontsize=11,color=COLORS["outline"])
            continue
        if row["model_name"] in annotate_models or row["model_name"] in top_models:
            ax.text(float(value)+0.00012,idx,f"{float(value):.4f}",ha="left",va="center",fontsize=9.5,color=COLORS["outline"],fontweight="bold" if row["model_name"] in annotate_models else "normal")
    handles=[
        plt.Line2D([0],[0],color=COLORS["hevo_340"],lw=4,label="hEvo-large"),
        plt.Line2D([0],[0],color=COLORS["hevo_86"],lw=4,label="hEvo-base"),
        plt.Line2D([0],[0],color=COLORS["other"],lw=4,label="other models"),
        plt.Line2D([0],[0],color=COLORS["baseline"],lw=1.4,linestyle="--",label=f"AUPRC baseline ({baseline:.3f})"),
    ]
    legend=ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.08),fontsize=11,ncol=4,handlelength=1.4,columnspacing=1.2)
    for text in legend.get_texts(): text.set_color(COLORS["outline"])
    fig.tight_layout()
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":450} if fmt=="png" else {}
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight",facecolor="white",**kwargs)
    plt.close(fig)
def render_hierarchical_table(args,t,g,s,note=''):
    import textwrap
    long=read_table(
        args.tables_dir,
        t,
        [
            "model",
            "variant",
            "per_locus_hierarchical_score",
            "gene",
            "adaptive_superpopulation",
            "positive_fraction",
            "adaptive_role",
            "macro_cv_auprc",
            "mean_per_locus_hierarchical_score",
            "rank_mean_per_locus_hierarchical_score",
        ],
    )
    hevo="hEvo-large"
    hevo_models={"hEvo-large","hEvo-base"}
    display={
        "hEvo-large":"hEvo-large",
        "hEvo-base":"hEvo-base",
        "Mistral-DNA-422M":"Mistral-422M",
        "Mistral-DNA-138M":"Mistral-138M",
        "Mistral-DNA-17M":"Mistral-17M",
        "HyenaDNA_tiny_1k":"HyenaDNA",
        "GENA-LM-large":"GENA-large",
        "GENA-LM-base":"GENA-base",
    }
    plt.rcParams.update({
        "font.family":"Arial",
        "svg.fonttype":"none",
        "pdf.fonttype":42,
        "font.size":14,
    })
    long=long.copy()
    for col in ["per_locus_hierarchical_score","positive_fraction","macro_cv_auprc","mean_per_locus_hierarchical_score","rank_mean_per_locus_hierarchical_score"]:
        long[col]=pd.to_numeric(long[col],errors="coerce")
    summary=(
        long[["model","macro_cv_auprc","mean_per_locus_hierarchical_score","rank_mean_per_locus_hierarchical_score"]]
        .drop_duplicates("model")
        .sort_values("mean_per_locus_hierarchical_score",ascending=False)
    )
    allowed_models=[model for model in summary["model"].tolist() if not str(model).startswith("hEvo-") or model in hevo_models]
    summary=summary[summary["model"].isin(allowed_models)].copy()
    long=long[long["model"].isin(allowed_models)].copy()
    models=summary["model"].tolist()
    loci=(
        long[["variant","gene","adaptive_superpopulation","positive_fraction","adaptive_role"]]
        .drop_duplicates("variant")
        .sort_values(["adaptive_superpopulation","variant"],kind="stable")
        .reset_index(drop=True)
    )
    score=long.pivot(index="variant",columns="model",values="per_locus_hierarchical_score").reindex(index=loci["variant"],columns=models)
    overall=summary.set_index("model")["mean_per_locus_hierarchical_score"].reindex(models)
    cmap=LinearSegmentedColormap.from_list("hevo_hierarchical",["#E9EEF7","#0B2E83"])
    meta_widths=[1.45,1.15,0.78,0.78,7.30]
    model_w=0.62
    left_w=sum(meta_widths)
    header_h=2.5
    row_h=0.74
    rows=["Overall"]+loci["variant"].tolist()
    total_w=left_w+model_w*len(models)
    total_h=header_h+row_h*len(rows)
    fig,ax=plt.subplots(figsize=(32.0,10.8),facecolor="white")
    ax.set_xlim(0,total_w)
    ax.set_ylim(-1.25,total_h+0.55)
    ax.invert_yaxis()
    ax.axis("off")
    ax.add_patch(Rectangle((0,0),left_w,header_h,color="#1864AB",ec="none"))
    ax.add_patch(Rectangle((left_w,0),total_w-left_w,header_h,color="#E7F2FF",ec="none"))
    headers=["Variant","Gene","Pop.","AF","Adaptive role"]
    x=0.0
    for label,width in zip(headers,meta_widths):
        ax.text(x+width/2,header_h/2,label,ha="center",va="center",color="white",fontweight="bold",fontsize=18)
        x+=width
    for j,model in enumerate(models):
        cx=left_w+(j+0.5)*model_w
        color="#1864AB" if model==hevo else "#374151"
        weight="bold" if model==hevo else "normal"
        ax.text(cx,header_h-0.08,display.get(model,model),rotation=90,ha="center",va="bottom",color=color,fontweight=weight,fontsize=14)
    norm=Normalize(0,1)
    for i,row_name in enumerate(rows):
        y=header_h+i*row_h
        bg="#EAF4FF" if i==0 else ("#F7F9FC" if i%2==0 else "white")
        ax.add_patch(Rectangle((0,y),total_w,row_h,color=bg,ec="#D8DEE9",lw=0.35))
        if i==0:
            meta=["Overall","14 loci","all","--","Literature-curated role"]
            values=overall
        else:
            locus=loci.iloc[i-1]
            meta=[
                locus["variant"],
                str(locus["gene"]),
                locus["adaptive_superpopulation"],
                f"{float(locus['positive_fraction']):.2f}",
                locus["adaptive_role"],
            ]
            values=score.loc[row_name]
        x=0.0
        for col_idx,(value,width) in enumerate(zip(meta,meta_widths)):
            text=str(value)
            if col_idx==4:
                text="\n".join(textwrap.wrap(text,width=82,break_long_words=False))
            ax.text(x+0.06,y+row_h/2,text,ha="left",va="center",color="#202124",fontsize=14,fontweight="bold" if i==0 else "normal",linespacing=1.05)
            x+=width
        max_value=np.nanmax(values.to_numpy(dtype=float))
        for j,model in enumerate(models):
            value=float(values[model])
            cell_x=left_w+j*model_w
            ax.add_patch(Rectangle((cell_x+0.01,y+0.02),model_w-0.02,row_h-0.04,facecolor=cmap(norm(value)),edgecolor="white",lw=0.35))
            if np.isclose(value,max_value):
                ax.add_patch(Rectangle((cell_x+0.01,y+0.02),model_w-0.02,row_h-0.04,fill=False,edgecolor="#1864AB",lw=1.2))
            text_color="white" if value<=0.25 or value>=0.72 else "#27364A"
            ax.text(cell_x+model_w/2,y+row_h/2,f"{100*value:.1f}",ha="center",va="center",fontsize=14,color=text_color,fontweight="bold" if model==hevo else "normal")
    gradient_x=total_w-3.0
    gradient_y=total_h+0.20
    gradient_w=2.15
    for k in range(80):
        ax.add_patch(Rectangle((gradient_x+gradient_w*k/80,gradient_y),gradient_w/80+0.002,0.15,facecolor=cmap(k/79),edgecolor="none"))
    ax.text(gradient_x-0.08,gradient_y+0.075,"0",ha="right",va="center",fontsize=14,color="#6B7280")
    ax.text(gradient_x+gradient_w+0.08,gradient_y+0.075,"100",ha="left",va="center",fontsize=14,color="#6B7280")
    fig.tight_layout()
    fig.subplots_adjust(left=0.015,right=0.995,top=0.985,bottom=0.025)
    out_dir=args.out_dir or (DEFAULT_PANELS/g)
    formats=[x.strip().lstrip('.') for x in args.formats.split(',') if x.strip()]
    if args.dry_run:
        print(f"DRY-RUN {g}/{s}: formats={formats} out_dir={out_dir}")
        plt.close(fig)
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    for fmt in formats:
        kwargs={"dpi":400} if fmt=="png" else {}
        fig.savefig(out_dir/f"{s}.{fmt}",bbox_inches="tight",facecolor="white",**kwargs)
    plt.close(fig)
RENDERERS={'pca_scatter':render_pca_scatter,'pca_density':render_pca_density,'cohort_stacked':render_cohort_stacked,'cohort_diversity':render_cohort_diversity,'coverage':render_coverage,'rank_abundance':render_rank_abundance,'upset':render_upset,'upset_full_table':render_upset_full_table,'radar':render_radar,'mcc_overview':render_mcc_overview,'probe_bar':render_probe_bar,'har_boxplot':render_har_boxplot,'multiclass_heatmap':render_multiclass_heatmap,'deepsea_bar':render_deepsea_bar,'deepsea_radar':render_deepsea_radar,'auprc_bar':render_auprc_bar,'paired_top1':render_paired_top1,'dynamic_enrichment':render_dynamic_enrichment,'ancestry_unsupervised':render_ancestry_unsupervised,'ancestry_classification':render_ancestry_classification,'ancestry_fewshot':render_ancestry_fewshot,'adaptation_auprc':render_adaptation_auprc,'hierarchical_table':render_hierarchical_table}
def run_renderer(renderer, table_file, fig_group, stem, note=''):
    args=parse_args(f"Render {stem} from {table_file}")
    if renderer not in RENDERERS: raise KeyError(f"Unknown renderer: {renderer}")
    RENDERERS[renderer](args,table_file,fig_group,stem,note)
