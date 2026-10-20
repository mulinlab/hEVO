"""Reference-layout Fig.5, entirely rendered as editable Matplotlib vectors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import NullLocator, FuncFormatter
from adjustText import adjust_text
import _core as core

W,H=210.,297.
ORANGE="#B35C1E"
GRAY="#65717F"
LIGHT="#B8BEC9"
INK="#000000"
CORPORA=[("PopReduce_60K","hEVO selected (60K)"),("1000G_3K","1000G (3K)"),("UKB_500K","UKB (500K)")]
D_STYLES={
    "hEvo-base":(ORANGE,"o","-"),
    "hEvo-large":(ORANGE,"*","-"),
    "hEvo-open-base":("#237B91","o","--"),
    "hEvo-open-large":("#237B91","*","--"),
}

def text(fig,x,top,s,fs=6,**kw):
    return fig.text(x/W,1-top/H,s,fontsize=fs,color=INK,ha=kw.pop("ha","left"),
                    va=kw.pop("va","center"),**kw)

def axes(fig,x,top,w,h,**kw):
    return fig.add_axes([x/W,1-(top+h)/H,w/W,h/H],**kw)

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(.45)
    ax.tick_params(labelsize=6,width=.45,length=2,pad=2)

def flowchart(fig):
    ax=axes(fig,7,13,44,64)
    ax.set_xlim(0,44); ax.set_ylim(64,0); ax.axis("off")
    def t(x,y,s,weight="normal"):
        ax.text(x,y,s,fontsize=6,ha="center",va="center",fontweight=weight)
    def box(x,y,w,h,s,emph=False):
        ax.add_patch(Rectangle((x,y),w,h,facecolor="#FBF1E7" if emph else "#F1F3F5",
                               edgecolor=ORANGE if emph else GRAY,lw=.45))
        if s:t(x+w/2,y+h/2,s,"bold" if emph else "normal")
    def arrow(x,y,xx,yy):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle="-|>",mutation_scale=4,
                                    linewidth=.45,color=INK))
    t(22,1.5,"Training corpus")
    box(0,4,44,4.8,"hEVO selected (60K)",True)
    box(0,10,44,4.8,"1000G (3K)")
    box(0,16,44,4.8,"UKB (500K)")
    arrow(22,22,22,25.5)
    t(10,27.3,"No sliding"); t(35,27.3,"Sliding")
    for i in range(3): box(i*6.5,30,6.2,1.6,"")
    for i in range(3): box(25+i*2,29.3+i*1.5,13,1.6,"",True)
    t(10,35,"1,024 bp"); t(34,36.4,"128 bp stride")
    ax.plot([0,44],[40,40],color=INK,lw=.35)
    arrow(22,40,22,43.5)
    box(0,45,19,10,""); box(25.8,45,18.2,10,"",True)
    t(9.5,48,"Stage I","bold"); t(9.5,52,"MLM")
    t(34.9,48,"Stage II","bold"); t(34.9,52,"Refinement")
    arrow(19.5,50,25.5,50)
    t(22,59.3,"3 corpora × 2 window schemes")
    t(22,62.8,"Two stages · matched 87.095M ablations")
    text(fig,29,9.2,"Ablation design",7,ha="center",fontweight="bold")
    return ax

def nt(fig,root):
    data=core.read_s41(root)
    means=data.groupby(["training_dataset","windowing"]).metric_mean.mean()
    ax=axes(fig,99,18,53,14)
    for i,(corpus,label) in enumerate(CORPORA):
        color=ORANGE if i==0 else GRAY
        xx=[means.loc[(corpus,w)] for w in ["NoSW","SW"]]
        ax.plot(xx,[i,i],color=color,lw=.5)
        for x,fill in zip(xx,["white",color]):
            ax.scatter(x,i,s=14,facecolor=fill,edgecolor=color,lw=.6,zorder=3)
            ax.annotate(f"{x:.3f}",(x,i),xytext=(0,4),textcoords="offset points",
                        fontsize=6,ha="center",va="bottom")
        text(fig,61,20+i*4.9,label,6,fontweight="bold" if i==0 else "normal")
    ax.set_ylim(2.45,-.4); ax.set_yticks([])
    ax.set_xlim(.52,.605); ax.set_xticks([.52,.54,.56,.58,.60])
    clean(ax); ax.set_xlabel("Mean MCC (18 tasks)",fontsize=7,labelpad=3)
    text(fig,111,9.2,"NT benchmark",7,ha="center",fontweight="bold")
    fig.legend(handles=[
        Line2D([],[],marker="o",markersize=3.4,markerfacecolor="white",markeredgecolor=ORANGE,color="none",label="No sliding"),
        Line2D([],[],marker="o",markersize=3.4,markerfacecolor=ORANGE,markeredgecolor=ORANGE,color="none",label="Sliding")],
        loc="center",bbox_to_anchor=(122/W,1-14.2/H),ncol=2,frameon=False,fontsize=6,columnspacing=4,handletextpad=.4)
    return ax

def clinvar(fig,root):
    d=core.read_s42(root)
    d=d[d.dataset.eq("all") & d.evidence_status.eq("PASS")]
    text(fig,111,45,"ClinVar",7,ha="center",fontweight="bold")
    result=[]
    for metric,x in [("AUPRC",105),("AUROC",132)]:
        ax=axes(fig,x,51,20,21)
        for i,(corpus,label) in enumerate(CORPORA):
            if metric=="AUPRC":
                text(fig,61,53.5+i*7.2,label,6,fontweight="bold" if i==0 else "normal")
            for j,stage in enumerate(["Stage I","Stage II"]):
                y=i*2.7+j*1.1
                sub=d[d.training_dataset.eq(corpus)&d.stage.eq(stage)&d.metric.eq(metric)].set_index("windowing")
                color=ORANGE if i==0 else GRAY
                xx=sub.loc[["NoSW","SW"],"metric_mean"].to_numpy(float)
                ax.plot(xx,[y,y],color=color,lw=.55)
                for k,w in enumerate(["NoSW","SW"]):
                    ax.scatter(xx[k],y,s=11,facecolor="white" if k==0 else color,
                               edgecolor=color,lw=.6,zorder=3)
                if metric=="AUPRC":
                    ax.text(-.20,y,stage,transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=6)
        ax.set_ylim(7.5,-.5); ax.set_yticks([])
        ax.set_xlim((.3,1) if metric=="AUPRC" else (.5,1))
        ax.set_xticks([.3,.6,1] if metric=="AUPRC" else [.5,.75,1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
        clean(ax); ax.set_xlabel(metric,fontsize=7,labelpad=3)
        result.append(ax)
    return result

def rank_profiles(fig,root):
    profiles,details=core._d_rank_profiles(root)
    groups=[
        ("Genomic elements",12,25,[("Genomic","Genomic"),("NT","NT"),("URE","URE"),("DeepSEA","DeepSEA")]),
        ("Variant scoring",36,49,[("ClinVar","ClinVar"),("De novo","De novo"),("GWAS causal","GWAS causal"),("Rare variants","Rare variants"),("MPRA","MPRA")]),
        ("Evolution & ancestry",61,74,[("HAR","HAR"),("HAQER","HAQER"),("Ancestry","Ancestry"),("Adaptive probe","Adaptation")]),
    ]
    drawn=set()
    incomplete={}
    for title,title_y,cy,keys in groups:
        text(fig,181,title_y,title,6.5,ha="center",fontweight="bold")
        ax=axes(fig,169.5,cy-11.5,23,23)
        ax.set_xlim(-1.52,1.52); ax.set_ylim(-1.52,1.52); ax.set_aspect("equal"); ax.axis("off")
        angles=np.pi/2-np.arange(len(keys))*2*np.pi/len(keys)
        unit=np.column_stack([np.cos(angles),np.sin(angles)])
        closed=np.vstack([unit,unit[0]])
        ax.plot(closed[:,0],closed[:,1],color="#97A5B7",lw=.45)
        for x,y in unit:ax.plot([0,x],[0,y],color="#AFBBCB",lw=.35)
        for model,(color,marker,ls) in D_STYLES.items():
            frames=[profiles[k].set_index("model") for k,_ in keys]
            present=np.array([model in f.index for f in frames],dtype=bool)
            if not present.any():continue
            r=np.array([f.loc[model,"radius"] if ok else np.nan
                        for f,ok in zip(frames,present)],dtype=float)
            xy=unit*r[:,None]
            # Never bridge an unmeasured axis. Connect only neighbouring
            # measured vertices, including the closing edge when both exist.
            for i in range(len(keys)):
                j=(i+1)%len(keys)
                if present[i] and present[j]:
                    ax.plot(xy[[i,j],0],xy[[i,j],1],color=color,lw=.65,
                            linestyle=ls,zorder=3)
            ax.scatter(xy[present,0],xy[present,1],color=color,marker=marker,
                       s=20 if marker=="*" else 7,zorder=4)
            if not present.all():
                incomplete.setdefault(model,{})[title]=[keys[i][1] for i in range(len(keys)) if not present[i]]
            drawn.add(model)
        for (x,y),(_,label) in zip(unit,keys):
            ax.text(x*1.12,y*1.13,label,fontsize=6,
                    ha="left" if x>.2 else "right" if x<-.2 else "center",
                    va="bottom" if y>.8 else "top" if y<-.7 else "center")
    # Fixed 2x2 legend positions cover all four hEvo variants.
    slots=[(160.5,88.5),(183.5,88.5),(160.5,93),(183.5,93)]
    for (model,(color,marker,ls)),(x,y) in zip(D_STYLES.items(),slots):
        if model not in drawn:continue
        fig.add_artist(Line2D([(x-1.4)/W,(x+1.4)/W],[1-y/H]*2,
                             transform=fig.transFigure,color=color,lw=.65,linestyle=ls,
                             marker=marker,markevery=[0],markersize=4 if marker=="*" else 2.7))
        text(fig,x+2.1,y,core.canonical_model_name(model),5.3)
    details["drawn_models"]=sorted(drawn)
    details["incomplete_model_groups"]=incomplete
    details["legend_capacity"]=4
    details["reserved_model"]="hEvo-open-large" if "hEvo-open-large" not in drawn else None
    return details

def nondominated(data,metric,higher):
    # Scan from best cost/throughput to worst; lower mean rank is better.
    d=data.sort_values([metric,"mean_rank"],ascending=[not higher,True])
    best=np.inf; keep=[]
    for row in d.itertuples():
        if row.mean_rank<best:
            keep.append(row.Index); best=row.mean_rank
    return data.loc[keep].sort_values(metric)

def pareto(fig,ax,summary,metric,higher,ids,label,xlabel):
    multiplier=.001 if metric=="throughput_bps" else 1
    plot=summary[np.isfinite(summary[metric])&np.isfinite(summary.mean_rank)&summary[metric].gt(0)].copy()
    complete=plot[(plot["n_"+metric]==plot.n_tasks)&(plot.n_perf==plot.n_tasks)&(plot.n_rank_tasks==45)]
    frontier=nondominated(complete,metric,higher)
    ax.plot(frontier[metric]*multiplier,frontier.mean_rank,color="#8F9AAE",lw=.55,
            linestyle=(0,(3,3)),drawstyle="steps-pre" if higher else "steps-post",zorder=1)
    for row in plot.itertuples():
        group=core._pareto_family_group(row.family)
        marker,color=core._pareto_group_style(group)
        if group=="Other":color=LIGHT
        if row.model=="hEvo-large":marker="*"
        incomplete=(getattr(row,"n_"+metric)<row.n_tasks or row.n_perf<row.n_tasks or row.n_rank_tasks<45)
        fill=color if row.model=="hEvo-base" or not incomplete else "white"
        size=46 if row.model=="hEvo-large" else 22 if group=="hEvo" else 13
        ax.scatter(getattr(row,metric)*multiplier,row.mean_rank,marker=marker,s=size,
                   facecolor=fill,edgecolor=color,lw=.6 if incomplete else .35,zorder=4)
    ax.set_xscale("log")
    lo,hi=plot[metric].min()*multiplier,plot[metric].max()*multiplier
    ax.set_xlim(lo*.57,hi*1.7)
    ax.set_ylim(35,.5);ax.set_yticks([5,10,15,20,25,30,35])
    clean(ax);ax.spines["left"].set_visible(True);ax.spines["left"].set_linewidth(.45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}" if x>=1 else f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    if metric=="params_m":ax.set_xticks([1,10,100,1000,10000])
    elif metric=="latency_ms":ax.set_xticks([1,10,100])
    elif metric=="throughput_bps":ax.set_xticks([10,100,1000])
    elif metric=="vram_mb":ax.set_xticks([200,1000,10000])
    ax.set_xlabel(xlabel,fontsize=7,labelpad=4)
    ax.set_ylabel("Mean rank across available tasks (45 total; lower is better)",fontsize=7,labelpad=3)
    labels=[]; xx=[]; yy=[]
    for row in plot.itertuples():
        x=getattr(row,metric)*multiplier;y=row.mean_rank
        special=row.model in ["hEvo-base","hEvo-large","evo2_7b_base"]
        name=core.canonical_model_name(row.model)
        s=f"{name} ({row.n_rank_tasks}/45)" if special else str(ids[row.model])
        lab=ax.text(x*1.04,y+(.3 if row.model!="hEvo-large" else -.3),s,
                    fontsize=6,fontweight="bold" if row.model.startswith("hEvo") else "normal",
                    ha="left",va="center",zorder=5)
        labels.append(lab);xx.append(x);yy.append(y)
    # Deterministic label packing is done in page pixel coordinates after layout.
    fig.canvas.draw()
    np.random.seed(42)
    adjust_text(labels,x=xx,y=yy,ax=ax,ensure_inside_axes=True,expand=(1.08,1.2),
                force_text=(.25,.4),force_static=(.15,.3),force_pull=(.01,.01),
                iter_lim=350,arrowprops=dict(arrowstyle="-",color="#718096",lw=.3))
    return {"metric":metric,"plotted_models":len(plot),"complete_frontier_models":frontier.model.tolist(),
            "rank_coverage":plot[["model","n_rank_tasks"]].to_dict("records"),
            "incomplete_models":plot.loc[(plot["n_"+metric]<plot.n_tasks)|(plot.n_perf<plot.n_tasks)|(plot.n_rank_tasks<45),["model","n_"+metric,"n_rank_tasks"]].to_dict("records")}

def make_a4(tables_dir=None):
    core.configure_matplotlib()
    root=core.tables_root(tables_dir)
    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor="white")
    panel_artists = {}
    for label, draw in [("a", lambda: flowchart(fig)),
                        ("b", lambda: nt(fig, root)),
                        ("c", lambda: clinvar(fig, root)),
                        ("d", lambda: rank_profiles(fig, root))]:
        before = set(fig.get_children())
        result = draw()
        panel_artists[label] = set(fig.get_children()) - before
        if label == "d": fig._d_details = result
    summary=pd.read_csv(root/core.S43_NAME,float_precision="round_trip")
    ranks=pd.read_csv(root/"S49_model_task_level_scores_and_ranks_for_fig5.csv",float_precision="round_trip")
    if ranks.duplicated(["model","task_unit"]).any():
        raise ValueError("S49 contains duplicate model/task rank cells")
    if ranks.task_unit.nunique()!=45:
        raise ValueError("S49 must contain the 45 normalized cross-task rank units")
    rank_summary=(ranks.groupby("model",sort=False).task_rank.agg(mean_rank="mean",n_rank_tasks="count").reset_index())
    if {"mean_rank","n_rank_tasks"}.issubset(summary.columns):
        expected=summary[["model"]].merge(rank_summary,on="model",how="left",validate="one_to_one")
        if expected[["mean_rank","n_rank_tasks"]].isna().any().any():
            raise ValueError("Every plotted model must have at least one S49 rank record")
        if not np.allclose(summary["mean_rank"],expected["mean_rank"],rtol=1e-10,atol=1e-12) or not np.array_equal(summary["n_rank_tasks"].to_numpy(int),expected["n_rank_tasks"].to_numpy(int)):
            raise ValueError("S43 mean ranks or rank coverage differ from S49")
    else:
        summary=summary.merge(rank_summary,on="model",how="left",validate="one_to_one")
    if summary[["mean_rank","n_rank_tasks"]].isna().any().any():
        raise ValueError("Every plotted model must have at least one S49 rank record")
    summary["n_rank_tasks"]=summary["n_rank_tasks"].astype(int)
    core._pareto_supplementary_ids(summary, root)
    if not summary.n_tasks.eq(18).all():raise ValueError("Unexpected S43 task coverage")
    others=sorted(set(summary.model)-{"hEvo-base","hEvo-large"})
    ids={model:i+1 for i,model in enumerate(others)}
    fig.legend(handles=core._pareto_family_handles_reference(),loc="center",
               bbox_to_anchor=(75/W,1-89/H),ncol=5,frameon=False,fontsize=6,
               columnspacing=3.3,handletextpad=.35)
    plans=[
        ("e",17,102,"params_m",False,"Predictor parameters (M)"),
        ("f",120,102,"latency_ms",False,"Latency (ms)"),
        ("g",17,182,"throughput_bps",True,"Throughput (kb/s)"),
        ("h",120,182,"vram_mb",False,"Incremental peak allocation (MiB)"),
    ]
    detailss={}
    for letter,x,top,metric,higher,xlabel in plans:
        before = set(fig.get_children())
        ax=axes(fig,x,top,84,61)
        detailss[letter]=pareto(fig,ax,summary,metric,higher,ids,letter,xlabel)
        text(fig,x+84,top-5.5,"X: higher is better" if higher else "X: lower is better",6,ha="right")
        panel_artists[letter] = set(fig.get_children()) - before
    # Model-number legend follows the reference's alphabetical display roster.
    for i,model in enumerate(others):
        col,row=divmod(i,6); x=8+col*40; y=260+row*3.5
        s=summary.set_index("model").loc[model]
        marker,color=core._pareto_group_style(core._pareto_family_group(s.family))
        if color=="#B8C2CE":color=LIGHT
        fig.add_artist(Line2D([x/W],[1-y/H],transform=fig.transFigure,
                             marker=marker,color="none",markerfacecolor=color,
                             markeredgecolor=color,markersize=2.7))
        text(fig,x+2, y,f"{ids[model]}  {core.canonical_model_name(model)} ({s.n_rank_tasks}/45)",6)
    text(fig,8,284,"X: arithmetic means across 18 NT tasks. Dashed lines: complete-coverage Pareto frontiers. Log-scaled x axes.",6)
    text(fig,8,288,"Model key/direct labels give rank coverage n/45. Open marks indicate incomplete rank or NT resource coverage.",6)
    text(fig,8,292,"Resources describe recorded settings; matched hardware, precision and batch size are not certified for historical runs.",6)
    text(fig,204,284,"d: outer = better rank",6,ha="right")
    for label,x,y in [("a",5,8.5),("b",58,8.5),("c",58,44.5),("d",157,8.5),
                      ("e",5,95),("f",108,95),("g",5,175),("h",108,175)]:
        panel_artists[label].add(text(fig,x,y,label,14,fontweight="bold"))
    fig._panel_artists = panel_artists
    fig._pareto_details=detailss
    fig._layout_details={"pareto_height_mm":61,"reference_pareto_height_mm":53,
                       "bottom_note_y_mm":288,"panel_a":"PROGRAMMATIC_VECTOR",
                       "deprecated_asset":"Fig5-a.pdf","d_legend_capacity":4}
    return fig
