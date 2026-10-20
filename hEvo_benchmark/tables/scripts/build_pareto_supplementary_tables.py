#!/usr/bin/env python3
"""Import current Pareto task JSONs as S44 and regenerate the S43 summary.

When historical per-task JSONs have been archived or removed, retain their
published S44 rows (including source hashes) and refresh rows for JSON files
that are still present. This allows new model measurements to be added without
dropping the existing 18-task resource baselines.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

SUMMARY="S43_model_level_prediction_performance_and_computational_resource_use.csv"
DETAIL="S44_task_level_prediction_performance_and_computational_resource_use.csv"
ALIASES={"hEvo-base":"hEVO-Base","hEvo-large":"hEVO-Large"}
BACKBONE_PARAMETER_COUNTS_M={"hEvo-base":87.094922,"hEvo-large":319.462410}
EFFICIENCY={
 "params_m":"total_params_M",
 "backbone_params_m":"backbone_params_M",
 "probe_params_m":"probe_params_M",
 "throughput_bps":"backbone_throughput_bps",
 "throughput_fps":"backbone_throughput_fps",
 "latency_ms":"backbone_latency_ms",
 "vram_mb":"backbone_peak_vram_mb",
 "flops_g":"backbone_flops_G",
 "sequence_length":"sequence_length",
 "embedding_dim":"embedding_dim",
 "projection_dim":"projection_dim",
}
AGGREGATES=["params_m","throughput_bps","latency_ms","vram_mb"]
GPN_STAR_MODELS={"GPN-Star-M","GPN-Star-P","GPN-Star-V"}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def finite(v):
    try:
        v=float(v)
    except (TypeError,ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan

def assert_current(path, expected):
    if not path.is_file():
        raise ValueError(f"Missing published table: {path}")
    actual=pd.read_csv(path,float_precision="round_trip")
    canonical=io.StringIO()
    expected.to_csv(canonical,index=False,na_rep="NA")
    canonical.seek(0)
    expected=pd.read_csv(canonical,float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,
                                      rtol=1e-10,atol=1e-12)
    except AssertionError as exc:
        raise ValueError(f"Published table differs from current source JSONs: {path.name}\n{exc}") from exc

def build(source,tables,check_only=False):
    old=pd.read_csv(tables/SUMMARY,float_precision="round_trip")
    previous_detail=pd.read_csv(tables/DETAIL,float_precision="round_trip")
    roster=pd.read_csv(tables/"S47_architecture_training_scope_and_provenance_of_evaluated_models.csv")
    old_models=set(old.model)
    roster_models=set(roster.model)
    eligible=old_models|GPN_STAR_MODELS
    if not eligible.issubset(roster_models):
        raise ValueError(f"S43 plot models missing from the S47 roster: {sorted(eligible-roster_models)}")
    if old.model.duplicated().any() or not old_models.issubset(roster_models):
        raise ValueError("S43 contains duplicate models or models missing from the S47 roster")
    old_families=old.set_index("model")["family"].to_dict()
    previous_rows=previous_detail.set_index(["model","task"],drop=False)
    tasks=sorted(p.name for p in source.iterdir() if p.is_dir())
    if len(tasks)!=18:
        raise ValueError(f"Expected 18 task directories, got {len(tasks)}")
    rows=[]
    retained_published_rows=set()
    for model in sorted(eligible):
        family=old_families.get(model, "GPN" if model.startswith("GPN-") else "")
        if not family:
            raise ValueError(f"No family label available for model {model}")
        raw=ALIASES.get(model,model)
        for task in tasks:
            rel=Path(task)/raw/f"results_{raw}_{task}.json"
            p=source/rel
            prior=None
            if (model,task) in previous_rows.index:
                prior=previous_rows.loc[(model,task)]
                if isinstance(prior,pd.DataFrame):
                    raise ValueError(f"Duplicate published S44 row: {model}/{task}")
            if not p.is_file() and (model,task) in previous_rows.index:
                if prior.result_status=="PRESENT":
                    rows.append(prior.to_dict())
                    retained_published_rows.add((model,task))
                    continue
            row={"table_id":"S44","model":model,"family":family,"task":task,
                 "raw_model_id":raw,"protocol":"no_projection","source_collection":"Probe_NT_Pareto",
                 "source_file":rel.as_posix(),"source_sha256":"","result_status":"MISSING_FILE"}
            for metric in ["mcc","accuracy","auc"]:
                for suffix in ["","_ci_low","_ci_high"]:row[metric+suffix]=np.nan
            row.update({k:np.nan for k in EFFICIENCY})
            if p.is_file():
                payload=json.loads(p.read_text())
                records=[(key,content[task]["no_projection"]) for key,content in payload.items()
                         if task in content and "no_projection" in content[task]]
                if len(records)!=1:
                    raise ValueError(f"Expected one no_projection record: {rel}")
                key,rec=records[0]
                row.update({"source_sha256":digest(p),"payload_model_key":key,"result_status":"PRESENT"})
                # Some result writers store the metric arrays directly on
                # ``no_projection`` instead of nesting them under performance.
                perf=rec.get("performance",rec)
                for metric in ["mcc","accuracy","auc"]:
                    values=perf.get(metric,[])
                    if not isinstance(values,list):values=[values]
                    for i,suffix in enumerate(["","_ci_low","_ci_high"]):
                        row[metric+suffix]=finite(values[i]) if i<len(values) else np.nan
                eff=rec.get("efficiency",{})
                row.update({k:finite(eff.get(rawkey)) for k,rawkey in EFFICIENCY.items()})
                # Some retained task JSONs contain performance only. In that
                # case, keep the already-published task-level efficiency values.
                if prior is not None and prior.result_status=="PRESENT":
                    reused=False
                    for column in EFFICIENCY:
                        if not np.isfinite(row[column]) and np.isfinite(finite(prior[column])):
                            row[column]=prior[column]
                            reused=True
                    for column in ["mcc","mcc_ci_low","mcc_ci_high","accuracy","accuracy_ci_low",
                                   "accuracy_ci_high","auc","auc_ci_low","auc_ci_high"]:
                        if not np.isfinite(row[column]) and np.isfinite(finite(prior[column])):
                            row[column]=prior[column]
                    if reused:
                        retained_published_rows.add((model,task))
                if model in BACKBONE_PARAMETER_COUNTS_M:
                    backbone_params=BACKBONE_PARAMETER_COUNTS_M[model]
                    row["backbone_params_m"]=backbone_params
                    probe_params=row["probe_params_m"]
                    row["params_m"]=backbone_params+(probe_params if np.isfinite(probe_params) else 0.0)
            row["missing_efficiency_fields"]=";".join(k for k in AGGREGATES if not np.isfinite(row[k])) or "none"
            rows.append(row)
    detail=pd.DataFrame(rows)
    if len(detail)!=len(eligible)*len(tasks) or detail.duplicated(["model","task"]).any():
        raise ValueError("Unexpected task-result grid")
    summary=[]
    for model,g in detail.groupby("model",sort=False):
        valid=g.mcc.notna()
        family=old_families.get(model, "GPN" if model.startswith("GPN-") else "")
        row={"table_id":"S43","model":model,"family":g.family.iloc[0],
             "mean_mcc":g.loc[valid,"mcc"].mean(),"n_perf":int(valid.sum()),"n_tasks":18,
             "missing_tasks":";".join(sorted(g.loc[~valid,"task"])) or "none"}
        for k in AGGREGATES:
            row[k]=g[k].mean()
            row["n_"+k]=int(g[k].notna().sum())
        row["n_eff"]=min(row["n_"+k] for k in AGGREGATES)
        summary.append(row)
    summary_frame=pd.DataFrame(summary)
    ranks=pd.read_csv(tables/"S49_model_task_level_scores_and_ranks_for_fig5.csv",float_precision="round_trip")
    if ranks.duplicated(["model","task_unit"]).any() or ranks.task_unit.nunique()!=45:
        raise ValueError("S49 must provide unique model/task rows across 45 task units")
    rank_summary=(ranks.groupby("model",sort=False).task_rank.agg(mean_rank="mean",n_rank_tasks="count").reset_index())
    summary_frame=summary_frame.merge(rank_summary,on="model",how="left",validate="one_to_one")
    if summary_frame[["mean_rank","n_rank_tasks"]].isna().any().any():
        raise ValueError("Every Pareto model must have at least one S49 cross-task rank")
    summary_frame["n_rank_tasks"]=summary_frame["n_rank_tasks"].astype(int)
    summary_columns=[column for column in old.columns if column not in {"mean_rank","n_rank_tasks"}]
    summary_frame=summary_frame[summary_columns+["mean_rank","n_rank_tasks"]].sort_values(["mean_mcc","model"],ascending=[False,True]).reset_index(drop=True)
    coverage_columns=["n_perf",*["n_"+k for k in AGGREGATES]]
    incomplete=summary_frame.loc[summary_frame[coverage_columns].ne(18).any(axis=1),["model",*coverage_columns]]
    if not incomplete.empty:
        raise ValueError("S43 resource and performance means require 18/18 NT task records:\n"+incomplete.to_string(index=False))
    changes=[]
    a=old.set_index("model")
    for _,r in summary_frame.iterrows():
        if r.model not in a.index:
            changes.append({"model":r.model,"field":"model_row","before":None,"after":"included"})
            continue
        for k in ["mean_mcc","n_perf",*AGGREGATES,*["n_"+k for k in AGGREGATES],"n_eff","mean_rank","n_rank_tasks"]:
            if k not in a.columns:
                changes.append({"model":r.model,"field":k,"before":None,"after":float(r[k])})
                continue
            if not np.isclose(a.loc[r.model,k],r[k],rtol=1e-10,equal_nan=True):
                changes.append({"model":r.model,"field":k,"before":float(a.loc[r.model,k]),"after":float(r[k])})
    current_json_rows=sum((source/Path(rel)).is_file() for rel in detail.source_file)
    report={"status":"PASS","table_ids":["S43","S44"],"models":len(eligible),"tasks":len(tasks),
           "task_rows":len(detail),"current_json_rows":int(current_json_rows),
           "retained_published_rows":len(retained_published_rows),
           "missing_results":detail.loc[detail.result_status.eq("MISSING_FILE"),["model","task"]].to_dict("records"),
           "newly_included_models":sorted(eligible-old_models),
           "aggregation":"MCC arithmetic mean; each efficiency metric arithmetic mean of finite measured values; mean rank is an equal-weight arithmetic mean across available task ranks",
           "imputation":"none","changes_from_previous_S43":changes}
    if check_only:
        assert_current(tables/SUMMARY,summary_frame)
        assert_current(tables/DETAIL,detail)
    else:
        detail.to_csv(tables/DETAIL,index=False,na_rep="NA")
        summary_frame.to_csv(tables/SUMMARY,index=False,na_rep="NA")
        report["outputs_sha256"]={name:digest(tables/name) for name in [SUMMARY,DETAIL]}
    print(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True,help="Probe_NT_Pareto task/model JSON tree")
    p.add_argument("--tables-dir",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--check-only",action="store_true",help="Fail if either published table differs from source JSONs; write nothing")
    a=p.parse_args()
    build(a.source_dir,a.tables_dir,a.check_only)
if __name__=="__main__":main()
