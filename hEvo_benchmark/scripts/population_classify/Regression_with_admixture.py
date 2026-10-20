"""Few-shot ancestry regression from the retained original benchmark implementation.

Numerical protocol retained; runtime paths are configurable. See README.
"""

import argparse
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from embedding_loader import available_model_names, load_embedding_map_for_model

try:
    from rich.console import Console
    from rich.table import Table

    CONSOLE = Console()
except ImportError:
    CONSOLE = None


BASE_DIR = str(
    Path(__file__).resolve().parents[2]
    / "results/1kg_popu_classification/2000v_1500s_511bp"
)
GT_PATH = str(
    Path(__file__).resolve().parents[2]
    / "results/ancestry_analysis/1kg_final_for_admixture.5.Q.merged"
)
REPORT_DIR = os.path.join(BASE_DIR, "report")
POPS = ["AFR", "AMR", "EUR", "EAS", "SAS"]
DEFAULT_TARGET_MODEL = "hEvo-60k-sliding128-340M"
DEFAULT_MIN_SHOTS = 10
DEFAULT_MAX_SHOTS = 60
DEFAULT_NUM_REPEATS = 10
DEFAULT_GLOBAL_SEED = 10
NUM_WORKERS = min(mp.cpu_count(), 16)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot ancestry regression benchmark with repeated random sampling "
            "and rank monitoring for a target model."
        )
    )
    parser.add_argument("--base-dir", default=BASE_DIR)
    parser.add_argument("--gt-path", default=GT_PATH)
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--min-shots", type=int, default=DEFAULT_MIN_SHOTS)
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS)
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_GLOBAL_SEED)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def load_ground_truth(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = ["SampleID", "AFR", "AMR", "EUR", "EAS", "SAS", "SuperPop"]
    return df.set_index("SampleID")


def load_single_model(args):
    model_name, base_dir = args
    try:
        emb_result = load_embedding_map_for_model(base_dir, model_name)
    except Exception as exc:
        return model_name, {"error": str(exc)}
    if len(emb_result) < 100:
        return model_name, {"error": f"npy files too few ({len(emb_result)})"}
    return model_name, emb_result


def build_model_cache(base_dir, gt_df, num_workers):
    tasks = [(model_name, base_dir) for model_name in available_model_names(base_dir)]

    with mp.Pool(processes=num_workers) as pool:
        loaded_results = pool.map(load_single_model, tasks)

    model_cache = {}
    failed_models = []
    for model_name, emb_result in loaded_results:
        if isinstance(emb_result, dict) and "error" in emb_result:
            failed_models.append({"Model": model_name, "Reason": emb_result["error"]})
            continue

        common_ids = sorted(set(emb_result.keys()) & set(gt_df.index))
        if not common_ids:
            failed_models.append(
                {"Model": model_name, "Reason": "no shared SampleID with ground truth"}
            )
            continue

        X = np.array([emb_result[sid][1] for sid in common_ids], dtype=np.float32)
        Y = gt_df.loc[common_ids, POPS].values.astype(np.float32)
        groups = gt_df.loc[common_ids, "SuperPop"].values

        model_cache[model_name] = {
            "X": X,
            "Y": Y,
            "groups": groups,
            "sample_ids": common_ids,
            "n_samples": len(common_ids),
        }

    return model_cache, failed_models


def evaluate_single_model(model_name, model_data, n_shots, seed, alpha):
    groups = model_data["groups"]
    unique_groups = sorted(np.unique(groups))
    total_needed = n_shots * len(unique_groups)

    if model_data["n_samples"] <= total_needed:
        return {
            "error": f"not enough samples ({model_data['n_samples']}) for {total_needed} shots"
        }

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for group in unique_groups:
        idx_in_group = np.where(groups == group)[0]
        shuffled_idx = idx_in_group.copy()
        rng.shuffle(shuffled_idx)

        if len(shuffled_idx) <= n_shots:
            return {
                "error": f"group {group} has only {len(shuffled_idx)} samples, cannot allocate {n_shots} shots"
            }

        train_indices.extend(shuffled_idx[:n_shots])
        test_indices.extend(shuffled_idx[n_shots:])

    if not test_indices:
        return {"error": "empty test set"}

    X_train = model_data["X"][train_indices]
    X_test = model_data["X"][test_indices]
    Y_train = model_data["Y"][train_indices]
    Y_test = model_data["Y"][test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg = Ridge(alpha=alpha)
    reg.fit(X_train, Y_train)
    preds = softmax(reg.predict(X_test))

    pop_metrics = []
    for i, pop in enumerate(POPS):
        y_true = Y_test[:, i]
        y_pred = preds[:, i]

        if np.unique(y_pred).size > 1 and np.std(y_true) > 0:
            corr = float(pearsonr(y_true, y_pred)[0])
        else:
            corr = 0.0

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        pop_metrics.append(
            {
                "Model": model_name,
                "Component": pop,
                "Correlation": corr,
                "RMSE": rmse,
                "Shot": n_shots,
                "Seed": seed,
            }
        )

    return {
        "avg_corr": float(np.mean([m["Correlation"] for m in pop_metrics])),
        "avg_rmse": float(np.mean([m["RMSE"] for m in pop_metrics])),
        "test_size": len(test_indices),
        "detail_df": pd.DataFrame(pop_metrics),
    }


def run_benchmark(model_cache, failed_models, shots_list, seeds, alpha, target_model):
    run_rows = []
    detail_frames = []
    target_rows = []
    runtime_failures = list(failed_models)

    for n_shots in shots_list:
        for seed in seeds:
            current_results = []
            for model_name, model_data in model_cache.items():
                res = evaluate_single_model(
                    model_name=model_name,
                    model_data=model_data,
                    n_shots=n_shots,
                    seed=seed,
                    alpha=alpha,
                )
                if "error" in res:
                    runtime_failures.append(
                        {
                            "Model": model_name,
                            "Reason": res["error"],
                            "Shot": n_shots,
                            "Seed": seed,
                        }
                    )
                    continue

                run_rows.append(
                    {
                        "Shot": n_shots,
                        "Seed": seed,
                        "Model": model_name,
                        "Avg_Corr": res["avg_corr"],
                        "Avg_RMSE": res["avg_rmse"],
                        "Test_Samples": res["test_size"],
                    }
                )
                detail_frames.append(res["detail_df"])
                current_results.append(
                    {
                        "Model": model_name,
                        "Avg_Corr": res["avg_corr"],
                        "Avg_RMSE": res["avg_rmse"],
                        "Test_Samples": res["test_size"],
                    }
                )

            if not current_results:
                continue

            ranked_df = pd.DataFrame(current_results).sort_values(
                ["Avg_Corr", "Avg_RMSE", "Model"], ascending=[False, True, True]
            )
            ranked_df = ranked_df.reset_index(drop=True)
            ranked_df["Rank"] = ranked_df.index + 1
            ranked_df["Shot"] = n_shots
            ranked_df["Seed"] = seed

            target_match = ranked_df[ranked_df["Model"] == target_model]
            if not target_match.empty:
                row = target_match.iloc[0]
                target_rows.append(
                    {
                        "Shot": n_shots,
                        "Seed": seed,
                        "Model": target_model,
                        "Rank": int(row["Rank"]),
                        "Total_Models": int(len(ranked_df)),
                        "Avg_Corr": float(row["Avg_Corr"]),
                        "Avg_RMSE": float(row["Avg_RMSE"]),
                        "Test_Samples": int(row["Test_Samples"]),
                        "Top1": int(row["Rank"] == 1),
                        "Top3": int(row["Rank"] <= 3),
                        "Top5": int(row["Rank"] <= 5),
                    }
                )

    return run_rows, detail_frames, target_rows, runtime_failures


def build_target_summary(target_df):
    if target_df.empty:
        return pd.DataFrame()

    summary_df = (
        target_df.groupby("Shot")
        .agg(
            Runs=("Rank", "count"),
            Mean_Rank=("Rank", "mean"),
            Median_Rank=("Rank", "median"),
            Best_Rank=("Rank", "min"),
            Worst_Rank=("Rank", "max"),
            Mean_Corr=("Avg_Corr", "mean"),
            Mean_RMSE=("Avg_RMSE", "mean"),
            Top1_Count=("Top1", "sum"),
            Top3_Count=("Top3", "sum"),
            Top5_Count=("Top5", "sum"),
        )
        .reset_index()
    )

    summary_df["Top1_Rate"] = summary_df["Top1_Count"] / summary_df["Runs"]
    summary_df["Top3_Rate"] = summary_df["Top3_Count"] / summary_df["Runs"]
    summary_df["Top5_Rate"] = summary_df["Top5_Count"] / summary_df["Runs"]
    return summary_df.sort_values("Shot")


def print_target_summary(target_model, target_summary_df):
    if target_summary_df.empty:
        print(
            f"\nTarget model {target_model} was not found in any successful benchmark run."
        )
        return

    if CONSOLE:
        table = Table(title=f"Target Model Rank Monitor: {target_model}")
        table.add_column("Shot", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("Mean Rank", justify="right")
        table.add_column("Best", justify="right")
        table.add_column("Worst", justify="right")
        table.add_column("Top1 Rate", justify="right")
        table.add_column("Top3 Rate", justify="right")

        for row in target_summary_df.itertuples():
            table.add_row(
                str(int(row.Shot)),
                str(int(row.Runs)),
                f"{row.Mean_Rank:.2f}",
                str(int(row.Best_Rank)),
                str(int(row.Worst_Rank)),
                f"{row.Top1_Rate:.2%}",
                f"{row.Top3_Rate:.2%}",
            )
        CONSOLE.print(table)
    else:
        print("\nTarget model rank summary:")
        print(target_summary_df.to_string(index=False))


def save_reports(
    report_dir, timestamp, run_rows, detail_frames, target_rows, failed_rows
):
    os.makedirs(report_dir, exist_ok=True)

    run_df = pd.DataFrame(run_rows)
    run_path = os.path.join(report_dir, f"fewshot_all_runs_{timestamp}.csv")
    run_df.to_csv(run_path, index=False)

    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)
        detail_path = os.path.join(report_dir, f"fewshot_detailed_{timestamp}.csv")
        detail_df.to_csv(detail_path, index=False)
    else:
        detail_path = None

    target_df = pd.DataFrame(target_rows)
    target_path = os.path.join(report_dir, f"target_rank_runs_{timestamp}.csv")
    target_df.to_csv(target_path, index=False)

    target_summary_df = build_target_summary(target_df)
    target_summary_path = os.path.join(
        report_dir, f"target_rank_summary_{timestamp}.csv"
    )
    target_summary_df.to_csv(target_summary_path, index=False)

    fail_df = pd.DataFrame(failed_rows)
    fail_path = os.path.join(report_dir, f"fewshot_failed_{timestamp}.csv")
    fail_df.to_csv(fail_path, index=False)

    overall_summary_path = None
    if not run_df.empty:
        overall_summary_df = (
            run_df.groupby("Model")
            .agg(
                Runs=("Avg_Corr", "count"),
                Mean_Avg_Corr=("Avg_Corr", "mean"),
                Std_Avg_Corr=("Avg_Corr", "std"),
                Mean_Avg_RMSE=("Avg_RMSE", "mean"),
            )
            .reset_index()
            .sort_values(
                ["Mean_Avg_Corr", "Mean_Avg_RMSE", "Model"],
                ascending=[False, True, True],
            )
        )
        overall_summary_path = os.path.join(
            report_dir, f"overall_model_summary_{timestamp}.csv"
        )
        overall_summary_df.to_csv(overall_summary_path, index=False)

    return {
        "run_path": run_path,
        "detail_path": detail_path,
        "target_path": target_path,
        "target_summary_path": target_summary_path,
        "fail_path": fail_path,
        "overall_summary_path": overall_summary_path,
        "target_summary_df": target_summary_df,
    }


def main():
    args = parse_args()
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    if args.min_shots <= 0 or args.max_shots < args.min_shots:
        raise ValueError("invalid shot range")
    if args.num_repeats <= 0:
        raise ValueError("num-repeats must be positive")

    shots_list = list(range(args.min_shots, args.max_shots + 1))
    seeds = [args.seed_start + i for i in range(args.num_repeats)]

    gt_df = load_ground_truth(args.gt_path)
    model_cache, failed_models = build_model_cache(
        base_dir=args.base_dir,
        gt_df=gt_df,
        num_workers=args.num_workers,
    )

    print(
        f"Few-shot ancestry regression | models={len(model_cache)} | "
        f"shots={args.min_shots}-{args.max_shots} | repeats={args.num_repeats} | "
        f"target={args.target_model}"
    )

    run_rows, detail_frames, target_rows, runtime_failures = run_benchmark(
        model_cache=model_cache,
        failed_models=failed_models,
        shots_list=shots_list,
        seeds=seeds,
        alpha=args.alpha,
        target_model=args.target_model,
    )

    report_info = save_reports(
        report_dir=args.report_dir,
        timestamp=timestamp,
        run_rows=run_rows,
        detail_frames=detail_frames,
        target_rows=target_rows,
        failed_rows=runtime_failures,
    )

    print_target_summary(args.target_model, report_info["target_summary_df"])
    print(f"\nReports written to: {args.report_dir}")
    print(f"All runs: {report_info['run_path']}")
    print(f"Target ranks: {report_info['target_path']}")
    print(f"Target rank summary: {report_info['target_summary_path']}")
    if report_info["overall_summary_path"]:
        print(f"Overall model summary: {report_info['overall_summary_path']}")
    if report_info["detail_path"]:
        print(f"Detailed metrics: {report_info['detail_path']}")
    print(f"Failures: {report_info['fail_path']}")


if __name__ == "__main__":
    main()
