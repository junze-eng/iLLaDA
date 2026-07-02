import os
import sys
import argparse
from pathlib import Path

# Try importing required visualization libraries
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    print(f"Error: Missing visualization dependencies ({exc}).", file=sys.stderr)
    print("Please install them in your virtual environment by running:", file=sys.stderr)
    print("  pip install pandas matplotlib seaborn", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot iLLaDA/LLaDA benchmark results from summary_all.csv.")
    parser.add_argument(
        "--csv", 
        default="outputs/illada_runs/summary_all.csv", 
        help="Path to the summary_all.csv file."
    )
    parser.add_argument(
        "--output-dir", 
        default="outputs/plots", 
        help="Directory to save the generated charts."
    )
    return parser.parse_args()


def clean_dataframe(df):
    """Clean the data types and null representations in the CSV."""
    df = df.copy()
    
    # Standardize column names (stripping whitespace)
    df.columns = [col.strip() for col in df.columns]
    
    # List of parameter columns to convert to numeric where possible
    numeric_params = [
        'primary_metric_value',
        'latency_mean_s',
        'tokens_per_second_mean',
        'param_gen_blocksize',
        'param_gen_steps',
        'param_gen_length',
        'param_context_length',
        'param_context_prefix_tokens'
    ]
    
    for col in numeric_params:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
            
    # Handle confidence threshold strings ('None', 'null' or float values)
    thresh_col = 'param_token_selection_confidence_threshold'
    if thresh_col in df.columns:
        df[thresh_col] = df[thresh_col].astype(str).str.strip().str.lower()
        df[thresh_col] = df[thresh_col].replace({'none': 'No Threshold', 'null': 'No Threshold'})
        
    return df


TASK_NAME_PREFIXES = {
    'parallel': 'task1_',
    'context': 'task2_',
    'arness': 'task3_',
}


def filter_task(df, task_name):
    """Select rows for a task group (parallel/context/arness).

    test_config.yaml names experiments task1_*/task2_*/task3_* for the
    parallel/context/arness groups respectively, so match on that prefix.
    Falls back to substring-matching the task name itself in `experiment`
    for older configs that don't follow the task1_/task2_/task3_ convention.
    """
    prefix = TASK_NAME_PREFIXES.get(task_name)
    if prefix:
        matched = df[df['experiment'].str.startswith(prefix, na=False)].copy()
        if not matched.empty:
            return matched
    return df[df['experiment'].str.contains(task_name, case=False, na=False)].copy()


def plot_parallel_capability(df, output_dir):
    """Plot Parallel Capability: Accuracy & Latency vs Block Size."""
    target_df = filter_task(df, 'parallel')
    if target_df.empty:
        print("[-] No parallel task data found. Skipping parallel capability plot.")
        return

    # `param_gen_blocksize` doesn't vary within a benchmark in the task1_* configs (it's fixed
    # per benchmark); `param_arness` (1/4/8) is the actual parallelism knob that varies across
    # ar_like/mild_parallel/strong_parallel, so that's the meaningful x-axis. Fall back to
    # blocksize for older configs that don't set `arness` as a param.
    x_col = 'param_arness' if 'param_arness' in target_df.columns and target_df['param_arness'].notna().any() else 'param_gen_blocksize'
    y_col = 'primary_metric_value'

    if x_col not in target_df.columns or y_col not in target_df.columns:
        print(f"[-] Missing columns for parallel plot ({x_col} or {y_col}). Skipping.")
        return

    sns.set_theme(style="whitegrid")
    x_label = "ARness (Steps per Token, Higher = More Parallel)" if x_col == 'param_arness' else "Block Size (Parallel Token Commitment)"

    # 1. ARness vs Accuracy Plot
    plt.figure(figsize=(10, 6))
    hue_col = 'benchmark'

    sns.lineplot(
        data=target_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        marker='o',
        linewidth=2,
        markersize=8
    )

    plt.title("Parallel Capability: Effect of Parallelism on Performance", fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Accuracy / Score", fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(sorted(target_df[x_col].dropna().unique()), labels=[str(int(x)) for x in sorted(target_df[x_col].dropna().unique())])
    plt.legend(title="Benchmark", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = output_dir / "parallel_performance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved parallel performance chart to {save_path}")

    # 2. ARness vs Generation Speed
    speed_col = 'tokens_per_second_mean'
    if speed_col in target_df.columns and not target_df[speed_col].isna().all():
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=target_df,
            x=x_col,
            y=speed_col,
            hue=hue_col,
            marker='s',
            linewidth=2,
            markersize=8,
            palette="viridis"
        )
        plt.title("Generation Speed: Effect of Parallelism on Throughput", fontsize=14, fontweight='bold')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("Tokens Per Second", fontsize=12)
        plt.xscale('log', base=2)
        plt.xticks(sorted(target_df[x_col].dropna().unique()), labels=[str(int(x)) for x in sorted(target_df[x_col].dropna().unique())])
        plt.legend(title="Benchmark", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path_speed = output_dir / "parallel_speed.png"
        plt.savefig(save_path_speed, dpi=300)
        plt.close()
        print(f"[+] Saved parallel speed chart to {save_path_speed}")


def plot_context_window(df, output_dir):
    """Plot Context Window: Performance vs Context Length."""
    target_df = filter_task(df, 'context')
    if target_df.empty:
        print("[-] No context task data found. Skipping context window plot.")
        return

    # Check for context length column
    x_col = None
    for candidate in ['param_context_length', 'param_context_prefix_tokens']:
        if candidate in target_df.columns and not target_df[candidate].isna().all():
            x_col = candidate
            break
            
    y_col = 'primary_metric_value'
    
    if not x_col or y_col not in target_df.columns:
        print(f"[-] Missing columns for context window plot. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Use position/needle_position as style/hue if present (needle passkey task)
    hue_col = 'benchmark'
    style_col = None
    if 'param_needle_position' in target_df.columns:
        hue_col = 'param_needle_position'
        
    sns.lineplot(
        data=target_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=style_col,
        marker='o',
        linewidth=2.5,
        markersize=8,
        palette="crest"
    )
    
    label_name = "Context Length (Tokens)" if x_col == 'param_context_length' else "Context Prefix Distractor Tokens"
    plt.title("Context Window Performance (Needle / Distractor Scaling)", fontsize=14, fontweight='bold')
    plt.xlabel(label_name, fontsize=12)
    plt.ylabel("Accuracy / Retrieval Score", fontsize=12)
    
    # Mark x ticks clearly
    x_vals = sorted(target_df[x_col].dropna().unique())
    plt.xticks(x_vals, labels=[str(int(x)) for x in x_vals])
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = output_dir / "context_window_performance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved context window chart to {save_path}")


def plot_arness(df, output_dir):
    """Plot ARness: Confidence Threshold & Blocksize vs Performance."""
    target_df = filter_task(df, 'arness')
    if target_df.empty:
        print("[-] No arness task data found. Skipping ARness plot.")
        return

    x_col = 'param_token_selection_confidence_threshold'
    y_col = 'primary_metric_value'
    hue_col = 'param_gen_blocksize'
    
    if x_col not in target_df.columns or y_col not in target_df.columns:
        print(f"[-] Missing columns for ARness plot. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # We can plot as a barplot or grouped pointplot since confidence threshold is categorical
    # Convert blocksize to string for discrete colors
    if hue_col in target_df.columns:
        target_df[hue_col] = "Blocksize: " + target_df[hue_col].astype(str)
        
    sns.pointplot(
        data=target_df,
        x=x_col,
        y=y_col,
        hue=hue_col if hue_col in target_df.columns else None,
        markers=['o', '^', 's', 'd'][:target_df[hue_col].nunique() if hue_col in target_df.columns else 1],
        linestyles='--',
        dodge=True,
        capsize=0.1
    )
    
    plt.title("ARness Analysis: Confidence Threshold vs Performance", fontsize=14, fontweight='bold')
    plt.xlabel("Token Selection Confidence Threshold", fontsize=12)
    plt.ylabel("Accuracy / Score", fontsize=12)
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = output_dir / "arness_threshold_performance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved ARness threshold chart to {save_path}")


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    
    if not csv_path.exists():
        print(f"Error: Summary file '{csv_path}' does not exist.", file=sys.stderr)
        print("Please run your benchmark experiments using `python run_test.py` first to generate results.", file=sys.stderr)
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and clean CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)
        
    if df.empty:
        print("Warning: CSV file is empty. No charts to plot.", file=sys.stderr)
        sys.exit(0)
        
    print(f"[+] Loaded {len(df)} records from {csv_path}")
    df_cleaned = clean_dataframe(df)
    
    # Generate plots
    plot_parallel_capability(df_cleaned, output_dir)
    plot_context_window(df_cleaned, output_dir)
    plot_arness(df_cleaned, output_dir)
    
    print(f"[+] Plotting complete. Charts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
