from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

def generate_weekly_report(snapshot_dir, report_path):
    """
    Reads snapshot + daily evaluation CSVs and builds a PDF report.
    """
    # --- Load snapshot ---
    # Allow passing either the snapshot directory or the snapshot.csv file path
    if os.path.isdir(snapshot_dir):
        snapshot_path = os.path.join(snapshot_dir, "snapshot.csv")
    elif os.path.isfile(snapshot_dir) and snapshot_dir.endswith('.csv'):
        snapshot_path = snapshot_dir
        snapshot_dir = os.path.dirname(snapshot_dir)
    else:
        raise FileNotFoundError(f"Snapshot dir or file not found: {snapshot_dir}")
    snapshot = pd.read_csv(snapshot_path)

    # --- Load daily evals ---
    eval_files = sorted(glob.glob(os.path.join(snapshot_dir, "eval", "*_eval.csv")))
    daily_dfs = [pd.read_csv(f) for f in eval_files]
    
    # Combine all daily evals (if any)
    daily_all = pd.concat(daily_dfs, keys=[os.path.basename(f).replace("_eval.csv","") for f in eval_files]) if daily_dfs else None

    # directional accuracy
    def directional_acc(df, pred_col):
        df["actual_dir"] = df["actual_price"].diff().apply(lambda x: 1 if x > 0 else -1)
        df["pred_dir"] = df[pred_col].diff().apply(lambda x: 1 if x > 0 else -1)
        return (df["actual_dir"].dropna() == df["pred_dir"].dropna()).mean()
    
    # --- Summary metrics ---
    if daily_dfs:
        mae_xgb = np.mean([mean_absolute_error(df["actual_price"], df.get("pred_xgb")) for df in daily_dfs if "pred_xgb" in df.columns])
        mae_lstm = np.mean([mean_absolute_error(df["actual_price"], df.get("pred_lstm")) for df in daily_dfs if "pred_lstm" in df.columns])
        acc_xgb = np.mean([directional_acc(df.copy(), "pred_xgb") for df in daily_dfs if "pred_xgb" in df.columns])
        acc_lstm = np.mean([directional_acc(df.copy(), "pred_lstm") for df in daily_dfs if "pred_lstm" in df.columns])
        p20_xgb = np.mean([df.get("p20_xgb", pd.Series([np.nan])).iloc[0] for df in daily_dfs])
        p20_lstm = np.mean([df.get("p20_lstm", pd.Series([np.nan])).iloc[0] for df in daily_dfs])
    else:
        mae_xgb = mae_lstm = acc_xgb = acc_lstm = p20_xgb = p20_lstm = np.nan

    summary_metrics = pd.DataFrame({
        "Model": ["XGB", "LSTM"],
        "MAE": [round(mae_xgb,2), round(mae_lstm,2)],
        "Dir Accuracy": [f"{acc_xgb:.0%}", f"{acc_lstm:.0%}"],
        "Precision@20": [f"{p20_xgb:.0%}", f"{p20_lstm:.0%}"]
    })

    acc_xgb = np.mean([directional_acc(df.copy(), "pred_xgb") for df in daily_dfs]) if daily_dfs else np.nan
    acc_lstm = np.mean([directional_acc(df.copy(), "pred_lstm") for df in daily_dfs]) if daily_dfs else np.nan

    summary_metrics = pd.DataFrame({
        "Model": ["XGB", "LSTM"],
        "MAE": [round(mae_xgb,2), round(mae_lstm,2)],
        "RMSE": ["-", "-"],  # optional
        "Dir Accuracy": [f"{acc_xgb:.0%}", f"{acc_lstm:.0%}"],
        "Precision@20": ["TBD", "TBD"] # fill in if you calculate this
    })

    # --- Daily accuracy table ---
    daily_accuracy = []
    if daily_dfs:
        for f, df in zip(eval_files, daily_dfs):
            day = os.path.basename(f).replace("_eval.csv", "")
            acc_x = directional_acc(df.copy(), "pred_xgb") if "pred_xgb" in df.columns else np.nan
            acc_l = directional_acc(df.copy(), "pred_lstm") if "pred_lstm" in df.columns else np.nan
            daily_accuracy.append([day, acc_x, acc_l])
    daily_metrics = pd.DataFrame(daily_accuracy, columns=["Date", "XGB_Acc", "LSTM_Acc"]) if daily_accuracy else pd.DataFrame(columns=["Date","XGB_Acc","LSTM_Acc"])

    # --- Daily metrics (Accuracy + Precision@20) ---
    daily_stats = []
    if daily_dfs:
        for f, df in zip(eval_files, daily_dfs):
            day = os.path.basename(f).replace("_eval.csv", "")
            acc_x = directional_acc(df.copy(), "pred_xgb") if "pred_xgb" in df.columns else np.nan
            acc_l = directional_acc(df.copy(), "pred_lstm") if "pred_lstm" in df.columns else np.nan
            p20_x = df.get("p20_xgb", pd.Series([np.nan])).iloc[0]
            p20_l = df.get("p20_lstm", pd.Series([np.nan])).iloc[0]
            daily_stats.append([day, acc_x, acc_l, p20_x, p20_l])
    daily_metrics = pd.DataFrame(daily_stats, columns=["Date", "XGB_Acc", "LSTM_Acc", "XGB_P@20", "LSTM_P@20"]) if daily_stats else pd.DataFrame(columns=["Date","XGB_Acc","LSTM_Acc","XGB_P@20","LSTM_P@20"])


    # --- Top mispredictions (worst errors) ---
    latest_eval = daily_dfs[-1].copy()  # use Friday's eval
    latest_eval["abs_err"] = abs(latest_eval["error_xgb"])
    mispred = latest_eval.nlargest(5, "abs_err")[["symbol","pred_xgb","actual_price","error_xgb"]]
    mispred.columns = ["Symbol","Predicted","Actual","Error"]
    mispred["Attribution"] = "TBD (SHAP/analysis)"
    mispredictions = pd.concat([pd.DataFrame([["Symbol","Predicted","Actual","Error","Attribution"]], 
                                             columns=["Symbol","Predicted","Actual","Error","Attribution"]), 
                                mispred])

    # --- Drift Report (placeholder: compare snapshot vs last eval) ---
    drift_report = pd.DataFrame([
        ["Feature", "Drift Status", "Change"],
        ["RSI", "Stable", "p=0.25"],
        ["Volume", "Drifted", "p=0.01"],
        ["MACD", "Stable", "p=0.20"]
    ])

    # --- Attribution (placeholder: from SHAP/IG if available) ---
    attribution = pd.DataFrame([
        ["Feature", "Importance"],
        ["Momentum", "30%"],
        ["Moving Avg", "25%"],
        ["Volume", "20%"],
        ["RSI", "15%"],
        ["MACD", "10%"]
    ])

    # --- Build PDF ---
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Weekly Prediction Performance Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # Executive Summary
    elements.append(Paragraph("<b>1. Executive Summary</b>", styles['Heading2']))
    summary_table = Table([summary_metrics.columns.tolist()] + summary_metrics.values.tolist())
    summary_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.grey),
                                       ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                                       ("ALIGN", (0,0), (-1,-1), "CENTER"),
                                       ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # --- Daily Performance Chart (Accuracy) ---
    elements.append(Paragraph("<b>2. Daily Directional Accuracy</b>", styles['Heading2']))
    plt.figure(figsize=(6,4))
    plt.bar(daily_metrics["Date"], daily_metrics["XGB_Acc"], alpha=0.6, label="XGB")
    plt.bar(daily_metrics["Date"], daily_metrics["LSTM_Acc"], alpha=0.6, label="LSTM")
    plt.ylabel("Directional Accuracy")
    plt.title("Daily Accuracy")
    plt.legend()
    acc_chart_path = os.path.join(snapshot_dir, "daily_accuracy.png")
    plt.savefig(acc_chart_path, bbox_inches="tight")
    plt.close()
    elements.append(Image(acc_chart_path, width=400, height=250))
    elements.append(Spacer(1, 12))

    # --- Daily Performance Chart (Precision@20) ---
    elements.append(Paragraph("<b>3. Daily Precision@20</b>", styles['Heading2']))
    plt.figure(figsize=(6,4))
    plt.plot(daily_metrics["Date"], daily_metrics["XGB_P@20"], marker="o", label="XGB")
    plt.plot(daily_metrics["Date"], daily_metrics["LSTM_P@20"], marker="o", label="LSTM")
    plt.ylabel("Precision@20")
    plt.ylim(0, 1)
    plt.title("Daily Precision@20")
    plt.legend()
    p20_chart_path = os.path.join(snapshot_dir, "daily_p20.png")
    plt.savefig(p20_chart_path, bbox_inches="tight")
    plt.close()
    elements.append(Image(p20_chart_path, width=400, height=250))
    elements.append(Spacer(1, 12))



    # Top Mis-predicted Stocks
    elements.append(Paragraph("<b>3. Top Mis-predicted Stocks</b>", styles['Heading2']))
    mispred_table = Table(mispredictions.values.tolist())
    mispred_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                       ("ALIGN", (0,0), (-1,-1), "CENTER"),
                                       ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
    elements.append(mispred_table)
    elements.append(Spacer(1, 12))

    # Feature Drift
    elements.append(Paragraph("<b>4. Feature Drift Analysis</b>", styles['Heading2']))
    drift_table = Table(drift_report.values.tolist())
    drift_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                     ("ALIGN", (0,0), (-1,-1), "CENTER"),
                                     ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
    elements.append(drift_table)
    elements.append(Spacer(1, 12))

    # Attribution
    elements.append(Paragraph("<b>5. Attribution Insights</b>", styles['Heading2']))
    attr_table = Table(attribution.values.tolist())
    attr_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                                    ("GRID", (0,0), (-1,-1), 0.5, colors.black)]))
    elements.append(attr_table)
    elements.append(Spacer(1, 12))

    # Recommendations
    elements.append(Paragraph("<b>6. Recommendations</b>", styles['Heading2']))
    recommendations = [
        "Down-weight volume-based signals if drift detected.",
        "Add sentiment/macro indicators for sector shocks.",
        "Retrain model if cumulative error exceeds threshold.",
        "Monitor LSTM vs XGB weekly and drop underperforming configs."
    ]
    for rec in recommendations:
        elements.append(Paragraph(f"- {rec}", styles['Normal']))
    elements.append(Spacer(1, 12))

    doc.build(elements)
    print(f"✅ Report generated: {report_path}")


import argparse
from sklearn.metrics import mean_absolute_error


def main():
    parser = argparse.ArgumentParser(description="Generate weekly PDF report from snapshot and evals")
    parser.add_argument("--snapshot_dir", required=False, default=None, help="Path to snapshot directory OR snapshot.csv (e.g., snapshot/2025-08-18 or .../snapshot.csv)")
    parser.add_argument("--out", default=None, help="Output PDF path (default reports/<date>_report.pdf)")
    parser.add_argument("--latest", action='store_true', help="Use the latest snapshot directory automatically")
    args = parser.parse_args()

    snap_arg = args.snapshot_dir
    if args.latest or not snap_arg:
        cand = sorted([p for p in glob.glob(os.path.join('snapshot', '*')) if os.path.isdir(p)])
        if not cand:
            raise FileNotFoundError("No snapshot directories found under ./snapshot")
        snap_arg = cand[-1]

    # Derive date label for output path
    base_for_label = snap_arg if os.path.isdir(snap_arg) else os.path.dirname(snap_arg)
    date_label = os.path.basename(os.path.normpath(base_for_label))
    report_path = args.out or os.path.join("reports", f"{date_label}_report.pdf")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    generate_weekly_report(snap_arg, report_path)


if __name__ == "__main__":
    main()
