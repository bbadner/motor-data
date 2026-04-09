import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk
import threading
import statistics
import fitz
import queue
import calendar
import subprocess
import platform

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # This file must be in the same directory

# ---------- Utility: Open File ----------
def open_file(filepath):
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except Exception as e:
        print(f"[WARN] Could not open file: {e}")

# ---------- Data Extract Helpers ----------
def find_input_power_column(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        for idx, val in enumerate(row):
            if "input" in val and "power" in val:
                return idx
    return None

def read_input_power(filepath):
    try:
        excel = pd.ExcelFile(filepath)
        for sheet in excel.sheet_names:
            df = excel.parse(sheet, header=None)
            col = find_input_power_column(df)
            if col is not None:
                values = pd.to_numeric(df.iloc[10:, col], errors="coerce").dropna()
                if len(values) > 0:
                    return values.tolist()
        return []
    except Exception as e:
        print(f"[ERROR] Reading Excel: {e}")
        return []

def extract_date_from_filename(fname):
    """
    Extracts a date from a filename using the C6521 pattern.
    The format C6521YYMMDD means:
        - Year = 20YY
        - Month = MM
        - Day = DD
    """
    m = re.search(r"C6521(\d{6})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except Exception as e:
            print(f"[ERROR] Invalid C6521 date in filename: {fname} -> {e}")
            return None
    return None

# ---------- Motor Diagnostics Helpers ----------
def _safe_stdev(x):
    x = [v for v in x if v is not None]
    if len(x) < 2:
        return 0.0
    try:
        return float(statistics.stdev(x))
    except Exception:
        return 0.0

def _boxplot_whiskers(values):
    """
    Returns (lower_whisker, upper_whisker) using matplotlib's boxplot computation
    (1.5*IQR rule, clipped to data).
    """
    if not values:
        return (None, None)
    # matplotlib expects a list of datasets
    bp = plt.boxplot([values], showfliers=False)
    # whiskers are Line2D objects: [lower_line, upper_line]
    whisk_lines = bp.get("whiskers", [])
    try:
        lower = float(min(whisk_lines[0].get_ydata()))
        upper = float(max(whisk_lines[1].get_ydata()))
    except Exception:
        lower, upper = None, None
    plt.close()
    return lower, upper

def compute_month_stats(values):
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "lower_whisker": None,
            "upper_whisker": None,
        }
    vals = [float(v) for v in values]
    mean = float(statistics.mean(vals))
    median = float(statistics.median(vals))
    std = float(_safe_stdev(vals))
    lw, uw = _boxplot_whiskers(vals)
    return {
        "n": len(vals),
        "mean": mean,
        "median": median,
        "std": std,
        "lower_whisker": lw,
        "upper_whisker": uw,
    }

def compare_last_month_vs_history(last_stats, hist_stats_list, two_year_ago_stats=None):
    """
    Diagnostic decisioning for the most-recent month vs historical behavior.

    Uses ALL historical months (excluding the most recent month) as the baseline.

    Outputs:
      - For each metric (mean, median, std): z-score vs historical monthly distribution
        * |z| >= 3  -> DIFFERENT
        * 2 <= |z| < 3 -> WATCH
        * else -> OK
      - Optional: 2-year drift note (if the month ~24 months prior exists)
    """
    def _z(last, hist_vals):
        hist_vals = [v for v in hist_vals if v is not None]
        if last is None or len(hist_vals) < 2:
            return None
        mu = float(statistics.mean(hist_vals))
        sd = float(_safe_stdev(hist_vals))
        if sd == 0:
            return None
        return (float(last) - mu) / sd

    def _flag(z):
        if z is None:
            return "N/A"
        az = abs(z)
        if az >= 3:
            return "DIFFERENT"
        if az >= 2:
            return "WATCH"
        return "OK"

    out = {"metrics": {}, "overall": "OK"}

    for metric in ["mean", "median", "std"]:
        last_val = last_stats.get(metric)
        hist_vals = [h.get(metric) for h in hist_stats_list]
        z = _z(last_val, hist_vals)
        flag = _flag(z)
        out["metrics"][metric] = {
            "last": last_val,
            "z": z,
            "flag": flag,
            "hist_mean": float(statistics.mean([v for v in hist_vals if v is not None])) if len([v for v in hist_vals if v is not None]) >= 1 else None,
            "hist_std": float(_safe_stdev([v for v in hist_vals if v is not None])) if len([v for v in hist_vals if v is not None]) >= 2 else None,
        }

    # Overall status = worst of mean/median/std (and your rule: any being different)
    priority = {"DIFFERENT": 2, "WATCH": 1, "OK": 0, "N/A": 0}
    worst = "OK"
    for metric in out["metrics"].values():
        f = metric["flag"]
        if priority.get(f, 0) > priority.get(worst, 0):
            worst = f
    out["overall"] = worst

    # Optional: compare against a month approximately 2 years ago (if provided)
    if two_year_ago_stats:
        drift = {}
        for metric in ["mean", "median", "std"]:
            a = two_year_ago_stats.get(metric)
            b = last_stats.get(metric)
            if a is None or b is None:
                drift[metric] = {"two_year_ago": a, "last": b, "delta": None}
            else:
                drift[metric] = {"two_year_ago": float(a), "last": float(b), "delta": float(b) - float(a)}
        out["two_year_drift"] = drift

        # Add a "significant drift" marker if the delta exceeds 3*hist sigma of monthly metrics
        sig = {}
        for metric in ["mean", "median", "std"]:
            hist_vals = [h.get(metric) for h in hist_stats_list if h.get(metric) is not None]
            if metric in drift and drift[metric].get("delta") is not None and len(hist_vals) >= 2:
                sd = float(_safe_stdev(hist_vals))
                sig[metric] = (sd > 0 and abs(drift[metric]["delta"]) >= 3 * sd)
            else:
                sig[metric] = None
        out["two_year_drift_significant"] = sig

    return out



def build_watt_diagnostic_lines(hist_stats_list):
    """
    Diagnostic bands for the Motor 'Input Power (W)' boxplot.

    User intent:
      - Plot +/- 3σ lines around the HISTORICAL median (red)
      - Plot +/- 3σ lines around the HISTORICAL whiskers (blue)

    Per your preference ("within month"), σ is computed as a *typical within‑month sigma*:
      σ_typical = median( monthly_std ) across historical months (excluding the most recent month).

    Baselines:
      baseline_median  = median( monthly_median ) across historical months
      baseline_lw/uw   = median( monthly_lower_whisker / monthly_upper_whisker ) across historical months
    """
    hist_medians = [h.get("median") for h in hist_stats_list if h.get("median") is not None]
    hist_lw = [h.get("lower_whisker") for h in hist_stats_list if h.get("lower_whisker") is not None]
    hist_uw = [h.get("upper_whisker") for h in hist_stats_list if h.get("upper_whisker") is not None]
    hist_stds = [h.get("std") for h in hist_stats_list if h.get("std") is not None]

    def _baseline(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return float(statistics.median(vals))

    # Typical within-month sigma (median of monthly std devs)
    sigma_typical = None
    if len(hist_stds) >= 2:
        try:
            sigma_typical = float(statistics.median(hist_stds))
        except Exception:
            sigma_typical = None

    # Fallback if we can't compute a stable typical within-month sigma:
    # use month-to-month variation of the metric itself.
    def _fallback_sigma(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        return float(_safe_stdev(vals))

    med0 = _baseline(hist_medians)
    lw0 = _baseline(hist_lw)
    uw0 = _baseline(hist_uw)

    med_sigma = sigma_typical if sigma_typical is not None else _fallback_sigma(hist_medians)
    lw_sigma = sigma_typical if sigma_typical is not None else _fallback_sigma(hist_lw)
    uw_sigma = sigma_typical if sigma_typical is not None else _fallback_sigma(hist_uw)

    return {
        "median": {"baseline": med0, "sigma": med_sigma, "color": "red"},
        "lower_whisker": {"baseline": lw0, "sigma": lw_sigma, "color": "blue"},
        "upper_whisker": {"baseline": uw0, "sigma": uw_sigma, "color": "blue"},
    }

# ---------- PDF: Motor Report ----------
def create_motor_pdf(path, months, all_values, means, stds, motor_diag=None, diag_lines=None):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    # Boxplot with diagnostic lines
    plt.figure(figsize=(8, 4))
    plt.boxplot(all_values, labels=[m.strftime("%b %Y") for m in months], patch_artist=True, showfliers=False)
    plt.ylim(700, 1100)
    plt.title("Input Power (W): Monthly Distribution")
    plt.ylabel("Watts")
    plt.xticks(rotation=45)

    # Add requested +/-3 sigma diagnostic lines (historical-based)
    if diag_lines:
        # Median +/-3σ (red)
        med = diag_lines.get("median", {})
        if med.get("baseline") is not None and med.get("sigma") is not None:
            base = med["baseline"]
            sig = med["sigma"]
            plt.axhline(base + 3 * sig, color="red", linestyle="--", linewidth=1.3, label="Median + 3σ (hist)")
            plt.axhline(base - 3 * sig, color="red", linestyle="--", linewidth=1.3, label="Median - 3σ (hist)")

        # Whiskers +/-3σ (blue)
        for key, lab in [("lower_whisker", "Lower whisker"), ("upper_whisker", "Upper whisker")]:
            d = diag_lines.get(key, {})
            if d.get("baseline") is not None and d.get("sigma") is not None:
                base = d["baseline"]
                sig = d["sigma"]
                plt.axhline(base + 3 * sig, color="blue", linestyle="--", linewidth=1.1, label=f"{lab} ±3σ (hist)")
                plt.axhline(base - 3 * sig, color="blue", linestyle="--", linewidth=1.1)

        # Deduplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set()
        h2, l2 = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)
        if l2:
            plt.legend(h2, l2, loc="upper right", fontsize=7)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig("motor_boxplot_temp.png", dpi=150)
    plt.close()
    pdf.image("motor_boxplot_temp.png", x=10, y=30, w=190)

    # Std Dev chart
    plt.figure(figsize=(8, 2.8))
    plt.plot([m.strftime("%b %Y") for m in months], stds, marker='o', linestyle='-')
    plt.title("Standard Deviation Over Time")
    plt.ylabel("Std Dev")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("motor_std_temp.png", dpi=150)
    plt.close()
    pdf.image("motor_std_temp.png", x=10, y=115, w=190)

    # Page 2: Summary + Diagnostics
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, m in enumerate(months):
        pdf.cell(0, 8, f"{m.strftime('%Y-%m')}: Mean = {means[i]:.2f} W, Std Dev = {stds[i]:.4f}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Diagnostic (Last Month vs Historical)", ln=True)
    pdf.set_font("Arial", "", 11)

    if motor_diag:
        # Motor diagnostic summary (most recent month vs ALL historical months)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Motor Diagnostics (Input Power W):", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 7, f"Overall: {motor_diag.get('overall','')}", ln=True)

        metrics = motor_diag.get("metrics", {})
        for key, label in [("mean", "Mean"), ("median", "Median"), ("std", "Std Dev")]:
            m = metrics.get(key, {})
            last_v = m.get("last")
            hist_mu = m.get("hist_mean")
            hist_sd = m.get("hist_std")
            z = m.get("z")
            flag = m.get("flag", "")
            def _fmt(v):
                return "N/A" if v is None else f"{v:.2f}"
            ztxt = "N/A" if z is None else f"{z:.2f}"
            pdf.cell(0, 7, f"{label}: {flag} | last={_fmt(last_v)} | hist_mean={_fmt(hist_mu)} | hist_sd={_fmt(hist_sd)} | z={ztxt}", ln=True)

        # 2-year drift note (if available)
        drift = motor_diag.get("two_year_drift")
        if drift:
            sig = motor_diag.get("two_year_drift_significant", {})
            pdf.ln(1)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, "2-Year Drift Check:", ln=True)
            pdf.set_font("Arial", "", 11)
            for key, label in [("mean", "Mean"), ("median", "Median"), ("std", "Std Dev")]:
                d = drift.get(key, {})
                delta = d.get("delta")
                if delta is None:
                    pdf.cell(0, 7, f"{label}: N/A", ln=True)
                else:
                    mark = sig.get(key)
                    tag = " (SIGNIFICANT)" if mark else ""
                    pdf.cell(0, 7, f"{label}: delta vs ~2y ago = {delta:.2f}{tag}", ln=True)
    else:
        pdf.multi_cell(0, 6, "Not enough history to run diagnostics (need at least 3 months of data).")

    pdf.output(path)
    open_file(path)
    print(f"[INFO] ✅ Motor report created: {path}")

# ---------- PDF: Magnet Report ----------
def create_magnet_pdf(path, rows):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    # CpK chart
    months = [calendar.month_abbr[r[0].month] for r in rows]
    cpks = [r[4] for r in rows]

    plt.figure(figsize=(8, 4))
    plt.plot(months, cpks, marker='o', linestyle='-')
    plt.axhline(y=1.33, color='red', linestyle='-', linewidth=1.5, label="CpK Threshold = 1.33")
    plt.ylim(0, 4)
    plt.title("Magnet CpK Over Time")
    plt.xlabel("Month")
    plt.ylabel("CpK")
    plt.grid(True)
    plt.legend()
    # BIG status text across chart based on most recent CpK
    latest_cpk = cpks[-1] if cpks else None
    if latest_cpk is not None:
        if latest_cpk >= 1.33:
            plt.text(0.5, 0.5, "CpK is Good", transform=plt.gca().transAxes,
                     fontsize=22, fontweight="bold", ha="center", va="center", color="green", alpha=0.35)
        else:
            plt.text(0.5, 0.5, "CpK is BAD", transform=plt.gca().transAxes,
                     fontsize=22, fontweight="bold", ha="center", va="center", color="red", alpha=0.35)

    plt.tight_layout()
    plt.savefig("magnet_chart_temp.png", dpi=150)
    plt.close()
    pdf.image("magnet_chart_temp.png", x=10, y=30, w=190)

    # Summary table
    pdf.set_xy(10, 115)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Month Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for r in rows:
        d = r[0].strftime("%B %Y")
        pdf.cell(0, 8, f"{d} | Mean: {r[1]:.4f} | Std: {r[2]:.4f} | Cp: {r[3]:.2f} | CpK: {r[4]:.2f}", ln=True)

    pdf.output(path)
    open_file(path)
    print(f"[INFO] ✅ Magnet report created: {path}")

# ---------- Analysis Logic ----------
def run_motor_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx")]
    dated = [(extract_date_from_filename(f), f) for f in files]
    dated = [(d, f) for d, f in dated if d]
    monthly = {}

    for d, f in sorted(dated):
        m = d.strftime("%Y-%m")
        values = read_input_power(os.path.join(base, f))
        if values:
            monthly.setdefault(m, []).extend(values)

    if not monthly:
        print("❌ No Input Power data.")
        return

    all_months = sorted([datetime.strptime(k, "%Y-%m") for k in monthly])
    recent_months = all_months[-12:]  # show rolling 12 months on the chart
    all_values = [monthly[m.strftime("%Y-%m")] for m in recent_months]
    means = [statistics.mean(v) for v in all_values]
    stds = [_safe_stdev(v) for v in all_values]

    # Diagnostics baseline uses ALL historical months (excluding the most recent month)
    motor_diag = None
    diag_lines = None

    if len(all_months) >= 3:
        # Full-history month stats
        full_month_stats = []
        for mdt in all_months:
            vals = monthly.get(mdt.strftime("%Y-%m"), [])
            full_month_stats.append(compute_month_stats(vals))

        last_stats = full_month_stats[-1]
        hist_stats = full_month_stats[:-1]

        # Optional: compare to ~2 years ago (same calendar month if available)
        two_year_ago_stats = None
        try:
            last_month_dt = all_months[-1]
            target_2y = last_month_dt.replace(year=last_month_dt.year - 2)
            key_2y = target_2y.strftime("%Y-%m")
            if key_2y in monthly:
                two_year_ago_stats = compute_month_stats(monthly[key_2y])
        except Exception:
            two_year_ago_stats = None

        motor_diag = compare_last_month_vs_history(last_stats, hist_stats, two_year_ago_stats=two_year_ago_stats)
        diag_lines = build_watt_diagnostic_lines(hist_stats)
    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    path = os.path.join(report_dir, filename)
    create_motor_pdf(path, recent_months, all_values, means, stds, motor_diag=motor_diag, diag_lines=diag_lines)

def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    pdfs = [f for f in os.listdir(base) if f.lower().endswith(".pdf") and "magnet" in f.lower()]
    results = []

    for f in pdfs:
        path = os.path.join(base, f)
        with fitz.open(path) as doc:
            text = "".join(p.get_text() for p in doc)

        matches = re.findall(r"3\.\d{2,4}(?=°)", text)
        values = []
        for v in matches:
            try:
                values.append(float(v.strip("°")))
            except:
                continue

        if not values:
            print(f"[SKIP] No values from: {f}")
            continue

        mean = round(statistics.mean(values), 4)
        std = round(_safe_stdev(values), 4)
        # Spec limits (keep as-is; adjust here if your spec changes)
        lsl = 3.2
        usl = 4.2
        cp = (usl - lsl) / (6 * std) if std and std > 0 else 0.0
        cpk = min((mean - lsl) / (3 * std), (usl - mean) / (3 * std)) if std and std > 0 else 0.0

        date_match = re.search(r"(\d{8})", f)
        if date_match:
            dt = datetime.strptime(date_match.group(1), "%Y%m%d").date()
            results.append([dt, mean, std, cp, cpk])

    if not results:
        print("❌ No CpK data.")
        return

    # Sort by date so the "latest" annotation is correct
    results.sort(key=lambda r: r[0])

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    path = os.path.join(report_dir, filename)
    create_magnet_pdf(path, results)

# ---------- GUI ----------
def start_threaded(funcs):
    root = tk.Tk()
    root.title("Running...")
    root.geometry("400x100")
    tk.Label(root, text="Processing...", font=("Segoe UI", 12)).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(pady=5)
    bar.start()

    q = queue.Queue()
    cancel_flag = {"stop": False}

    def worker():
        for f in funcs:
            if cancel_flag.get("stop"):
                break
            f(q, cancel_flag)
        q.put("done")

    def check_done():
        try:
            if q.get_nowait() == "done":
                root.destroy()
        except queue.Empty:
            root.after(100, check_done)

    threading.Thread(target=worker, daemon=True).start()
    check_done()
    root.mainloop()

def main():
    root = tk.Tk()
    root.title("Rexair Analysis Tool")
    root.geometry("400x220")
    tk.Label(root, text="Motor & Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=10)
    tk.Button(root, text="Run Motor Analysis", width=40,
              command=lambda: [root.destroy(), start_threaded([run_motor_analysis])]).pack(pady=5)
    tk.Button(root, text="Run Magnet Analysis", width=40,
              command=lambda: [root.destroy(), start_threaded([run_magnet_analysis])]).pack(pady=5)
    tk.Button(root, text="Run Both Analyses", width=40,
              command=lambda: [root.destroy(), start_threaded([run_motor_analysis, run_magnet_analysis])]).pack(pady=10)
    tk.Button(root, text="Exit", width=20, command=root.destroy).pack(pady=5)
    root.mainloop()

if __name__ == "__main__":
    main()
