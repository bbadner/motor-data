import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from fpdf import FPDF

DATA_FOLDER = r"C:\Python Projects\Data Agent"
REPORT_FOLDER = r"C:\Python Projects\Data Agent\Magnet Reports"

os.makedirs(REPORT_FOLDER, exist_ok=True)

LSL = 3.50
USL = 3.90


# ===============================
# TEXT CLEANER (fix Unicode)
# ===============================

def clean_text(text):
    return text.encode("latin-1","ignore").decode("latin-1")


# ===============================
# EXTRACT MAGNET TIMING VALUES
# ===============================

def extract_values(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    # Extract ONLY magnet timing values
    numbers = re.findall(r"\b3\.\d{3}\b|\b3\.\d{2}\b", text)

    values = [float(x) for x in numbers]

    return values


# ===============================
# CPK CALCULATION
# ===============================

def calc_stats(values):

    mean = np.mean(values)

    std = np.std(values, ddof=1)

    cp = (USL - LSL) / (6 * std)

    cpk = min(
        (USL - mean) / (3 * std),
        (mean - LSL) / (3 * std)
    )

    return mean, std, cp, cpk


# ===============================
# CREATE CPK TREND CHART
# ===============================

def create_chart(results):

    labels = [r["label"] for r in results]

    cpks = [r["cpk"] for r in results]

    plt.figure(figsize=(10,5))

    plt.plot(labels, cpks, marker="o")

    plt.axhline(1.33, linestyle="--")

    plt.ylim(0,4)

    plt.ylabel("Cpk")

    plt.title("Magnet Timing CpK Over Time (Per File)")

    plt.xlabel("File within Month")

    plt.grid(True)

    status = "CpK is GOOD" if min(cpks) >= 1.33 else "CpK is BAD"

    color = "green" if min(cpks) >= 1.33 else "red"

    plt.text(len(labels)/2,2,status,color=color,fontsize=20,ha="center")

    chart = os.path.join(REPORT_FOLDER,"cpk_chart.png")

    plt.tight_layout()

    plt.savefig(chart)

    plt.close()

    return chart


# ===============================
# CREATE PDF REPORT
# ===============================

def create_pdf(results):

    chart = create_chart(results)

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial","B",20)

    pdf.cell(0,10,"Rexair Magnet Timing CpK Report",ln=True)

    pdf.ln(5)

    pdf.image(chart,x=10,w=190)

    pdf.ln(85)

    pdf.set_font("Arial","B",12)

    pdf.cell(0,10,"Per-File Summary Table",ln=True)

    pdf.set_font("Arial","",12)

    for r in results:

        line = (
            f"{clean_text(r['label'])} "
            f"Mean: {r['mean']:.4f}  "
            f"Std: {r['std']:.4f}  "
            f"Cp: {r['cp']:.2f}  "
            f"CpK: {r['cpk']:.2f}"
        )

        pdf.cell(0,8,line,ln=True)

    out = os.path.join(REPORT_FOLDER,"Rexair_Magnet_Timing_CpK_Report.pdf")

    pdf.output(out)

    print("\nReport created:")
    print(out)


# ===============================
# MAIN
# ===============================

def run():

    results = []

    for file in os.listdir(DATA_FOLDER):

        if not file.lower().endswith(".pdf"):
            continue

        if "rexair_magnet_timing_cpk_report" in file.lower():
            continue

        path = os.path.join(DATA_FOLDER,file)

        values = extract_values(path)

        if len(values) < 5:
            print("Skipping",file)
            continue

        mean,std,cp,cpk = calc_stats(values)

        label = clean_text(file.replace(".pdf",""))

        print(file,"Cpk =",round(cpk,2))

        results.append({

            "label": label,
            "values": values,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk

        })

    if not results:

        print("No magnet data found")

        return

    create_pdf(results)


if __name__ == "__main__":
    run()