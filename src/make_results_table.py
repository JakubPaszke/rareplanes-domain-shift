"""
Tabela zbiorcza wszystkich wynikow (results/per_size/*.json) -> markdown + CSV.
Wymog PDF: tabela metryk z punktem odniesienia i wariantami porownawczymi.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "per_size"

# kolejnosc i etykiety (czytelne nazwy do raportu)
ORDER = [
    ("real_baseline",          "Baseline A: real->real (gorny ref)"),
    ("syn_to_real_baseline",   "Baseline B: synth 6460 -> real"),
    ("syn45k_to_real_baseline","Baseline B: synth 45k -> real"),
    ("expA1_weak_10k",         "Eksp A: slaby HSV (10k)"),
    ("expA2_med_10k",          "Eksp A: sredni HSV (10k)"),
    ("expA3_strong_10k",       "Eksp A: mocny HSV (10k)"),
    ("expA_final_45k",         "Eksp A: slaby HSV (45k, finalny)"),
    ("yolo11l_10k_ml",         "Arch: YOLO11l CNN duzy (10k)"),
    ("rtdetr_l_10k_ml",        "Arch: RT-DETR-l Transf maly (10k)"),
    ("rtdetrx_10k_ml",         "Arch: RT-DETR-x Transf duzy (10k)"),
    ("expD_320_10k_ml",        "Eksp D: imgsz 320 (10k)"),
    ("expD_768_10k_ml",        "Eksp D: imgsz 768 (10k)"),
    ("expD_1024_10k_ml",       "Eksp D: imgsz 1024 (10k)"),
]
COLS = ["AP@.5", "AP@[.5:.95]", "AP_small", "AP_medium", "AP_large", "AR@100"]
HDR = ["mAP@50", "mAP@50:95", "AP_S", "AP_M", "AP_L", "AR@100"]


def main():
    rows = []
    for key, label in ORDER:
        p = SRC / f"{key}.json"
        if not p.exists():
            continue
        m = json.load(open(p))["metrics"]
        rows.append((label, [m.get(c) for c in COLS]))

    # markdown
    md = ["| Wariant | " + " | ".join(HDR) + " |",
          "|" + "---|" * (len(HDR) + 1)]
    for label, vals in rows:
        md.append("| " + label + " | " +
                  " | ".join(f"{v:.3f}" if v is not None else "—" for v in vals) + " |")
    md_txt = "\n".join(md)
    (ROOT / "results" / "tabela_zbiorcza.md").write_text(
        md_txt + "\n\n*RT-DETR z best.pt po 2 epokach (trening rozbiezny, patrz notes/07)\n")

    # csv
    csv = [",".join(["wariant"] + HDR)]
    for label, vals in rows:
        csv.append(",".join([label] + [f"{v:.4f}" if v is not None else "" for v in vals]))
    (ROOT / "results" / "tabela_zbiorcza.csv").write_text("\n".join(csv) + "\n")

    print(md_txt)
    print(f"\n[zapisano] results/tabela_zbiorcza.{{md,csv}}")


if __name__ == "__main__":
    main()
