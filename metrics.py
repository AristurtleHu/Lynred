import os, csv, json, cv2, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# -------------------------------------------------
# Utils
# -------------------------------------------------
def read_json(p):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf‑8"))


def process_json(data):
    # Returns a boolean mask (True if defective column)
    indices = []
    for c in data:
        indices += c["x_coord"]
    return np.array(indices, dtype=int)


def load_pair(gt_path, def_path, res_path):
    """Lit deux images en float32 (sans copie supplémentaire)."""
    # -1 = IMREAD_UNCHANGED for 16bits images
    gt = cv2.imread(gt_path, -1).astype(np.float32, copy=False)
    default = cv2.imread(def_path, -1).astype(np.float32, copy=False)
    res = cv2.imread(res_path, -1).astype(np.float32, copy=False)
    return gt, default, res


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate_sequence(args):
    sensor, seq_path, simulation_idx = args
    gt_path = os.path.join(seq_path, "low dyn")
    def_path = os.path.join(seq_path, f"low dyn with columns {simulation_idx}")
    res_path = os.path.join(def_path, "result")

    # Verification
    if not (
        os.path.isdir(gt_path) and os.path.isdir(def_path) and os.path.isdir(res_path)
    ):
        return None

    # JSON for defective columns information
    json_file = next(
        (f for f in os.listdir(def_path) if f.lower().endswith(".json")), None
    )
    if json_file is None:
        return None
    true_def = process_json(read_json(os.path.join(def_path, json_file)))
    nb_cols = cv2.imread(
        sorted(
            [
                os.path.join(gt_path, f)
                for f in os.listdir(gt_path)
                if f.lower().endswith(".png")
            ]
        )[0],
        -1,
    ).shape[1]
    true_mask = np.isin(np.arange(nb_cols), true_def)  # booléen 1‑D

    # List of files
    gt_files = sorted(
        [
            os.path.join(gt_path, f)
            for f in os.listdir(gt_path)
            if f.lower().endswith(".png")
        ]
    )
    def_files = sorted(
        [
            os.path.join(def_path, f)
            for f in os.listdir(gt_path)
            if f.lower().endswith(".png")
        ]
    )
    res_files = sorted(
        [
            os.path.join(res_path, f)
            for f in os.listdir(res_path)
            if f.lower().endswith(".png")
        ]
    )
    if len(gt_files) != len(res_files):
        return None

    # Agregates
    tp, fp, fn = 0, 0, 0
    sq_def, cnt_def = 0.0, 0
    sq_ok, cnt_ok = 0.0, 0

    # Parallel reading
    with ThreadPoolExecutor(max_workers=4) as pool:
        for gt_f, def_f, res_f in zip(gt_files, def_files, res_files):
            gt, default, res = pool.submit(load_pair, gt_f, def_f, res_f).result()

            # Residuals by column
            residu_detect = ((res - default) ** 2).sum(axis=0)
            detected = np.flatnonzero(residu_detect)

            # TP / FP / FN
            tp += np.intersect1d(detected, true_def, assume_unique=True).size
            fp += np.setdiff1d(detected, true_def, assume_unique=True).size
            fn += np.setdiff1d(true_def, detected, assume_unique=True).size

            residu_sq = ((res - gt) ** 2).sum(axis=0)
            # Separated RMSE
            mask_def = true_mask
            mask_ok = ~true_mask

            sq_def += (residu_sq[mask_def]).sum()
            cnt_def += mask_def.sum() * gt.shape[0]

            sq_ok += (residu_sq[mask_ok]).sum()
            cnt_ok += mask_ok.sum() * gt.shape[0]

    # Normalization by number of columns
    nb_cols_f = float(nb_cols * len(gt_files))
    tp /= nb_cols_f
    fp /= nb_cols_f
    fn /= nb_cols_f

    # Metrics
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rmse_def = np.sqrt(sq_def / cnt_def) if cnt_def else 0.0
    rmse_ok = np.sqrt(sq_ok / cnt_ok) if cnt_ok else 0.0

    rmse_def_norm = 1.0 - min([rmse_def / 40, 1.0])
    rmse_ok_norm = 1.0 - min([rmse_ok / 40, 1.0])
    overall = 0.34 * f1 + 0.33 * (rmse_def_norm) + 0.33 * (rmse_ok_norm)

    # Tuple results
    return (
        sensor,
        os.path.basename(seq_path),
        def_path,
        f"{tp:.6f}",
        f"{fp:.6f}",
        f"{fn:.6f}",
        f"{prec:.6f}",
        f"{rec:.6f}",
        f"{f1:.6f}",
        f"{rmse_def:.6f}",
        f"{rmse_ok:.6f}",
        f"{rmse_def_norm:.6f}",
        f"{rmse_ok_norm:.6f}",
        f"{overall:.6f}",
        f"[{sensor}/{os.path.basename(seq_path)} {simulation_idx}] "
        f"TP={tp:.3f} FP={fp:.3f} FN={fn:.3f} "
        f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} "
        f"RMSE_def={rmse_def:.2f} RMSE_ok={rmse_ok:.2f} "
        f"RMSE_def_norm={rmse_def_norm:.2f} RMSE_ok_norm={rmse_ok_norm:.2f} "
        f"Score final : {overall:.4f}",
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    root_path = r"train"
    sensors = ["HD", "SXGA", "VGA"]
    csv_path = os.path.join(root_path, "results.csv")

    # List of final scores
    final_scores = []

    # CSV initialization
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sensor",
                "sequence",
                "def_path",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "F1",
                "RMSE_def",
                "RMSE_ok",
                "RMSE_def_norm",
                "RMSE_ok_norm",
                "final_score",
            ]
        )

        # For each sequence and simulation
        tasks = []
        for sensor in sensors:
            sensor_path = os.path.join(root_path, sensor)
            for entry in os.scandir(sensor_path):
                if not entry.is_dir():
                    continue
                seq = entry.path
                for sim in (1, 2, 3):
                    tasks.append((sensor, seq, sim))

        # parallele execution
        with ProcessPoolExecutor() as pool:
            for result in pool.map(evaluate_sequence, tasks):
                if result is None:
                    continue

                # Console print
                print(result[-1])

                # CSV writing
                writer.writerow(result[:-1])

                # Final score
                final_scores.append(float(result[13]))

    # -------------------------------------------------
    # Final score print
    # -------------------------------------------------
    if final_scores:
        mean_score = np.mean(final_scores)
        print("\n=== SCORE FINAL MOYEN SUR TOUTES LES SÉQUENCES ===")
        print(f"Score moyen = {mean_score:.6f}")
    else:
        print("\nAucun résultat n’a été produit.")
