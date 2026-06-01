import os, re, csv, json, cv2, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


# Utils
def read_json(p):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def process_json(data):
    per_frame = {}

    for entry in data:
        # Columns normalization-
        x = entry["x_coord"]
        if isinstance(x, (list, tuple)):
            cols = [int(c) for c in x]  # multiple columns
        else:
            cols = [int(x)]  # one column

        signals = entry["signal"]
        if isinstance(signals, dict):
            frame_ids = signals.keys()
        elif isinstance(signals, (list, tuple)):
            frame_ids = []
            for item in signals:
                if isinstance(item, (list, tuple)):
                    frame_ids.append(item[0])
                else:
                    frame_ids.append(item)
        else:
            continue

        for f in frame_ids:
            try:
                frame = int(f)
            except Exception:
                continue

            for col in cols:
                per_frame.setdefault(frame, []).append(col)

    for f, lst in per_frame.items():
        per_frame[f] = np.array(lst, dtype=int)

    return per_frame


def frame_id_from_path(p):
    name = Path(p).stem
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def load_pair(gt_path, def_path, res_path):
    """Lit trois images en float32 (sans copie supplémentaire)."""
    gt = cv2.imread(gt_path, -1).astype(np.float32, copy=False)
    default = cv2.imread(def_path, -1).astype(np.float32, copy=False)
    res = cv2.imread(res_path, -1).astype(np.float32, copy=False)
    return gt, default, res


# Main function
def evaluate_sequence(args):
    """
    args = (sensor, seq_path, simulation_idx)

    Retourne un tuple contenant toutes les métriques et une chaîne de résumé.
    """
    sensor, seq_path, simulation_idx = args

    gt_path = os.path.join(seq_path, "low dyn")
    def_path = os.path.join(seq_path, f"low dyn with columns {simulation_idx}")
    res_path = os.path.join(def_path, "result")

    # Verifications
    if not (
        os.path.isdir(gt_path) and os.path.isdir(def_path) and os.path.isdir(res_path)
    ):
        return None

    # Read JSON
    json_file = next(
        (f for f in os.listdir(def_path) if f.lower().endswith(".json")), None
    )
    if json_file is None:
        return None

    defective_per_frame = process_json(read_json(os.path.join(def_path, json_file)))

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
            for f in os.listdir(def_path)
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

    if not (len(gt_files) == len(def_files) == len(res_files)):
        return None

    # Global parameters
    nb_cols = cv2.imread(gt_files[0], -1).shape[1]  # number of columns for on eimage

    # Agregates initialization
    tp, fp, fn = 0, 0, 0
    sq_def, cnt_def = 0.0, 0
    sq_ok, cnt_ok = 0.0, 0

    # Loop over frames
    for gt_f, def_f, res_f in zip(gt_files, def_files, res_files):
        gt, default, res = load_pair(gt_f, def_f, res_f)

        frame_id = frame_id_from_path(gt_f)
        if frame_id is None:
            frame_id = gt_files.index(gt_f)

        # defective columns
        true_def = defective_per_frame.get(frame_id, np.array([], dtype=int))
        true_mask = np.isin(np.arange(nb_cols), true_def)

        # Column by column detection
        residu_detect = ((res - default) ** 2).sum(axis=0)
        detected = np.flatnonzero(residu_detect > 1e-6)

        # Agregates
        tp += np.intersect1d(detected, true_def, assume_unique=True).size
        fp += np.setdiff1d(detected, true_def, assume_unique=True).size
        fn += np.setdiff1d(true_def, detected, assume_unique=True).size

        # RMSE
        residu_sq = ((res - gt) ** 2).sum(axis=0)

        sq_def += residu_sq[true_mask].sum()
        cnt_def += true_mask.sum() * gt.shape[0]

        sq_ok += residu_sq[~true_mask].sum()
        cnt_ok += (~true_mask).sum() * gt.shape[0]

    # Normalization
    nb_cols_f = float(nb_cols * len(gt_files))
    tp /= nb_cols_f
    fp /= nb_cols_f
    fn /= nb_cols_f

    # Scores
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    rmse_def = np.sqrt(sq_def / cnt_def) if cnt_def else 0.0
    rmse_ok = np.sqrt(sq_ok / cnt_ok) if cnt_ok else 0.0

    rmse_def_norm = 1.0 - min([rmse_def / 40, 1.0])
    rmse_ok_norm = 1.0 - min([rmse_ok / 40, 1.0])

    overall = 0.34 * f1 + 0.33 * rmse_def_norm + 0.33 * rmse_ok_norm

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


if __name__ == "__main__":
    root_path = r"train/"
    sensors = ["HD", "SXGA", "VGA"]
    csv_path = os.path.join(root_path, "results.csv")

    # Tasks construction
    tasks = []
    for sensor in sensors:
        sensor_path = os.path.join(root_path, sensor)
        for entry in os.scandir(sensor_path):
            if not entry.is_dir():
                continue
            seq = entry.path
            for sim in (1, 2, 3):
                tasks.append((sensor, seq, sim))

    # CSV writing
    results = []
    with ProcessPoolExecutor() as pool:
        for res in pool.map(evaluate_sequence, tasks):
            if res is not None:
                results.append(res)

    # CSV writing
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

        final_scores = []
        for r in results:
            print(r[-1])
            writer.writerow(r[:-1])
            final_scores.append(float(r[13]))

    # Scores display
    if final_scores:
        mean_score = np.mean(final_scores)
        print("\n=== SCORE FINAL MOYEN SUR TOUTES LES SÉQUENCES ===")
        print(f"Score moyen = {mean_score:.6f}")
    else:
        print("\nAucun résultat n’a été produit.")
