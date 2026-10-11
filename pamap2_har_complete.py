#!/usr/bin/env python3
"""Standalone PAMAP2 human-activity-recognition pipeline for Google Colab.

Run:  !python /content/pamap2_har_complete.py

What it does
  * downloads the public PAMAP2 archive (no manual upload), extracts ONLY the
    Protocol/subject101..109.dat files (handles a ZIP nested inside the ZIP)
  * segments every subject into contiguous single-activity runs (no gaps, no
    activity changes), interpolates short sensor dropouts INSIDE each run only,
    then cuts 2 s windows (200 samples @100 Hz) with 50 % overlap
  * trains on subjects 101-106, tests on subjects 107-108 (109 excluded)
  * the best model is chosen by leave-subject-group-out CV on the TRAIN subjects
    only; the test subjects are never used to fit, tune or select anything
  * saves all outputs to <ROOT>/outputs
"""
import os, sys, time, shutil, zipfile, warnings, urllib.request
from pathlib import Path
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless: `!python` has no display
import matplotlib.pyplot as plt
import sklearn
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)

# ----------------------------------------------------------------- configuration
_default_root = "/content/pamap2_har" if Path("/content").exists() else "./pamap2_har"
ROOT = Path(os.environ.get("PAMAP2_ROOT", _default_root))
ZIP_PATH, DATA_DIR, OUT_DIR = ROOT/"pamap2.zip", ROOT/"dataset", ROOT/"outputs"
URLS = [
 "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
 "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
]
if os.environ.get("PAMAP2_URL"):           # optional override (mirror / testing)
    URLS.insert(0, os.environ["PAMAP2_URL"])

TRAIN_SUBJECTS = {101, 102, 103, 104, 105, 106}
TEST_SUBJECTS = {107, 108}
EXCLUDED_SUBJECTS = {109}
ALL_SUBJECTS = TRAIN_SUBJECTS | TEST_SUBJECTS | EXCLUDED_SUBJECTS
REQUIRED_SUBJECTS = TRAIN_SUBJECTS | TEST_SUBJECTS

FS = 100                         # PAMAP2 sampling rate (Hz)
WIN, STEP, SEED = 200, 100, 42   # 2 s windows, 50 % overlap
MAX_GAP_S = 0.05                 # timestamp jump (s) that splits a run (5 missing samples)
INTERP_LIMIT = 10                # fill NaN gaps of at most 10 samples (0.1 s)
MAX_NAN_FRAC = 0.10              # skip a window if a partly-missing channel exceeds this
CV_FOLDS = int(os.environ.get("PAMAP2_CV_FOLDS", "3"))                     # subject-grouped CV folds on the training subjects

ACT = {0:"Other",1:"Lying",2:"Sitting",3:"Standing",4:"Walking",5:"Running",
       6:"Cycling",7:"Nordic Walking",9:"Watching TV",10:"Computer Work",
       11:"Car Driving",12:"Ascending Stairs",13:"Descending Stairs",
       16:"Vacuum Cleaning",17:"Ironing",18:"Folding Laundry",19:"House Cleaning",
       20:"Playing Soccer",24:"Rope Jumping"}

_IMU = ["temperature","acc16_x","acc16_y","acc16_z","acc6_x","acc6_y","acc6_z",
        "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
        "orientation_1","orientation_2","orientation_3","orientation_4"]
COLS = ["timestamp","activity_id","heart_rate"] + [f"{loc}_{c}" for loc in ("hand","chest","ankle") for c in _IMU]
assert len(COLS) == 54             # official PAMAP2 .dat column count
SENSOR_COLS = [c for c in COLS if c not in {"timestamp","activity_id","heart_rate"} and "orientation" not in c]
STAT_NAMES = ["mean","std","min","max","median","iqr","rms"]
FEAT_NAMES = [f"{c}__{s}" for c in SENSOR_COLS for s in STAT_NAMES]
NEEDED_FILES = {f"subject{i}.dat" for i in ALL_SUBJECTS}

# --------------------------------------------------------- download / extraction
def protocol_files():
    d = DATA_DIR / "Protocol"
    return {sid: d/f"subject{sid}.dat" for sid in sorted(ALL_SUBJECTS)
            if (d/f"subject{sid}.dat").exists() and (d/f"subject{sid}.dat").stat().st_size > 0}

def download(url, dest, retries=3):
    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"[DOWNLOAD] {url} (attempt {attempt}/{retries})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            done, last = 0, time.time()
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                total = r.headers.get("Content-Length")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk: break
                    out.write(chunk); done += len(chunk)
                    if time.time() - last > 15:
                        print(f"  ... {done/1e6:.0f} MB"); last = time.time()
            if total and done != int(total):
                raise IOError(f"incomplete download: {done} of {total} bytes")
            if not zipfile.is_zipfile(tmp):
                raise RuntimeError("downloaded file is not a valid ZIP (error page or truncated)")
            tmp.replace(dest)
            print(f"[OK] Downloaded {done/1e6:.0f} MB")
            return True
        except Exception as e:
            print(f"[WARN] {type(e).__name__}: {e}")
            if tmp.exists(): tmp.unlink()
            time.sleep(2 * attempt)
    return False

def _extract_needed(zpath, out_dir, depth=0):
    """Extract Protocol/subject10x.dat from a ZIP, descending into nested ZIPs."""
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            p = Path(info.filename); base = p.name
            if base.lower().endswith(".zip") and not base.startswith("._") and depth < 2:
                nested = out_dir / f"_nested{depth}_{base}"
                with z.open(info) as src, open(nested, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                try: n += _extract_needed(nested, out_dir, depth + 1)
                finally: nested.unlink(missing_ok=True)
            elif base in NEEDED_FILES and "protocol" in [q.lower() for q in p.parts[:-1]]:
                target = out_dir / "Protocol" / base
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                with z.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(target); n += 1
    return n

def download_extract():
    for d in (ROOT, DATA_DIR, OUT_DIR): d.mkdir(parents=True, exist_ok=True)
    have = protocol_files()
    if REQUIRED_SUBJECTS <= set(have):
        print("[OK] Reusing extracted Protocol files:", sorted(have)); return
    for attempt in (1, 2):
        if not zipfile.is_zipfile(ZIP_PATH):
            if ZIP_PATH.exists(): ZIP_PATH.unlink()
            if not any(download(u, ZIP_PATH) for u in URLS):
                raise RuntimeError("Dataset download failed from all URLs. Check internet access and rerun.")
        print("[EXTRACT] Extracting Protocol/subject*.dat only ...")
        try:
            _extract_needed(ZIP_PATH, DATA_DIR); break
        except (zipfile.BadZipFile, OSError, EOFError) as e:
            print(f"[WARN] Extraction failed ({type(e).__name__}: {e}); discarding cached ZIP.")
            ZIP_PATH.unlink(missing_ok=True)
            if attempt == 2: raise
    have = protocol_files()
    missing = sorted(REQUIRED_SUBJECTS - set(have))
    if missing:
        raise FileNotFoundError(f"Protocol files missing after extraction for subjects {missing}; "
                                "archive layout may have changed.")
    print("[OK] Dataset ready:", sorted(have))

# ------------------------------------------------------------------ data loading
def read_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64, engine="c")
    if df.shape[1] != len(COLS):
        raise ValueError(f"{path.name}: {df.shape[1]} columns, expected {len(COLS)}")
    df.columns = COLS
    df = df[df.activity_id.notna()].copy()
    df["activity_id"] = df.activity_id.astype(int)
    # keep the ORIGINAL row index: it is used later to detect removed rows
    return df[df.activity_id.isin(list(ACT)) & (df.activity_id != 0)]

def segment_bounds(df):
    """Contiguous runs: same activity, consecutive original rows, no timestamp gap."""
    n = len(df)
    if n == 0: return np.array([], int), np.array([], int)
    y = df.activity_id.to_numpy()
    brk = y[1:] != y[:-1]                                   # activity change
    brk |= np.diff(df.index.to_numpy()) != 1                # rows removed (activity 0 etc.)
    d = np.diff(df.timestamp.to_numpy(dtype=np.float64))
    brk |= ~np.isfinite(d) | (d <= 0) | (d > MAX_GAP_S)     # dropout / non-monotonic time
    cuts = np.flatnonzero(brk) + 1
    return np.r_[0, cuts], np.r_[cuts, n]

def feature_vector(w):
    if not np.isnan(w).any():                    # fast path (almost all windows)
        mean = w.mean(axis=0); std = w.std(axis=0)
        mn = w.min(axis=0); mx = w.max(axis=0)
        q25, med, q75 = np.percentile(w, [25, 50, 75], axis=0)
        rms = np.sqrt((w * w).mean(axis=0))
    else:                                        # windows with fully-missing channels
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(w, axis=0); std = np.nanstd(w, axis=0)
            mn = np.nanmin(w, axis=0); mx = np.nanmax(w, axis=0)
            q25, med, q75 = np.nanpercentile(w, [25, 50, 75], axis=0)
            rms = np.sqrt(np.nanmean(w * w, axis=0))
    return np.stack([mean, std, mn, mx, med, q75 - q25, rms], axis=1).reshape(-1)  # channel-major

def make_windows(df, cols):
    stats = Counter()
    x = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = df.activity_id.to_numpy(dtype=np.int16)
    starts, ends = segment_bounds(df)
    Xw, yw = [], []
    for a, b in zip(starts, ends):
        stats["segments"] += 1
        if b - a < WIN:
            stats["segments_too_short"] += 1; continue
        # interpolate short gaps INSIDE this run only (never across activities/gaps)
        seg = pd.DataFrame(x[a:b]).interpolate(method="linear", limit=INTERP_LIMIT,
                                               limit_direction="both").to_numpy(dtype=np.float32)
        label = int(y[a])
        for pos in range(0, len(seg) - WIN + 1, STEP):
            w = seg[pos:pos + WIN]
            nf = np.isnan(w).mean(axis=0)
            if np.any((nf > MAX_NAN_FRAC) & (nf < 1.0)):
                stats["windows_skipped_missing"] += 1; continue
            Xw.append(feature_vector(w)); yw.append(label)
    return Xw, yw, stats

# ------------------------------------------------------------------- ML helpers
def make_imputer():
    try: return SimpleImputer(strategy="median", keep_empty_features=True)  # sklearn >= 1.2
    except TypeError: return SimpleImputer(strategy="median")

def build_models():
    return {
      "Random Forest": make_pipeline(make_imputer(), RandomForestClassifier(
          n_estimators=200, max_features="sqrt", class_weight="balanced_subsample",
          n_jobs=-1, random_state=SEED)),
      "Extra Trees": make_pipeline(make_imputer(), ExtraTreesClassifier(
          n_estimators=200, max_features="sqrt", class_weight="balanced",
          n_jobs=-1, random_state=SEED)),
      "Logistic Regression": make_pipeline(make_imputer(), StandardScaler(), LogisticRegression(
          max_iter=1000, class_weight="balanced", random_state=SEED)),
      "Linear SVM": make_pipeline(make_imputer(), StandardScaler(), LinearSVC(
          class_weight="balanced", random_state=SEED, dual=False, max_iter=5000)),
    }

def scores(y_true, y_pred):
    """Metrics over the classes actually present in y_true (absent classes are not scored)."""
    lab = np.unique(y_true)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"Accuracy": accuracy_score(y_true, y_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
                "Macro F1": f1_score(y_true, y_pred, labels=lab, average="macro", zero_division=0),
                "Weighted F1": f1_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0)}

def group_cv(model, X, y, groups):
    """Leave-subject-group-out CV on TRAIN subjects only (imputer/scaler refit per fold)."""
    k = min(CV_FOLDS, len(set(groups.tolist())))
    f1s = []
    for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        f1s.append(scores(y[va], m.predict(X[va]))["Macro F1"])
    return float(np.mean(f1s)), float(np.std(f1s))

# -------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    print("=" * 65, "\nPAMAP2 HUMAN ACTIVITY CLASSIFICATION\n" + "=" * 65)
    print(f"[ENV] numpy {np.__version__} | pandas {pd.__version__} | sklearn {sklearn.__version__}")
    download_extract(); files = protocol_files()
    Xtr, ytr, gtr, Xte, yte, gte = [], [], [], [], [], []
    for sid in sorted(files):
        if sid in EXCLUDED_SUBJECTS:
            print(f"[INFO] Excluding subject {sid}."); continue
        if sid not in REQUIRED_SUBJECTS: continue
        df = read_file(files[sid])
        xx, yy, st = make_windows(df, SENSOR_COLS)
        print(f"[LOAD] Subject {sid}: rows={len(df):,} runs={st['segments']} "
              f"(too short: {st['segments_too_short']}) windows={len(xx):,} "
              f"(skipped for missing data: {st['windows_skipped_missing']})")
        tgt = (Xtr, ytr, gtr) if sid in TRAIN_SUBJECTS else (Xte, yte, gte)
        tgt[0].extend(xx); tgt[1].extend(yy); tgt[2].extend([sid] * len(yy))
        del df
    if not Xtr or not Xte: raise RuntimeError(f"Empty windows: train={len(Xtr)}, test={len(Xte)}")
    Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr, dtype=np.int16); gtr = np.asarray(gtr)
    Xte = np.asarray(Xte, dtype=np.float32); yte = np.asarray(yte, dtype=np.int16); gte = np.asarray(gte)
    Xtr[~np.isfinite(Xtr)] = np.nan; Xte[~np.isfinite(Xte)] = np.nan
    assert Xtr.shape[1] == len(FEAT_NAMES) == Xte.shape[1]
    print(f"\n[DATA] Train={Xtr.shape}; Test={Xte.shape}; features={Xtr.shape[1]} "
          f"(channels={len(SENSOR_COLS)}, heart rate/orientation excluded)")
    print("[TRAIN CLASSES]", {ACT[int(k)]: int(v) for k, v in sorted(Counter(ytr.tolist()).items())})
    print("[TEST CLASSES] ", {ACT[int(k)]: int(v) for k, v in sorted(Counter(yte.tolist()).items())})
    absent = sorted(set(ytr.tolist()) - set(yte.tolist()))
    if absent: print("[INFO] Classes in train but absent from test (not scored):", [ACT[a] for a in absent])

    rows, fitted = [], {}
    for name, model in build_models().items():
        print(f"\n[CV] {name} ({CV_FOLDS}-fold, grouped by training subject)")
        st = time.time(); cv_mean, cv_std = group_cv(model, Xtr, ytr, gtr)
        print(f"  CV Macro F1 = {cv_mean:.4f} +/- {cv_std:.4f}")
        print(f"[TRAIN] {name} on all training subjects, evaluate on test subjects")
        model.fit(Xtr, ytr); pred = model.predict(Xte)
        vals = {"Model": name, "CV Macro F1 (train subjects)": cv_mean, "CV Macro F1 std": cv_std,
                **scores(yte, pred), "Seconds": time.time() - st}
        rows.append(vals); fitted[name] = model
        print("  TEST Accuracy={Accuracy:.4f} | Balanced Acc={Balanced Accuracy:.4f} | Macro F1={Macro F1:.4f}".format(**vals))
    result = pd.DataFrame(rows).sort_values("CV Macro F1 (train subjects)", ascending=False)
    result.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n[MODEL COMPARISON] (best model chosen by CV on TRAIN subjects, not by test score)\n",
          result.to_string(index=False))

    best_name = result.iloc[0]["Model"]; best = fitted[best_name]; pred = best.predict(Xte)
    final = scores(yte, pred)
    lab_true = sorted(set(yte.tolist()))
    report = classification_report(yte, pred, labels=lab_true,
                                   target_names=[ACT[i] for i in lab_true], zero_division=0)
    print(f"\n[BEST MODEL] {best_name}\n\n{report}")
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    lab_cm = sorted(set(yte.tolist()) | set(pred.tolist()))
    cm = confusion_matrix(yte, pred, labels=lab_cm)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=[ACT[i] for i in lab_cm]).plot(
        ax=ax, xticks_rotation=70, colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {best_name} (test subjects)"); fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    est = best[-1]
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({"Feature": FEAT_NAMES, "Importance": est.feature_importances_}
                           ).sort_values("Importance", ascending=False)
        imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
        top = imp.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 8)); ax.barh(top.Feature, top.Importance)
        ax.set_title("Top 20 Features - " + best_name); fig.tight_layout()
        fig.savefig(OUT_DIR / "top_features.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    joblib.dump({"model": best, "model_name": best_name,       # full pipeline incl. imputer/scaler
                 "imputer": best.steps[0][1], "sensor_columns": SENSOR_COLS,
                 "feature_names": FEAT_NAMES, "activity_names": ACT,
                 "window_size": WIN, "window_step": STEP, "sampling_rate_hz": FS,
                 "train_subjects": sorted(TRAIN_SUBJECTS), "test_subjects": sorted(TEST_SUBJECTS),
                 "excluded_subjects": sorted(EXCLUDED_SUBJECTS),
                 "sklearn_version": sklearn.__version__}, OUT_DIR / "best_pamap2_model.joblib")
    pd.DataFrame({"subject_id": gte, "true_activity_id": yte,
                  "true_activity": [ACT[int(v)] for v in yte],
                  "predicted_activity_id": pred,
                  "predicted_activity": [ACT[int(v)] for v in pred]}
                 ).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    (OUT_DIR / "run_summary.txt").write_text(
        f"Best model (chosen by CV on train subjects): {best_name}\n"
        f"Train windows: {len(ytr)}\nTest windows: {len(yte)}\nFeatures: {Xtr.shape[1]}\n"
        f"Test accuracy: {final['Accuracy']:.4f}\nTest balanced accuracy: {final['Balanced Accuracy']:.4f}\n"
        f"Test macro F1: {final['Macro F1']:.4f}\nTest weighted F1: {final['Weighted F1']:.4f}\n"
        f"Total minutes: {(time.time()-t0)/60:.1f}\n"
        "Train subjects 101-106; test subjects 107-108; subject 109 excluded.\n"
        "Activity 0, heart rate, and orientation features excluded.\n", encoding="utf-8")
    print("\n[OUTPUTS]")
    for p in sorted(OUT_DIR.iterdir()): print(" ", p)
    print(f"\n[DONE] Total time: {(time.time()-t0)/60:.1f} minutes. Outputs: {OUT_DIR}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit("Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        print("Check Colab internet access and available RAM, then rerun.", file=sys.stderr)
        raise
