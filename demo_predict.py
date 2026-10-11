#!/usr/bin/env python3
"""Live demo: classify real 2-second sensor windows with the trained model.

Run AFTER pamap2_har_complete.py has finished (needs the saved model + dataset):
    python demo_predict.py            # 3 random windows per activity from test subject 107
    python demo_predict.py 108 5      # subject 108, 5 windows per activity
"""
import sys, random
import numpy as np
import joblib
import pamap2_har_complete as har          # reuse the exact same preprocessing/features

subject = int(sys.argv[1]) if len(sys.argv) > 1 else 107
per_act = int(sys.argv[2]) if len(sys.argv) > 2 else 3
if subject in har.TRAIN_SUBJECTS:
    print(f"[NOTE] subject {subject} was used for TRAINING; use 107 or 108 for an honest demo.")

bundle = joblib.load(har.OUT_DIR / "best_pamap2_model.joblib")
model, names = bundle["model"], bundle["activity_names"]
print(f"Loaded model: {bundle['model_name']}  | window = {bundle['window_size']} samples "
      f"({bundle['window_size']/bundle['sampling_rate_hz']:.0f} s @ {bundle['sampling_rate_hz']} Hz)")

df = har.read_file(har.protocol_files()[subject])
starts, ends = har.segment_bounds(df)
x = df[har.SENSOR_COLS].to_numpy(dtype=np.float32)
y = df.activity_id.to_numpy()
rng = random.Random(0)
by_act = {}
for a, b in zip(starts, ends):
    if b - a >= har.WIN: by_act.setdefault(int(y[a]), []).append((a, b))

correct = total = 0
print(f"\n{'True activity':<20}{'Predicted':<20}Result")
for act in sorted(by_act):
    for _ in range(per_act):
        a, b = rng.choice(by_act[act]); pos = rng.randrange(a, b - har.WIN + 1)
        w = x[pos:pos + har.WIN]
        pred = int(model.predict(har.feature_vector(w).reshape(1, -1))[0])
        ok = pred == act; correct += ok; total += 1
        print(f"{names[act]:<20}{names[pred]:<20}{'OK' if ok else 'WRONG'}")
print(f"\nDemo accuracy on {total} sampled windows: {correct/total:.1%}")
