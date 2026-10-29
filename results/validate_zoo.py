"""Read results/das_siit_seed*.csv, mark each model validated (or not) in its metadata JSON,
and rebuild metadata/model_zoo.jsonl without duplicate model_ids.

Run from the repo root:  python validate_zoo.py

Pass rule (decoupled test, layer 0, k=1): u >= 0.95 and both a and b <= 0.6.
Change THRESH_* below if your team agrees different cutoffs.
"""
import glob, json, os
import pandas as pd

THRESH_U, THRESH_DECOY = 0.95, 0.60

for csv in sorted(glob.glob("results/das_siit_seed*.csv")):
    df = pd.read_csv(csv)
    ckpt = df["ckpt"].iloc[0]
    model_id = os.path.basename(ckpt).replace(".pt", "")
    meta_path = os.path.join("metadata", model_id + ".json")
    if not os.path.exists(meta_path):
        print("no metadata for", model_id); continue
    k1 = df[(df.layer == 0) & (df.k == 1)].set_index("hyp")["acc_flip"]
    ok = bool(k1["u"] >= THRESH_U and k1["a"] <= THRESH_DECOY and k1["b"] <= THRESH_DECOY)
    meta = json.load(open(meta_path))
    meta["validated"] = ok
    meta["das_decoupled_k1"] = {h: round(float(k1[h]), 3) for h in k1.index}
    meta.pop("validation_todo", None)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"{model_id}: u={k1['u']:.3f} a={k1['a']:.3f} b={k1['b']:.3f} -> validated={ok}")

# rebuild the jsonl: one line per model_id, latest metadata file wins
rows = {}
for p in sorted(glob.glob("metadata/*.json")):
    if os.path.basename(p).startswith("model_zoo"):  # skip the schema file
        continue
    m = json.load(open(p))
    if isinstance(m, dict) and "model_id" in m:
        rows[m["model_id"]] = m
with open("metadata/model_zoo.jsonl", "w") as f:
    for m in rows.values():
        f.write(json.dumps(m) + "\n")
print("rewrote metadata/model_zoo.jsonl with", len(rows), "models")