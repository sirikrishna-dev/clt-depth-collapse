import argparse, os, json
import numpy as np
import pandas as pd

BITS = 16
SPLITS = {"train": 5000, "validation": 1000, "test": 1000}
SEED = 20261025

def op_xor(a, b): return np.bitwise_xor(a, b)
def op_and(a, b): return np.bitwise_and(a, b)
def op_or(a, b): return np.bitwise_or(a, b)

def generate_task(task, n_rows, rng):
    depth = 2 if task == "2hop" else 3
    n_inputs = 3 if depth == 2 else 4
    seen, rows = set(), []
    for split, n in n_rows.items():
        made = 0
        while made < n:
            batch = rng.integers(0, 2, size=(max(256, (n-made)*2), n_inputs, BITS), dtype=np.int8)
            for x in batch:
                key = x.tobytes()
                if key in seen: continue
                seen.add(key)
                a,b,c = x[0],x[1],x[2]
                u = op_xor(a,b)
                if depth == 2:
                    y = op_and(u,c); v = None
                else:
                    d=x[3]; v=op_and(u,c); y=op_xor(v,d)
                r={"task":task,"split":split,"hop_depth":depth}
                for name,vec in zip(["a","b","c","d"],x):
                    for j,val in enumerate(vec): r[f"{name}_{j:02d}"]=int(val)
                for j,val in enumerate(u): r[f"u_{j:02d}"]=int(val)
                for j in range(BITS): r[f"v_{j:02d}"]="" if v is None else int(v[j])
                for j,val in enumerate(y): r[f"y_{j:02d}"]=int(val)
                rows.append(r); made += 1
                if made == n: break
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="data")
    ap.add_argument("--rows-train",type=int,default=5000)
    ap.add_argument("--rows-validation",type=int,default=1000)
    ap.add_argument("--rows-test",type=int,default=1000)
    ap.add_argument("--bits",type=int,default=16)
    args=ap.parse_args()
    global BITS
    BITS=args.bits
    sizes={"train":args.rows_train,"validation":args.rows_validation,"test":args.rows_test}
    os.makedirs(args.out,exist_ok=True)
    rng=np.random.default_rng(SEED)
    frames=[]
    for task in ["2hop","3hop"]:
        df=generate_task(task,sizes,rng); frames.append(df)
        df.to_csv(os.path.join(args.out,f"boolean_{task}.csv"),index=False)
    combined=pd.concat(frames,ignore_index=True)
    combined.to_csv(os.path.join(args.out,"boolean_combined.csv"),index=False)
    metadata={
      "seed":SEED,"bits_per_operand":BITS,"split_sizes_per_task":sizes,
      "tasks":{
        "2hop":{"equations":["u = XOR(a,b)","y = AND(u,c)"],"hop_depth":2,"intermediates":["u"]},
        "3hop":{"equations":["u = XOR(a,b)","v = AND(u,c)","y = XOR(v,d)"],"hop_depth":3,"intermediates":["u","v"]}
      },
      "note":"Random input-example splits; these are not held-out expression-composition splits."
    }
    with open(os.path.join(args.out,"task_metadata.json"),"w") as f: json.dump(metadata,f,indent=2)
    print(f"Wrote {len(combined):,} combined rows to {args.out}/")
if __name__=="__main__": main()
