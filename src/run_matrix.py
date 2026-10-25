import argparse, subprocess, sys, itertools, os
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="0,1,2,3,4")
    p.add_argument("--tasks",default="2hop,3hop")
    p.add_argument("--layers",default="2,4,6")
    p.add_argument("--epochs",type=int,default=20)
    a=p.parse_args()
    seeds=[int(x) for x in a.seeds.split(",")]
    tasks=a.tasks.split(","); layers=[int(x) for x in a.layers.split(",")]
    os.makedirs("logs",exist_ok=True)
    for task,layer,seed in itertools.product(tasks,layers,seeds):
        cmd=[sys.executable,"src/train_transformer.py","--task",task,"--layers",str(layer),"--seed",str(seed),"--epochs",str(a.epochs)]
        log=f"logs/{task}_L{layer}_seed{seed}.log"
        print("RUN", " ".join(cmd), "->",log,flush=True)
        with open(log,"w") as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode: print(f"FAILED: {log}; continuing",flush=True)
if __name__=="__main__": main()
