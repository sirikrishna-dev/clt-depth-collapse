import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def load_split(path, task, split):
    df=pd.read_csv(path)
    df=df[(df.task==task)&(df.split==split)]
    bits=16
    names=["a","b","c"] if task=="2hop" else ["a","b","c","d"]
    X=np.stack([df[[f"{n}_{i:02d}" for i in range(bits)]].to_numpy(dtype=np.float32) for n in names],axis=1)
    Y=df[[f"y_{i:02d}" for i in range(bits)]].to_numpy(dtype=np.float32)
    return torch.tensor(X),torch.tensor(Y)

class BitVectorTransformer(nn.Module):
    def __init__(self,n_inputs,layers,d_model=64,nhead=4):
        super().__init__()
        self.input_proj=nn.Linear(n_inputs,d_model)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=128,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=layers)
        self.head=nn.Linear(d_model,1)
    def forward(self,x):
        # x shape [batch, operands, bits]. Treat each bit position as a token;
        # each token has the operand bits as its features.
        x=x.transpose(1,2)
        h=self.input_proj(x)
        h=self.encoder(h)
        return self.head(h).squeeze(-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/boolean_combined.csv")
    ap.add_argument("--task",choices=["2hop","3hop"],default="2hop")
    ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--out",default="checkpoints")
    args=ap.parse_args()
    if args.layers not in [2,3,4,5,6]: raise ValueError("layers must be between 2 and 6")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device="mps" if torch.backends.mps.is_available() else "cpu"
    trainX,trainY=load_split(args.data,args.task,"train")
    valX,valY=load_split(args.data,args.task,"validation")
    testX,testY=load_split(args.data,args.task,"test")
    model=BitVectorTransformer(trainX.shape[1],args.layers).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss_fn=nn.BCEWithLogitsLoss()
    loader=DataLoader(TensorDataset(trainX,trainY),batch_size=args.batch_size,shuffle=True)
    best_val=float("inf"); best_state=None
    for epoch in range(args.epochs):
        model.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits=model(valX.to(device)); val_loss=loss_fn(logits,valY.to(device)).item()
        if val_loss<best_val:
            best_val=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"epoch={epoch+1} val_bce={val_loss:.5f}")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits=model(testX.to(device)).cpu()
        preds=(torch.sigmoid(logits)>=0.5).float()
        bit_acc=(preds==testY).float().mean().item()
        exact_acc=(preds==testY).all(dim=1).float().mean().item()
    os.makedirs(args.out,exist_ok=True)
    model_id=f"{args.task}_L{args.layers}_seed{args.seed}"
    ckpt=os.path.join(args.out,model_id+".pt")
    torch.save({"state_dict":model.state_dict(),"task":args.task,"layers":args.layers,"seed":args.seed,"n_bits":16},ckpt)
    meta={
      "model_id":model_id,"source":"hand_built_transformer","task":args.task,
      "hop_depth":2 if args.task=="2hop" else 3,"layers":args.layers,"seed":args.seed,
      "checkpoint":ckpt,"best_validation_bce":best_val,"test_bit_accuracy":bit_acc,
      "test_exact_vector_accuracy":exact_acc,
      "intermediates":{"u":{"definition":"XOR(a,b)","location":"not_identified_yet"}},
      "patching_status":"not_run","validated":False
    }
    if args.task=="3hop": meta["intermediates"]["v"]={"definition":"AND(u,c)","location":"not_identified_yet"}
    os.makedirs("metadata",exist_ok=True)
    with open(os.path.join("metadata",model_id+".json"),"w") as f: json.dump(meta,f,indent=2)
    with open(os.path.join("metadata","model_zoo.jsonl"),"a") as f: f.write(json.dumps(meta)+"\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
