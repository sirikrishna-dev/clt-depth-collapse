"""
Transparent reference-circuit intervention checks.
This validates the equation-level ground truth, NOT a trained Transformer's internal circuit.
"""
import json, os
import numpy as np

def ref_2hop(a,b,c):
    u=np.bitwise_xor(a,b); y=np.bitwise_and(u,c)
    return u,y

def ref_3hop(a,b,c,d):
    u=np.bitwise_xor(a,b); v=np.bitwise_and(u,c); y=np.bitwise_xor(v,d)
    return u,v,y

def main():
    rng=np.random.default_rng(11); n=10000
    a,b,c,d=[rng.integers(0,2,size=(n,16),dtype=np.int8) for _ in range(4)]
    u,y=ref_2hop(a,b,c)
    # Intervene by replacing u with a counterfactual value (bitwise flip).
    u_cf=1-u; y_cf=np.bitwise_and(u_cf,c)
    effect2=float(np.mean(y!=y_cf))
    u3,v3,y3=ref_3hop(a,b,c,d)
    v_cf=1-v3; y3_cf=np.bitwise_xor(v_cf,d)
    effect3=float(np.mean(y3!=y3_cf))
    report={
      "reference_circuit_checks":{
        "2hop":{"intermediate":"u","intervention":"flip u bits","fraction_output_bits_changed":effect2,"passed":effect2>0},
        "3hop":{"intermediate":"v","intervention":"flip v bits","fraction_output_bits_changed":effect3>0,"passed":effect3>0}
      },
      "warning":"These are reference-equation interventions only. They do not validate any trained Transformer's internal representations. Learned-model patching_status remains not_run until activation patching is implemented and run."
    }
    os.makedirs("results",exist_ok=True)
    with open("results/reference_intervention_report.json","w") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
