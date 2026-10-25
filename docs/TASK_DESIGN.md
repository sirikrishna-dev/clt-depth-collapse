# Task design and ground truth

## Task equations
Use bitwise vector operations over 16-bit vectors:
- 2-hop: `u = XOR(a,b)`; `y = AND(u,c)`.
- 3-hop: `u = XOR(a,b)`; `v = AND(u,c)`; `y = XOR(v,d)`.

The 3-hop task extends the 2-hop task by adding a final operation. This is better controlled than choosing unrelated equations for the two depths.

## Why vector-valued inputs?
A 16-bit vector per operand creates a large input space, so a small truth table cannot enumerate the full space. The generator samples unique tuples for train/validation/test within each task.

## Held-out compositions: important limitation and next step
The generated CSVs hold out input examples, not whole expression compositions. To claim compositional generalization, you must also define a family of expressions from a shared operator vocabulary (e.g., XOR, AND, OR), reserve selected operator/parenthesization compositions from training, and evaluate them as separate task IDs. Do not call ordinary random row splits "held-out compositions".

## Known intermediate variables
Every row includes `u`, and 3-hop rows include `v`. These are ground-truth labels from the equations. They do not prove a learned model computes them internally.

## Avoiding shortcuts
XOR is associative, and Boolean expressions can have equivalent implementations. A model can produce correct outputs without using the named intermediate variables. Therefore:
1. Measure held-out performance.
2. Use causal interventions on a candidate internal representation.
3. Predefine a pass/fail criterion (for example, intervention causes a statistically reliable output change in the expected direction).
4. Record intervention results and do not mark failed or untested models as validated.
