"""Re-encoding bootstrapping of the BFV adaptation: every decomposition (base, levels) with
20 <= base*levels <= 62, all other parameters fixed, under the common model. The set used in
the paper is the one with the fewest levels (the cost of the re-encoding grows with them) that
meets 2^-128 for W = 16 to 128. W = 256 misses for every decomposition."""
import json, sys
from pathlib import Path
BFV = Path(__file__).resolve().parents[2] / "code/server-campaign/adapters/bfv-style"
sys.path.insert(0, str(BFV))
import analyze_noise as noise
base = json.loads((BFV / "parameters/bfv-n800-range-aware.json").read_text())
rows = []
for b in range(1, 24):
    for l in range(1, 64):
        if not 20 <= b * l <= 62:
            continue
        p = dict(base, encoding_pbs=[b, l])
        u = [noise.screen(p, noise.make_plan(w, True, 8), True, False)["raw_log2_union"] for w in (16, 32, 64, 128, 256)]
        if max(u[:4]) < -128:
            rows.append((l, b, [round(x, 1) for x in u]))
rows.sort()
for r in rows[:6]:
    print("levels %d base %d: W=16..256 %s" % r)
