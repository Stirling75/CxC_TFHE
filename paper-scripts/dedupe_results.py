"""Replace copies of the noise analysis and the schedule that repeat across the
thread budgets of one (method, width) by pointers to the run with the fewest
threads in the same campaign directory. Run once from paper-scripts/."""
import json, re, hashlib, os
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "results"
pat = re.compile(r"^(?P<m>.+)-w(?P<w>\d+)-t(?P<t>\d+)$")
saved = 0
groups = {}
for run in RES.rglob("run.json"):
    d = run.parent; mm = pat.match(d.name)
    if not mm: continue
    groups.setdefault((d.parent, mm["m"], mm["w"]), []).append((int(mm["t"]), d))
for (camp, m, w), runs in groups.items():
    runs.sort()
    base = runs[0][1]
    bj = json.loads((base / "run.json").read_text())
    bres = bj.get("resolved", {})
    bplan = base / "raw" / "plan.json"
    bplan_txt = bplan.read_text() if bplan.exists() else None
    for _, d in runs[1:]:
        rj = d / "run.json"; j = json.loads(rj.read_text()); res = j.get("resolved", {})
        changed = False
        for k, v in list(res.items()):
            if k in bres and len(json.dumps(v)) > 10000 and v == bres[k]:
                res[k] = {"same_as": f"../{base.name}/run.json", "key": f"resolved.{k}"}
                changed = True
        if changed:
            old = rj.stat().st_size; rj.write_text(json.dumps(j, indent=1)); saved += old - rj.stat().st_size
        p = d / "raw" / "plan.json"
        if bplan_txt is not None and p.exists() and p.read_text() == bplan_txt:
            old = p.stat().st_size
            p.write_text(json.dumps({"same_as": f"../../{base.name}/raw/plan.json"}) + "\n")
            saved += old - p.stat().st_size
print(f"{len(groups)} groups, saved {saved/2**20:.1f} MiB")

# Second pass: identical schedules across campaigns, and the per-campaign
# source manifests, which repeat for every campaign built from the same source.
seen = {}
for name in ("plan.json", "schedule.json"):
    for p in sorted(RES.rglob(name)):
        txt = p.read_bytes()
        if len(txt) < 10000: continue
        h = hashlib.sha256(txt).hexdigest()
        if h in seen:
            old = len(txt)
            p.write_text(json.dumps({"same_as": os.path.relpath(seen[h], p.parent)}) + "\n")
            saved += old - p.stat().st_size
        else:
            seen[h] = p
man = RES / "source-manifests"; man.mkdir(exist_ok=True)
for c in sorted(RES.rglob("campaign.json")):
    j = json.loads(c.read_text())
    if isinstance(j.get("source_sha256"), dict):
        sid = j.get("source_id") or hashlib.sha256(json.dumps(j["source_sha256"], sort_keys=True).encode()).hexdigest()[:16]
        f = man / f"{sid}.json"
        if not f.exists(): f.write_text(json.dumps(j["source_sha256"], indent=1))
        old = c.stat().st_size
        j["source_sha256"] = {"same_as": os.path.relpath(f, c.parent)}
        c.write_text(json.dumps(j, indent=1)); saved += old - c.stat().st_size
print(f"total saved {saved/2**20:.1f} MiB")
