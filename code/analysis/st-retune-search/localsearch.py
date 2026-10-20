"""Local cost descent from the retuned set, with the sign terminal read and theta=2,
under the common model: each decomposition in turn is replaced by the cheapest
alternative that keeps the estimate below 2^-128 at every width."""
import json,sys,math
sys.argv=['x','2048x1','2','3']
exec(open('search2.py').read().split("PBS = decomps")[0])
PBS = decomps(range(6, 25), (1, 2, 3), 12, 62)
KS = decomps(range(1, 9), range(2, 31), 14, 40)
AUTO = decomps(range(4, 17), range(2, 11), 24, 62)
SS = decomps(range(8, 21), range(2, 6), 24, 62)
CBS = [(b, 4) for b in range(3, 9)]
start=json.load(open('../../code/server-campaign/parameters/st-cc2-r2-n688.json'))
start=dict(start,terminal_read="sign",terminal_lut_count_log=2)
best=(cost(start),start); print("start",round(best[0]/1e12,3),ok(start),flush=True)
improved=True
while improved:
    improved=False
    for field,opts in (("pbs",PBS),("ks",KS),("auto",AUTO),("ss",SS),("cbs",CBS)):
        cands=sorted((cost(dict(best[1],**{field:list(o)})),o) for o in opts)
        for c,o in cands:
            if c>=best[0]: break
            q=dict(best[1],**{field:list(o)})
            if ok(q,(256,)) and ok(q):
                best=(c,q); improved=True; print("improve",field,o,round(c/1e12,3),flush=True); break
q=best[1]
print(json.dumps({k:q[k] for k in ("lwe_dimension","lwe_sigma","pbs","ks","auto","ss","cbs","terminal_lut_count_log")}))
print([round(st_retune.screen(q,w)["conditional_log2_union"],2) for w in (16,32,64,128,256)])
for n,s in LWE:
    r=dict(q,lwe_dimension=n,lwe_sigma=s)
    print(n,round(math.log2(s),3),ok(r),round(cost(r)/1e12,3),flush=True)
json.dump(q,open('st-sign-local-best.json','w'),indent=1)
