import json,sys,math
sys.argv=['x','1024x2','2','7']
exec(open('search2.py').read().split("PBS = decomps")[0])
AUTO = decomps(range(4, 17), range(2, 11), 24, 62)
SS = decomps(range(8, 21), range(2, 6), 24, 62)
CBS = [(b, 4) for b in range(3, 9)]
rest=[(a,s,c) for a in AUTO for s in SS for c in CBS]
found=[]
for n,sig in LWE:
    for pb in [(15,2),(11,3),(8,4),(23,1),(12,3),(10,4)]:
        for ks in [(2,7),(2,8),(3,5),(1,15),(2,9),(3,6)]:
            base=dict(name="t",lwe_dimension=n,polynomial_size=1024,glwe_dimension=2,lwe_sigma=sig,glwe_sigma=GLWE,
                      terminal_lut_count_log=2,split_trace_fft=True,cc2_big_key=True,terminal_read="sign",pbs=list(pb),ks=list(ks))
            r=min(rest,key=lambda r: st_retune.variances(dict(base,auto=list(r[0]),ss=list(r[1]),cbs=list(r[2])))["cmux"])
            q=dict(base,auto=list(r[0]),ss=list(r[1]),cbs=list(r[2]))
            u=max(st_retune.screen(q,w)["conditional_log2_union"] for w in (16,32,64,128,256))
            if u<=-128: found.append((cost(q),n,pb,ks,r,round(u,1)))
    print(n, "best so far", sorted(found)[:1], flush=True)
print(sorted(found)[:5])
