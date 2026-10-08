"""Derive a fixed public schedule; fused variances are diagnostic references only."""
import hashlib
import math
from pathlib import Path

import verify_constructions as c
import fused_model

def parameter_environment(mode, squaring, capacity=24, cbs_level=7, kernel="tree", auto_fft="vanilla", chunk_bits=8):
    if auto_fft not in ("vanilla", "split40"):
        raise ValueError("unknown automorphism FFT backend")
    common = c.base.CONFIG["hybrid_common"]
    profile = c.base.CONFIG[f"hybrid_{chunk_bits}x{chunk_bits}"]
    env = {
        "CBS_FAILURE_PRESET": f"hybrid-{chunk_bits}x{chunk_bits}", "CBS_PARAM_PROFILE": "m2c2-gaussian",
        "CBS_CACHED_PRODUCT": str(int(mode == "cached")),
        "CBS_SHARED_PREPROCESSING": str(int(mode in ("cached", "fused"))),
        "FUSED_PRODUCTS_PER_GROUP": str(capacity if mode == "fused" else 0),
        "FUSED_KERNEL": kernel,
        "FUSED_EXPLORATORY": "1",
        "CACHED_MULT_SQUARING": str(int(squaring)),
        "CBS_CENTERED_MS": "1", "CBS_CENTER_SELECTOR_BOX": "1",
        "CBS_NORMALIZER_CENTERED_MS": "1", "CBS_NORMALIZER_CHUNK_CAP": "15",
        "DIRECT_LOG_LUT_COUNT": "2", "CBS_PARALLEL_DIGIT_LIFT": "1",
        "CBS_PARALLEL_CMUX_CELLS": "1",
        "DIRECT_AUTO_FFT": auto_fft,
    }
    for source, key, prefix in (
        (common,"normalizer_pbs","PBS"),
        (common,"normalizer_key_switch","KS"),
        (common,"automorphism","DIRECT_AUTO"),
        (common,"scheme_switch","DIRECT_SS"),
        (profile,"circuit_bootstrap","DIRECT_CBS"),
        (profile,"selector_lift_pbs","CBS_LIFT_PBS"),
        (profile,"selector_lift_key_switch","CBS_LIFT_KS"),
    ):
        env[prefix+"_BASE_LOG"], env[prefix+"_LEVEL"] = map(str, source[key])
    if mode == "fused":
        env["DIRECT_CBS_LEVEL"] = str(cbs_level)
    return env

def metadata(term, noise_audit=False):
    out = {"bound": term.bound, "refreshed": term.refreshed,
           "source": getattr(term, "routing", None)}
    if noise_audit:
        out["variance"] = c.est.term_var_source_aware(term)
    return out

def derive(width, mode, squaring=False, noise_audit=False, capacity=24, cbs_level=7, kernel="tree", auto_fft="vanilla", chunk_bits=8,
           public_scalar=None, public_cap=64):
    assert width in c.base.WIDTHS and mode in ("baseline","cached","fused")
    assert not squaring
    assert kernel in ("tree", "reuse", "reuse-cache")
    assert chunk_bits in (4,8) and (chunk_bits==8 or mode=="baseline")
    p,v = fused_model.primitive(cbs_level) if mode == "fused" else c.base.primitive_for(chunk_bits)
    public_groups = None
    if public_scalar is not None:
        assert mode == "fused" and len(public_scalar) == width//2
        public_groups = fused_model.public_groups(public_scalar, public_cap)
    columns = (fused_model.public_columns(public_scalar,public_groups,v) if public_groups is not None else
               fused_model.columns(width,capacity,v) if mode == "fused" else
               c.cached_columns(width,squaring)[0] if mode == "cached" else
               c.est.product_terms(width//2,v,contract=c.base.CONTRACTS[chunk_bits],identify_operands=squaring))
    h,d = width//8,width//2
    if mode != "fused":
        positions = [0]*len(columns)
        for u in range(h):
            for r in range(h-u):
                offsets = ((0,0),) if chunk_bits==8 else ((0,0),(0,2),(2,0),(2,2))
                for a,b in offsets:
                    if 4*(u+r)+a+b>=d:
                        continue
                    for t in range(chunk_bits):
                        q = 4*(u+r)+a+b+t
                        columns[q][positions[q]].routing = [4*u+a,4*r+b,t]
                        positions[q] += 1
    policy = "source-balanced" if mode in ("cached", "fused") else "baseline"
    reference = c.base.simulate(columns,d,p,v,policy=policy)
    # Keep the old rejection screen, but never treat passing it as approval.
    if not (math.isfinite(reference.union_log2_pfail) and reference.union_log2_pfail < -128):
        raise ValueError(
            f"legacy reference screen rejected the fixed public partition for W={width}, "
            f"mode={mode}, chunk_bits={chunk_bits}, capacity={capacity}: "
            f"log2 union {reference.union_log2_pfail} is not below -128 "
            "(this planning reference only selects the schedule; it is not the campaign screen)")
    columns = columns[:d]
    product_metadata = [[metadata(t, True) for t in col] for col in columns] if noise_audit else None
    waves = []
    while any(len(col)>2 for col in columns):
        next_columns = [[] for _ in range(d)]
        records,jobs = [],[]
        for q,col in enumerate(columns):
            groups = ([col] if 0<len(col)<=2 else [] if not col else
                      c.base.source_balanced_partition(col,15,p,v) if mode in ("cached", "fused") else
                      c.est.chunk_terms_by_bound(col,15,True))
            indices = {id(t):i for i,t in enumerate(col)}
            records.append({"terms":[metadata(t, noise_audit) for t in col],
                            "groups":[[indices[id(t)] for t in group] for group in groups]})
            if noise_audit:
                records[-1]["group_variances"] = [c.est.chunk_input_var_source_aware(group) for group in groups]
            for group in groups:
                if len(group)<3: next_columns[q].extend(group)
                else: jobs.append((q,group))
        for q,group in jobs:
            bound = sum(t.bound for t in group)
            next_columns[q].append(c.est.Term(3,v.normalizer_pbs_var,"norm-digit",True))
            if q+1<d and bound//4:
                next_columns[q+1].append(c.est.Term(bound//4,v.normalizer_pbs_var,"norm-carry",True))
        assert sum(map(len,next_columns))<sum(map(len,columns))
        waves.append(records)
        columns = next_columns
    snapshot = Path(__file__).parent/"source_snapshot"
    return {"width":width,"mode":mode,"squaring":squaring,"chunk_bits":chunk_bits,
        "log2_failure":None,
        "reference_screen_log2":reference.union_log2_pfail,
        "variance_reference_only":True,
        "shared_selector_analysis_status":fused_model.ANALYSIS_STATUS,
        "kernel":kernel,
        "paired_product_reference":noise_audit and mode=="fused",
        "public_scalar":None if public_scalar is None else list(public_scalar),
        "public_groups":public_groups,
        "env":parameter_environment(mode,squaring,capacity,cbs_level,kernel,auto_fft,chunk_bits),"waves":waves,
        "final_columns":[[metadata(t, noise_audit) for t in col] for col in columns],
        "audit_products":product_metadata,
        "model_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*") if p.is_file()},
        "planner_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "note":"Fixed reference partitions and variances are diagnostic comparators. Trace-to-GGSW coefficient covariance needs a derivation, including for the shared lift used by baseline/cached modes. No fused failure bound is claimed; baseline/cached values only reproduce the prior model."}
