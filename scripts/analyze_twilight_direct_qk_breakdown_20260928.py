#!/usr/bin/env python3
"""Analyze saved direct-QK profiling without mixing formal and diagnostic clocks."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from analyze_twilight_latest_breakdown_v1 import ownership, vram_subcategory

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results/twilight_direct_qk_breakdown_20260928_v1"
OUT = ART / "analysis"
REQUESTS = ("001_niah_multikey_3_i011", "002_vt_i002", "003_qa_1_i011")
MIB = 2**20


def load(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    if not rows:
        raise ValueError(name)
    path = OUT / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formal_ms(path):
    result = load(path)
    assert result["status"] == "ok" and result["D2_D128_tpot"]["tokens"] == 31
    rows = list(csv.DictReader(path.with_name("result.per_token.csv").open()))
    assert len(rows) == 32
    ms = fmean(float(x["latency_seconds"]) * 1000 for x in rows[1:])
    assert abs(ms - 1000 * result["D2_D128_tpot"]["mean_seconds_per_token"]) < 1e-6
    return result, ms


def components(memory):
    out = defaultdict(int)
    for storage in memory["cache_storages"]:
        if storage["device"].startswith("cuda"):
            out[vram_subcategory(storage)] += storage["storage_bytes"]
    assert sum(out.values()) == memory["cache_unique_gpu_storage_bytes"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    formal = []
    for rep in (1, 2):
        for request in REQUESTS:
            pair = {}
            for arm in ("exact", "direct"):
                d, ms = formal_ms(ART / "formal" / f"rep{rep}" / request / arm / "result.json")
                assert d["twilight_qk_backend"] == ("triton" if arm == "direct" else "triton_prepare")
                pair[arm] = (d, ms)
                formal.append({"request": request, "rep": rep, "arm": arm,
                               "formal_ms_per_token": ms,
                               "scope": "unprofiled synchronized wall D2-D32"})
            assert pair["direct"][0]["prompt_sha256"] == pair["exact"][0]["prompt_sha256"]
            assert pair["direct"][0]["checkpoint_logits_sha256"]["P4"] == pair["exact"][0]["checkpoint_logits_sha256"]["P4"]
    write_csv("formal_tpot.csv", formal)
    formal_direct = fmean(x["formal_ms_per_token"] for x in formal if x["arm"] == "direct")
    formal_exact = fmean(x["formal_ms_per_token"] for x in formal if x["arm"] == "exact")

    normal = []
    detailed = []
    for request in REQUESTS:
        for stage, field, target in (("normal", "normal_overlap_profile_tokens", normal),
                                     ("selection", "diagnostic_breakdown_tokens", detailed)):
            d = load(ART / stage / "rep1" / request / "direct/result.json")
            assert d["status"] == "ok" and d["twilight_qk_backend"] == "triton"
            assert len(d[field]) == 3
            formal_check = load(ART / "formal/rep1" / request / "direct/result.json")
            assert d["checkpoint_logits_sha256"] == formal_check["checkpoint_logits_sha256"]
            assert d["final_logits_sha256"] == formal_check["final_logits_sha256"]
            target.extend((request, t) for t in d[field])

    timeline = []
    for request, token in normal:
        amounts = ownership(token)
        row = {"request": request,
               "decode_step": token["cache_timeline"]["selection"][0]["decode_step"],
               "diagnostic_wall_ms": 1000 * token["diagnostic_token_wall_seconds"], **amounts}
        timeline.append(row)
    categories = sorted({k for row in timeline for k in row} - {"request", "decode_step", "diagnostic_wall_ms"})
    for row in timeline:
        for c in categories:
            row.setdefault(c, 0.0)
        assert abs(sum(row[c] for c in categories) - row["diagnostic_wall_ms"]) < 1e-5
    write_csv("timeline_per_token.csv", timeline)
    normal_wall = fmean(x["diagnostic_wall_ms"] for x in timeline)
    breakdown = [{"component": c, "mutually_exclusive_exposed_ms": fmean(x[c] for x in timeline),
                  "percent_of_normal_diagnostic_wall": 100 * fmean(x[c] for x in timeline) / normal_wall,
                  "scope": "common-origin priority interval ownership"} for c in categories]
    write_csv("tpot_breakdown.csv", breakdown)
    groups = {
        "kv_selection": ("selection_query_prep", "selection_quest_first_round", "selection_twilight_second_round"),
        "selection_post_gqa_handoff": ("post_selection_gqa_union", "group_length_handoff"),
        "previous_token_cache_management": ("cache_gpu_hit_miss_map", "cache_new_token_insert",
            "cache_other_control", "cache_position_bitmap_state", "cache_resident_snapshot"),
        "resident_hit_mapped_miss_assembly": ("mapped_host_read_fused_assembly",),
        "attention": ("attention",),
        "other_model_compute": ("other_model_compute",),
        "new_kv_writeback": ("new_kv_writeback",),
        "runtime_control_residual": ("runtime_control_residual",),
    }
    major = []
    for name, leaves in groups.items():
        ms = sum(fmean(x[c] for x in timeline) for c in leaves)
        major.append({"component": name, "exposed_ms": ms,
                      "percent_of_normal_diagnostic_wall": 100*ms/normal_wall,
                      "leaf_categories": "+".join(leaves)})
    assert abs(sum(x["exposed_ms"] for x in major)-normal_wall) < 1e-6
    write_csv("tpot_major_breakdown.csv", major)
    selection_core_exposed = sum(fmean(x[c] for x in timeline) for c in (
        "selection_query_prep", "selection_quest_first_round", "selection_twilight_second_round"))
    selection_stages = []
    for c in ("selection_query_prep", "selection_quest_first_round", "selection_twilight_second_round",
              "post_selection_gqa_union", "group_length_handoff"):
        ms = fmean(x[c] for x in timeline)
        selection_stages.append({"component": c, "exposed_ms": ms,
                                 "percent_of_selection_core_exposed": 100*ms/selection_core_exposed,
                                 "percent_of_normal_diagnostic_wall": 100*ms/normal_wall})
    write_csv("selection_common_origin.csv", selection_stages)

    detailed_wall = fmean(1000*t["diagnostic_token_wall_seconds"] for _, t in detailed)
    selection_core = fmean(1000*t["twilight_selection_core_cuda_seconds"] for _, t in detailed)
    selection_keys = (
        "twilight_query_prepare", "twilight_quest_metadata_score", "twilight_quest_page_topk",
        "twilight_b0_token_expand", "twilight_quest_b0", "twilight_int4_metadata_stack",
        "twilight_fused_int4_qk", "twilight_qk_scale", "twilight_int4_qk",
        "twilight_top_p_graph_replay", "twilight_top_p", "twilight_membership_decision",
        "twilight_selection_core", "twilight_gpu_gqa_union",
    )
    selection = []
    for key in selection_keys:
        ms = fmean(1000*t.get(key+"_cuda_seconds", 0) for _, t in detailed)
        selection.append({"component": key, "cuda_event_ms": ms,
                          "percent_of_selection_core": 100*ms/selection_core,
                          "percent_of_detailed_diagnostic_wall": 100*ms/detailed_wall,
                          "scope": "detailed diagnostic; parent/child overlap; Graph internals not split"})
    write_csv("selection_detailed.csv", selection)

    cache_names = ("gpu_bitmap_hit_miss_mapping", "fused_resident_hit_mapped_miss_assembly",
                   "current_new_token_slot_copy", "resident_kv_snapshot_update",
                   "resident_position_bitmap_state_update", "cache_metadata_and_control")
    cache = []
    for name in cache_names:
        gpu, host = [], []
        for _, t in normal:
            matches = [p for row in t["cache_timeline"]["cache_update"]
                       for p in row["subphases"] if p["name"] == name]
            assert len(matches) == 28
            gpu.append(sum(p["gpu_end_ms"]-p["gpu_start_ms"] for p in matches))
            host.append(sum(p["host_end_ms"]-p["host_start_ms"] for p in matches))
        cache.append({"component": name, "cuda_event_ms": fmean(gpu),
                      "host_span_ms": fmean(host), "percent_of_normal_diagnostic_wall":
                      100*fmean(gpu)/normal_wall, "scope": "parent/child and CPU/GPU overlap"})
    handoff = fmean(sum(row["index_d2h_end_ms"]-row["index_d2h_start_ms"]
                        for row in t["cache_timeline"]["selection"]) for _, t in normal)
    cache.append({"component": "group_length_handoff", "cuda_event_ms": handoff,
                  "host_span_ms": fmean(sum(row["cpu_decode_end_ms"]-row["cpu_decode_start_ms"]
                      for row in t["cache_timeline"]["selection"]) for _, t in normal),
                  "percent_of_normal_diagnostic_wall": 100*handoff/normal_wall,
                  "scope": "post-selection group handoff; not additive to cache parent"})
    write_csv("cache_details.csv", cache)
    new_kv = []
    for label, key, source in (("gpu_staging_and_schedule", "new_kv/stage_and_schedule", "modules"),
                               ("gpu_d2h_copy", "d2h", "timeline"),
                               ("cpu_host_wait", "host_wait", "timeline"),
                               ("cpu_host_scatter", "host_scatter", "timeline")):
        values = []
        for _, t in normal:
            if source == "modules":
                spans = [x for x in t["module_intervals"] if x["name"] == key]
                values.append(sum(x["gpu_end_ms"]-x["gpu_start_ms"] for x in spans))
            else:
                spans = t["cache_timeline"]["new_kv"]
                values.append(sum(x[f"{key}_end_ms"]-x[f"{key}_start_ms"] for x in spans
                                  if x.get(f"{key}_start_ms") is not None))
        new_kv.append({"component": label, "active_ms": fmean(values),
                       "clock_domain": "host wall" if key.startswith("host_") else "CUDA Event",
                       "scope": "may overlap; not additive to exposed ownership"})
    write_csv("new_kv_details.csv", new_kv)

    memory_rows, workspace_rows, storage_rows = [], [], []
    for request in REQUESTS:
        for arm in ("exact", "direct"):
            m = load(ART / "vram/rep1" / request / arm / "inventory.json")
            assert len(m["selection_workspace_probe_D33"]) == 28
            c = components(m)
            for name, size in c.items():
                storage_rows.append({"request": request, "arm": arm, "component": name,
                                     "unique_bytes": size, "unique_mib": size/MIB,
                                     "percent_of_unique_cache": 100*size/m["cache_unique_gpu_storage_bytes"]})
            end = m["end_of_d32_cuda"]
            model = m["model_only_cuda"]
            process_peak = max([end["peak_allocated_bytes"]] + [
                x.get("peak_allocated_bytes", 0) for x in m["stages"].values()])
            process_reserved_peak = max([end["peak_reserved_bytes"]] + [
                x.get("peak_reserved_bytes", 0) for x in m["stages"].values()])
            memory_rows.append({"request": request, "arm": arm,
                                "model_only_allocated_mib": model["allocated_bytes"]/MIB,
                                "d32_live_allocated_mib": end["allocated_bytes"]/MIB,
                                "d32_decode_peak_allocated_mib": end["peak_allocated_bytes"]/MIB,
                                "process_peak_allocated_mib": process_peak/MIB,
                                "reserved_mib": end["reserved_bytes"]/MIB,
                                "peak_reserved_mib": process_reserved_peak/MIB,
                                "incremental_allocated_mib": (end["allocated_bytes"]-model["allocated_bytes"])/MIB,
                                "unique_cache_gpu_mib": m["cache_unique_gpu_storage_bytes"]/MIB,
                                "cpu_historical_pinned_mib": m["cache_cpu_pinned_categories"]["full_historical_kv_cpu_pinned"]["storage_bytes"]/MIB})
            for p in m["selection_workspace_probe_D33"]:
                workspace_rows.append({"request": request, "arm": arm, "layer": p["layer"],
                                       "post_top_p_live_local_unique_mib": p["live_local_temporary_unique_storage_bytes"]/MIB,
                                       "selection_peak_above_entry_mib":
                                       (p["selection_peak_allocated_bytes"]-p["selection_start_allocated_bytes"])/MIB})
    write_csv("vram_process.csv", memory_rows)
    write_csv("vram_storage.csv", storage_rows)
    write_csv("selection_workspace.csv", workspace_rows)
    graph_rows = []
    for request in REQUESTS:
        m = load(ART / "graph_inventory" / request / "inventory.json")
        for item in m["top_p_graph_static_storages"]:
            graph_rows.append({"request": request, "component": item["component"],
                               "storage_bytes": item["storage_bytes"],
                               "storage_mib": item["storage_bytes"]/MIB,
                               "already_in_cache_inventory": item["already_in_cache_inventory"]})
    write_csv("graph_static_storage.csv", graph_rows)

    nohit = []
    for request in REQUESTS:
        direct, dm = formal_ms(ART / "nohit/rep1" / request / "direct/result.json")
        miss, mm = formal_ms(ART / "nohit/rep1" / request / "nohit/result.json")
        assert direct["checkpoint_logits_sha256"] == miss["checkpoint_logits_sha256"]
        assert direct["final_logits_sha256"] == miss["final_logits_sha256"]
        trace = [r for r in miss["resident_reuse_trace"] if 2 <= r["decode_step"] <= 32]
        assert len(trace) == 31*28 and all(r["hit_rows"] == 0 for r in trace)
        nohit.append({"request": request, "direct_resident_ms": dm, "all_miss_ms": mm,
                      "reuse_net_benefit_ms": mm-dm,
                      "all_miss_mapped_logical_mib_per_token":
                      sum(r["mapped_cpu_kv_read_bytes"] for r in trace)/31/MIB,
                      "checkpoint_final_logits_exact": True,
                      "scope": "single matched pair per request; preliminary"})
    write_csv("nohit_preliminary.csv", nohit)

    summary = {"formal_exact_mean_ms": formal_exact, "formal_direct_mean_ms": formal_direct,
               "formal_gain_percent": 100*(formal_exact-formal_direct)/formal_exact,
               "normal_diagnostic_wall_ms": normal_wall,
               "detailed_diagnostic_wall_ms": detailed_wall,
               "detailed_selection_core_cuda_event_ms": selection_core,
               "selection_workspace_exact_peak_mib": fmean(x["selection_peak_above_entry_mib"] for x in workspace_rows if x["arm"]=="exact"),
               "selection_workspace_direct_peak_mib": fmean(x["selection_peak_above_entry_mib"] for x in workspace_rows if x["arm"]=="direct"),
               "selection_workspace_exact_live_mib": fmean(x["post_top_p_live_local_unique_mib"] for x in workspace_rows if x["arm"]=="exact"),
               "selection_workspace_direct_live_mib": fmean(x["post_top_p_live_local_unique_mib"] for x in workspace_rows if x["arm"]=="direct"),
               "nohit_mean_benefit_ms": fmean(x["reuse_net_benefit_ms"] for x in nohit),
               "full_32k_gpu_kv_bytes": load(ART/"full_gpu_kv_32k.json")["full_gpu_kv_cuda_storage_bytes"],
               "direct_unique_cache_gpu_mib": fmean(x["unique_cache_gpu_mib"] for x in memory_rows if x["arm"]=="direct"),
               "direct_incremental_allocated_mib": fmean(x["incremental_allocated_mib"] for x in memory_rows if x["arm"]=="direct"),
               "top_p_graph_static_unique_mib": fmean(
                   load(ART/"graph_inventory"/r/"inventory.json")["top_p_graph_static_unique_bytes"]/MIB
                   for r in REQUESTS)}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
