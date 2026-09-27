"""Diagnostic-only wrapper to identify Top-p Graph static CUDA storages."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import twilight_vram_inventory_v1 as inv  # noqa: E402

original = inv.inventory


def with_graph_inventory(model, cache, model_only_cuda):
    result = original(model, cache, model_only_cuda)
    known = {(x["device"], x["storage_data_ptr"])
             for x in result["cache_storages"] if x["device"].startswith("cuda")}
    graph = getattr(cache, "_exact_top_p_graph", None)
    entries = []
    seen = set()
    if graph is not None:
        for key, state in graph._graphs.items():
            for label, tensor in zip(("static_input", "static_order", "static_desired"), state[:3]):
                storage = tensor.untyped_storage()
                identity = (str(tensor.device), storage.data_ptr())
                if identity in seen:
                    continue
                seen.add(identity)
                entries.append({"shape_key": str(key), "component": label,
                                "storage_data_ptr": storage.data_ptr(),
                                "storage_bytes": storage.nbytes(),
                                "already_in_cache_inventory": identity in known})
    result["top_p_graph_static_storages"] = entries
    result["top_p_graph_static_unique_bytes"] = sum(
        x["storage_bytes"] for x in entries if not x["already_in_cache_inventory"])
    return result


inv.inventory = with_graph_inventory
runner = ROOT / "scripts/run_ruler_partial_h2d_tpot_case_v1.py"
sys.argv = [str(runner), *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
