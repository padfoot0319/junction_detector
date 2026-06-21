#!/usr/bin/env python3

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"

PATTERNS = {
    "geomFiles": ("lane_geometry_polyline", "lane-geometry-polyline"),
    "topoFiles": ("lane_topology", "lane-topology"),
    "attrFiles": ("lane_attributes", "lane-attributes"),
    "refFiles": ("lane_road_references", "lane-road-references"),
    "routeFiles": ("routing_attributes", "routing-attributes"),
}


def ids_for(folder, tag):
    ids = set()
    pattern = re.compile(rf"{re.escape(tag)}_(\d+)\.json$")
    for p in (ROOT / folder).glob("*.json"):
        m = pattern.search(p.name)
        if m:
            ids.add(m.group(1))
    return ids


def file_name(folder, tag, tile_id):
    return f"{folder}/here-hdlm-protobuf-weu-2_{tag}_{tile_id}.json"


def make_config(tile_ids, description):
    cfg = {
        "geomFiles": [],
        "topoFiles": [],
        "attrFiles": [],
        "refFiles": [],
        "routeFiles": [],
        "OriginAltCm": 0,
        "Undirected": True,
        "node_merge_eps": 0.10,
        "k_end_candidates": 1,
        "exports_dir": "exports",
        "Description": description,
    }

    for tile_id in tile_ids:
        for key, (folder, tag) in PATTERNS.items():
            cfg[key].append(file_name(folder, tag, tile_id))

    return cfg


def main():
    ids_by_type = {}

    for key, (folder, tag) in PATTERNS.items():
        ids_by_type[key] = ids_for(folder, tag)

    common_ids = set.intersection(*ids_by_type.values())
    common_ids = sorted(common_ids, key=int)

    if not common_ids:
        raise RuntimeError("No common tile ids found.")

    heavy_sets = [
        ("config12.json", common_ids[:25], "Heavy config with 25 common tiles"),
        ("config13.json", common_ids[:75], "Heavy config with 75 common tiles"),
        ("config14.json", common_ids, "Heavy config with all common tiles"),
    ]

    CONFIG_DIR.mkdir(exist_ok=True)

    for name, tile_ids, description in heavy_sets:
        out = CONFIG_DIR / name
        cfg = make_config(tile_ids, description)

        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Wrote {out} with {len(tile_ids)} tiles.")


if __name__ == "__main__":
    main()