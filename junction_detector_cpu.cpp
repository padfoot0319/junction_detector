// C++ CPU version of the junction detector.
// Before running this C++ program, create the input once using python3 export_graph_input.py.
// Then run (after building the whole project): ./build/junction_detector_cpu

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;

enum LinkType : int {
    OTHER = 0,
    PRIMARY = 1,
    SECONDARY = 2,
};

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct DualGraph {
    std::vector<long long> link_ids;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<int> link_type;
    std::vector<std::vector<Point>> link_points;

    // endpoints only, for the matlab plots (not used by the algorithm)
    std::vector<Point> plot_start;
    std::vector<Point> plot_end;

    int n() const { return static_cast<int>(link_ids.size()); }
    int directed_edge_count() const { return static_cast<int>(col_indices.size()); }
};

struct BackgroundLink {
    long long link_id = 0;
    std::vector<Point> points;   // full geometry (distance checks)
    Point plot_start;            // endpoints for the matlab drawing
    Point plot_end;
};

struct GraphInput {
    std::string source_config;
    DualGraph graph;
    std::vector<BackgroundLink> background;
};

struct JunctionCandidate {
    int junction_id = -1;
    long long seed_link_id = -1;
    std::vector<long long> secondary_link_ids;
    std::vector<long long> primary_boundary_link_ids;
};

static int parse_config_id(int argc, char** argv) {
    if (argc != 2) {
        throw std::runtime_error("Example: ./build/junction_detector_cpu <config_number>");
    }

    return std::stoi(argv[1]);
}

// detection knobs - tuned by hand on the test configs
#define MIN_SECONDARY_LINKS 2
#define MIN_BOUNDARY_PRIMARIES 2
#define P_EXPAND_MAX_DEPTH 24
#define P_EXPAND_MAX_ROUNDS 0     // p-expansion off for now, was over-merging

// merge step
#define MERGE_CANDIDATES 1
#define MERGE_MIN_SHARED_PRIMARY 1
#define MERGE_PRIMARY_BOUNDARY_HOPS 0
#define MERGE_MAX_SECONDARY_DISTANCE_M 50.0
#define MERGE_CENTROID_DISTANCE_M 300.0


static void print_row(const std::string& run_name,
                          int config_id,
                          int roads,
                          int secondary_count,
                          int primary_count,
                          int edges,
                          std::size_t junction_count,
                          double algorithm_s) {
    std::cout << std::left
              << std::setw(8) << run_name
              << std::setw(12) << ("config " + std::to_string(config_id))
              << std::right
              << std::setw(10) << roads
              << std::setw(10) << secondary_count
              << std::setw(10) << primary_count
              << std::setw(10) << edges
              << std::setw(12) << junction_count
              << std::setw(14) << std::fixed << std::setprecision(6) << algorithm_s
              << "\n";
}

static std::string input_path_for(int config_id) {
    return "intermediate/graph_input_config" + std::to_string(config_id) + ".json";
}

static std::string output_path_for(int config_id) {
    return "exports/cpu_config" + std::to_string(config_id) + ".json";
}


static bool is_synthetic(int config_id) {
    std::ifstream f("configs/config" + std::to_string(config_id) + ".json");
    if (!f) {
        return false;
    }

    try {
        json j;
        f >> j;
        return j.value("Synthetic", false);
    } catch (...) {
        return false;
    }
}



static std::vector<Point> extract_points(const std::vector<int>& offsets,
                                      const std::vector<double>& xs,
                                      const std::vector<double>& ys,
                                      int i) {
    std::vector<Point> pts;
    if (i < 0 || i + 1 >= static_cast<int>(offsets.size())) {
        return pts;
    }
    const int begin = offsets[i];
    const int end = offsets[i + 1];
    for (int k = begin; k < end; ++k) {
        if (k >= 0 && k < static_cast<int>(xs.size()) && k < static_cast<int>(ys.size())) {
            pts.push_back({xs[k], ys[k]});
        }
    }
    return pts;
}

static std::vector<Point> read_plot_xy(const json& j,
                                       const std::string& x_name,
                                       const std::string& y_name) {
    std::vector<double> xs = j.at(x_name).get<std::vector<double>>();
    std::vector<double> ys = j.at(y_name).get<std::vector<double>>();

    if (xs.size() != ys.size()) {
        throw std::runtime_error("Plot coordinate arrays have different sizes.");
    }

    std::vector<Point> out;
    out.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        out.push_back({xs[i], ys[i]});
    }
    return out;
}

static GraphInput load_graph_input(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Could not open input JSON: " + path);
    }

    json j;
    f >> j;

    GraphInput input;
    input.source_config = j.value("source_config", "");

    input.graph.link_ids = j.at("link_ids").get<std::vector<long long>>();
    input.graph.row_offsets = j.at("row_offsets").get<std::vector<int>>();
    input.graph.col_indices = j.at("col_indices").get<std::vector<int>>();
    input.graph.link_type = j.at("link_type").get<std::vector<int>>();

    std::vector<int> point_offsets = j.at("point_offsets").get<std::vector<int>>();
    std::vector<double> point_x = j.at("point_x").get<std::vector<double>>();
    std::vector<double> point_y = j.at("point_y").get<std::vector<double>>();

    const int n = static_cast<int>(input.graph.link_ids.size());
    input.graph.link_points.resize(n);
    for (int i = 0; i < n; ++i) {
        input.graph.link_points[i] = extract_points(point_offsets, point_x, point_y, i);
    }

    if (j.contains("link_plot_x0")) {
        input.graph.plot_start = read_plot_xy(j, "link_plot_x0", "link_plot_y0");
        input.graph.plot_end = read_plot_xy(j, "link_plot_x1", "link_plot_y1");
    } else {
        // older inputs don't carry plot endpoints - fall back to geometry ends
        input.graph.plot_start.resize(n);
        input.graph.plot_end.resize(n);
        for (int i = 0; i < n; ++i) {
            if (input.graph.link_points[i].size() >= 2) {
                input.graph.plot_start[i] = input.graph.link_points[i].front();
                input.graph.plot_end[i] = input.graph.link_points[i].back();
            }
        }
    }

    if (j.contains("background_link_ids")) {
        std::vector<long long> bg_ids = j.at("background_link_ids").get<std::vector<long long>>();
        std::vector<int> bg_offsets = j.at("background_point_offsets").get<std::vector<int>>();
        std::vector<double> bg_x = j.at("background_x").get<std::vector<double>>();
        std::vector<double> bg_y = j.at("background_y").get<std::vector<double>>();

        std::vector<Point> bg_start;
        std::vector<Point> bg_end;
        if (j.contains("background_plot_x0")) {
            bg_start = read_plot_xy(j, "background_plot_x0", "background_plot_y0");
            bg_end = read_plot_xy(j, "background_plot_x1", "background_plot_y1");
        }

        input.background.reserve(bg_ids.size());
        for (int i = 0; i < static_cast<int>(bg_ids.size()); ++i) {
            BackgroundLink b;
            b.link_id = bg_ids[i];
            b.points = extract_points(bg_offsets, bg_x, bg_y, i);

            if (i < static_cast<int>(bg_start.size()) && i < static_cast<int>(bg_end.size())) {
                b.plot_start = bg_start[i];
                b.plot_end = bg_end[i];
            } else if (b.points.size() >= 2) {
                b.plot_start = b.points.front();
                b.plot_end = b.points.back();
            }

            input.background.push_back(std::move(b));
        }
    }

    if (static_cast<int>(input.graph.row_offsets.size()) != n + 1) {
        throw std::runtime_error("row_offsets size must be num_links + 1");
    }
    if (static_cast<int>(input.graph.link_type.size()) != n) {
        throw std::runtime_error("link_type size must be num_links");
    }

    return input;
}

static std::vector<int> neighbors(const DualGraph& graph, int idx) {
    const int begin = graph.row_offsets[idx];
    const int end = graph.row_offsets[idx + 1];
    return std::vector<int>(graph.col_indices.begin() + begin, graph.col_indices.begin() + end);
}

// BFS over connected secondary (ramp) links starting at seed.
// returns the ramp set plus whatever primaries hang off its boundary.
static std::pair<std::unordered_set<int>, std::unordered_set<int>>
grow_ramp_component(const DualGraph& graph,
                    int seed,
                    const std::unordered_set<int>& globally_processed_secondary) {
    std::queue<int> q;
    std::unordered_set<int> secondary_set;
    std::unordered_set<int> primary_boundary;

    q.push(seed);
    secondary_set.insert(seed);

    while (!q.empty()) {
        int v = q.front();
        q.pop();

        for (int u : neighbors(graph, v)) {
            if (graph.link_type[u] == SECONDARY) {
                if (!secondary_set.count(u) && !globally_processed_secondary.count(u)) {
                    secondary_set.insert(u);
                    q.push(u);
                }
            } else if (graph.link_type[u] == PRIMARY) {
                primary_boundary.insert(u);
            }
        }
    }

    return {secondary_set, primary_boundary};
}

static std::vector<int> reconstruct_path(const std::unordered_map<int, int>& parent, int end) {
    std::vector<int> path;
    int cur = end;
    path.push_back(cur);

    while (true) {
        auto it = parent.find(cur);
        if (it == parent.end() || it->second == -1) {
            break;
        }
        cur = it->second;
        path.push_back(cur);
    }

    std::reverse(path.begin(), path.end());
    return path;
}

static std::vector<int> secondary_path_to_primary(
    const DualGraph& graph,
    int start_secondary,
    const std::unordered_set<int>& boundary_primaries,
    int origin_primary,
    const std::unordered_set<int>& existing_secondaries,
    int max_depth) {

    std::queue<std::pair<int, int>> q;
    std::unordered_map<int, int> parent;

    q.push({start_secondary, 0});
    parent[start_secondary] = -1;

    while (!q.empty()) {
        auto [v, depth] = q.front();
        q.pop();

        for (int u : neighbors(graph, v)) {
            if (graph.link_type[u] == PRIMARY) {
                if (u != origin_primary && boundary_primaries.count(u)) {
                    return reconstruct_path(parent, v);
                }
            } else if (graph.link_type[u] == SECONDARY) {
                if (existing_secondaries.count(u)) {
                    continue;
                }
                if (!parent.count(u) && depth + 1 <= max_depth) {
                    parent[u] = v;
                    q.push({u, depth + 1});
                }
            }
        }
    }

    return {};
}

// Pull in ramps that bridge two primaries already on the boundary, so a
// junction split across a short ramp chain ends up as one component.
// NOTE: disabled in the runs (max_rounds=0) - kept around for reference.
static void expand_primaries(const DualGraph& graph,
                             std::unordered_set<int>& secondary_set,
                             std::unordered_set<int>& primary_boundary,
                             int max_depth,
                             int max_rounds) {
    for (int round = 0; round < max_rounds; ++round) {
        bool changed = false;
        std::vector<int> current_primaries(primary_boundary.begin(), primary_boundary.end());

        for (int p : current_primaries) {
            for (int u : neighbors(graph, p)) {
                if (graph.link_type[u] != SECONDARY) {
                    continue;
                }
                if (secondary_set.count(u)) {
                    continue;
                }

                std::vector<int> path = secondary_path_to_primary(
                    graph, u, primary_boundary, p, secondary_set, max_depth);

                if (path.empty()) {
                    continue;
                }

                std::size_t before = secondary_set.size();
                for (int s : path) {
                    secondary_set.insert(s);
                }

                for (int s : path) {
                    for (int w : neighbors(graph, s)) {
                        if (graph.link_type[w] == PRIMARY) {
                            primary_boundary.insert(w);
                        }
                    }
                }

                if (secondary_set.size() != before) {
                    changed = true;
                }
            }
        }

        if (!changed) {
            break;
        }
    }
}

static std::vector<long long> idx_to_link_ids(const DualGraph& graph, const std::unordered_set<int>& idxs) {
    std::vector<long long> ids;
    ids.reserve(idxs.size());
    for (int i : idxs) {
        ids.push_back(graph.link_ids[i]);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static std::vector<JunctionCandidate> detect_junctions_cpu(const DualGraph& graph,
                                                           int min_secondary_links,
                                                           int min_boundary_primaries,
                                                           int p_expand_max_depth,
                                                           int p_expand_max_rounds) {
    std::unordered_set<int> processed_secondary;
    std::vector<JunctionCandidate> junctions;

    for (int seed = 0; seed < graph.n(); ++seed) {
        if (graph.link_type[seed] != SECONDARY) {
            continue;
        }
        if (processed_secondary.count(seed)) {
            continue;
        }

        // one ramp component -> one candidate, mark its ramps so we don't reseed
        auto [secondary_set, primary_boundary] =
            grow_ramp_component(graph, seed, processed_secondary);

        expand_primaries(graph, secondary_set, primary_boundary,
                         p_expand_max_depth, p_expand_max_rounds);

        if (static_cast<int>(secondary_set.size()) >= min_secondary_links &&
            static_cast<int>(primary_boundary.size()) >= min_boundary_primaries) {
            JunctionCandidate j;
            j.junction_id = static_cast<int>(junctions.size()) + 1;
            j.seed_link_id = graph.link_ids[seed];
            j.secondary_link_ids = idx_to_link_ids(graph, secondary_set);
            j.primary_boundary_link_ids = idx_to_link_ids(graph, primary_boundary);
            junctions.push_back(std::move(j));
        }

        for (int s : secondary_set) {
            processed_secondary.insert(s);
        }
    }

    return junctions;
}

class DisjointSet {
public:
    explicit DisjointSet(int n) : parent_(n) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];   // path halving
            x = parent_[x];
        }
        return x;
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra != rb) {
            parent_[rb] = ra;
        }
    }

private:
    std::vector<int> parent_;
};

static std::unordered_map<long long, int> link_id_index(const DualGraph& graph) {
    std::unordered_map<long long, int> out;
    for (int i = 0; i < graph.n(); ++i) {
        out[graph.link_ids[i]] = i;
    }
    return out;
}

static std::unordered_set<int> ids_to_idx_set(const std::unordered_map<long long, int>& map,
                                                  const std::vector<long long>& ids) {
    std::unordered_set<int> out;
    for (long long id : ids) {
        auto it = map.find(id);
        if (it != map.end()) {
            out.insert(it->second);
        }
    }
    return out;
}

static bool primaries_within_hops(const DualGraph& graph,
                                  const std::unordered_set<int>& start_primaries,
                                  const std::unordered_set<int>& target_primaries,
                                  int max_hops) {
    for (int p : start_primaries) {
        if (target_primaries.count(p)) {
            return true;
        }
    }
    if (max_hops <= 0) {
        return false;
    }

    std::queue<std::pair<int, int>> q;
    std::unordered_set<int> seen;

    for (int p : start_primaries) {
        q.push({p, 0});
        seen.insert(p);
    }

    while (!q.empty()) {
        auto [v, depth] = q.front();
        q.pop();

        if (depth >= max_hops) {
            continue;
        }

        for (int u : neighbors(graph, v)) {
            if (seen.count(u)) {
                continue;
            }
            if (graph.link_type[u] != PRIMARY) {
                continue;
            }
            if (target_primaries.count(u)) {
                return true;
            }
            seen.insert(u);
            q.push({u, depth + 1});
        }
    }

    return false;
}

static std::vector<Point> collect_sec_points(const DualGraph& graph, const std::unordered_set<int>& secondary_idxs) {
    std::vector<Point> pts;
    for (int idx : secondary_idxs) {
        const auto& p = graph.link_points[idx];
        pts.insert(pts.end(), p.begin(), p.end());
    }
    return pts;
}

static double min_pt_dist(const std::vector<Point>& a, const std::vector<Point>& b) {
    if (a.empty() || b.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double best2 = std::numeric_limits<double>::infinity();
    for (const auto& p : a) {
        for (const auto& q : b) {
            double dx = p.x - q.x;
            double dy = p.y - q.y;
            best2 = std::min(best2, dx * dx + dy * dy);
        }
    }
    return std::sqrt(best2);
}

static Point centroid(const std::vector<Point>& pts) {
    if (pts.empty()) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    double sx = 0.0;
    double sy = 0.0;
    for (const auto& p : pts) {
        sx += p.x;
        sy += p.y;
    }
    double n = static_cast<double>(pts.size());
    return {sx / n, sy / n};
}

static double centroid_distance_m(const std::vector<Point>& a, const std::vector<Point>& b) {
    Point ca = centroid(a);
    Point cb = centroid(b);
    if (!std::isfinite(ca.x) || !std::isfinite(ca.y) || !std::isfinite(cb.x) || !std::isfinite(cb.y)) {
        return std::numeric_limits<double>::infinity();
    }
    double dx = ca.x - cb.x;
    double dy = ca.y - cb.y;
    return std::sqrt(dx * dx + dy * dy);
}

// O(n^2) pairwise merge: two candidates collapse if they share a primary
// (or sit one hop apart) AND their ramps are close enough. Centroid distance
// is the looser fallback. Union-find does the actual grouping at the end.
static std::vector<JunctionCandidate> merge_junction_candidates(const DualGraph& graph,
                                                                const std::vector<JunctionCandidate>& junctions,
                                                                int min_shared_primary,
                                                                int primary_boundary_hops,
                                                                double max_secondary_distance_m,
                                                                double centroid_distance_threshold_m) {
    if (junctions.size() <= 1) {
        return junctions;
    }

    auto link_to_idx = link_id_index(graph);

    std::vector<std::unordered_set<int>> secondary_sets;
    std::vector<std::unordered_set<int>> primary_sets;
    secondary_sets.reserve(junctions.size());
    primary_sets.reserve(junctions.size());

    for (const auto& j : junctions) {
        secondary_sets.push_back(ids_to_idx_set(link_to_idx, j.secondary_link_ids));
        primary_sets.push_back(ids_to_idx_set(link_to_idx, j.primary_boundary_link_ids));
    }

    std::vector<std::vector<Point>> secondary_points;
    secondary_points.reserve(junctions.size());
    for (const auto& s : secondary_sets) {
        secondary_points.push_back(collect_sec_points(graph, s));
    }

    DisjointSet dsu(static_cast<int>(junctions.size()));

    const bool use_local_distance = max_secondary_distance_m > 0.0;
    const bool use_centroid_distance = centroid_distance_threshold_m > 0.0;

    for (int i = 0; i < static_cast<int>(junctions.size()); ++i) {
        for (int k = i + 1; k < static_cast<int>(junctions.size()); ++k) {
            int shared_count = 0;
            for (int p : primary_sets[i]) {
                if (primary_sets[k].count(p)) {
                    ++shared_count;
                }
            }

            bool has_primary_relation = false;
            std::string primary_reason;

            if (shared_count >= min_shared_primary) {
                has_primary_relation = true;
                primary_reason = "shared_primary=" + std::to_string(shared_count);
            } else if (primary_boundary_hops > 0 &&
                       primaries_within_hops(graph, primary_sets[i], primary_sets[k], primary_boundary_hops)) {
                has_primary_relation = true;
                primary_reason = "primary_hops<=" + std::to_string(primary_boundary_hops);
            }

            if (has_primary_relation) {
                if (use_local_distance) {
                    double d = min_pt_dist(secondary_points[i], secondary_points[k]);
                    if (d <= max_secondary_distance_m) {
                        std::cout << "[merge] J" << junctions[i].junction_id << "-J" << junctions[k].junction_id
                                  << ": " << primary_reason << ", secondary_min_distance=" << d << " m\n";
                        dsu.unite(i, k);
                        continue;
                    }
                    std::cout << "[merge-skip] J" << junctions[i].junction_id << "-J" << junctions[k].junction_id
                              << ": " << primary_reason << ", but secondary_min_distance=" << d
                              << " m > " << max_secondary_distance_m << " m\n";
                } else {
                    std::cout << "[merge] J" << junctions[i].junction_id << "-J" << junctions[k].junction_id
                              << ": " << primary_reason << ", local distance disabled\n";
                    dsu.unite(i, k);
                    continue;
                }
            }

            if (use_centroid_distance) {
                double d = centroid_distance_m(secondary_points[i], secondary_points[k]);
                if (d <= centroid_distance_threshold_m) {
                    std::cout << "[merge] J" << junctions[i].junction_id << "-J" << junctions[k].junction_id
                              << ": secondary_centroid_distance=" << d
                              << " m <= " << centroid_distance_threshold_m << " m\n";
                    dsu.unite(i, k);
                } else {
                    std::cout << "[centroid-skip] J" << junctions[i].junction_id << "-J" << junctions[k].junction_id
                              << ": secondary_centroid_distance=" << d
                              << " m > " << centroid_distance_threshold_m << " m\n";
                }
            }
        }
    }

    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < static_cast<int>(junctions.size()); ++i) {
        groups[dsu.find(i)].push_back(i);
    }

    std::vector<int> roots;
    roots.reserve(groups.size());
    for (const auto& kv : groups) {
        roots.push_back(kv.first);
    }
    std::sort(roots.begin(), roots.end());

    std::vector<JunctionCandidate> merged;
    int new_id = 1;

    for (int root : roots) {
        std::unordered_set<int> sec;
        std::unordered_set<int> pri;
        const auto& members = groups[root];

        for (int m : members) {
            sec.insert(secondary_sets[m].begin(), secondary_sets[m].end());
            pri.insert(primary_sets[m].begin(), primary_sets[m].end());
        }

        JunctionCandidate j;
        j.junction_id = new_id++;
        j.seed_link_id = junctions[members.front()].seed_link_id;
        j.secondary_link_ids = idx_to_link_ids(graph, sec);
        j.primary_boundary_link_ids = idx_to_link_ids(graph, pri);
        merged.push_back(std::move(j));
    }

    return merged;
}

static std::vector<long long> all_link_ids(const JunctionCandidate& j) {
    std::vector<long long> out = j.secondary_link_ids;
    out.insert(out.end(), j.primary_boundary_link_ids.begin(), j.primary_boundary_link_ids.end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static json point_polyline_json(const std::vector<Point>& pts) {
    json arr = json::array();
    for (const auto& p : pts) {
        arr.push_back({p.x, p.y});
    }
    return arr;
}

static json ids_json(const std::vector<long long>& ids) {
    json arr = json::array();
    for (auto id : ids) {
        arr.push_back(id);
    }
    return arr;
}

static void write_matlab_json(const std::string& path,
                              const GraphInput& input,
                              const std::vector<JunctionCandidate>& junctions,
                              double runtime_s) {
    const auto link_to_idx = link_id_index(input.graph);

    json out;
    out["source_config"] = input.source_config;
    out["schema"] = "cpp_junction_detection_matlab_fast_v1";
    out["summary"] = {
        {"num_dual_vertices", input.graph.n()},
        {"num_dual_directed_edges", input.graph.directed_edge_count()},
        {"num_junctions", static_cast<int>(junctions.size())},
        {"runtime_s", runtime_s},
    };

    out["interchanges"] = json::array();

    for (const auto& j : junctions) {
        std::unordered_set<long long> sec(j.secondary_link_ids.begin(), j.secondary_link_ids.end());
        std::unordered_set<long long> pri(j.primary_boundary_link_ids.begin(), j.primary_boundary_link_ids.end());

        json J;
        J["id"] = j.junction_id;
        J["junction_id"] = j.junction_id;
        J["seed_link_id"] = j.seed_link_id;
        J["secondary_link_ids"] = ids_json(j.secondary_link_ids);
        J["primary_boundary_link_ids"] = ids_json(j.primary_boundary_link_ids);
        J["all_link_ids"] = ids_json(all_link_ids(j));

        J["graph"]["nodes"] = json::array();
        J["graph"]["edges"] = json::array();

        for (long long link_id : all_link_ids(j)) {
            auto it = link_to_idx.find(link_id);
            if (it == link_to_idx.end()) {
                continue;
            }
            int idx = it->second;

            std::string role = "undefined";
            if (sec.count(link_id)) {
                role = "ramp";
            } else if (pri.count(link_id)) {
                role = "main";
            }

            J["graph"]["edges"].push_back({
                {"u", link_id * 2},
                {"v", link_id * 2 + 1},
                {"here_link_id", link_id},
                {"link_id", link_id},
                {"role", role},
                {"polyline", point_polyline_json({input.graph.plot_start[idx], input.graph.plot_end[idx]})},
            });
        }

        out["interchanges"].push_back(J);
    }

    out["background_graph"]["nodes"] = json::array();
    out["background_graph"]["edges"] = json::array();
    for (const auto& b : input.background) {
        out["background_graph"]["edges"].push_back({
            {"u", b.link_id * 2},
            {"v", b.link_id * 2 + 1},
            {"here_link_id", b.link_id},
            {"link_id", b.link_id},
            {"polyline", point_polyline_json({b.plot_start, b.plot_end})},
        });
    }

    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("Could not open MATLAB output JSON: " + path);
    }
    f << out.dump(2);
}

int main(int argc, char** argv) {
    try {
        const int config_id = parse_config_id(argc, argv);

        const std::string input_path = input_path_for(config_id);
        const std::string matlab_out_path = output_path_for(config_id);

        auto t0 = std::chrono::steady_clock::now();

        GraphInput input = load_graph_input(input_path);
        const DualGraph& graph = input.graph;

        int secondary_count = 0;
        int primary_count = 0;
        for (int t : graph.link_type) {
            if (t == SECONDARY) ++secondary_count;
            if (t == PRIMARY) ++primary_count;
        }

        std::ostringstream quiet;
        std::streambuf* old_cout = std::cout.rdbuf(quiet.rdbuf());

        auto algorithm_t0 = std::chrono::steady_clock::now();
        std::vector<JunctionCandidate> junctions = detect_junctions_cpu(
            graph,
            MIN_SECONDARY_LINKS,
            MIN_BOUNDARY_PRIMARIES,
            P_EXPAND_MAX_DEPTH,
            P_EXPAND_MAX_ROUNDS);

        if (MERGE_CANDIDATES) {
            junctions = merge_junction_candidates(
                graph,
                junctions,
                MERGE_MIN_SHARED_PRIMARY,
                MERGE_PRIMARY_BOUNDARY_HOPS,
                MERGE_MAX_SECONDARY_DISTANCE_M,
                MERGE_CENTROID_DISTANCE_M);
        }
        auto algorithm_t1 = std::chrono::steady_clock::now();

        std::cout.rdbuf(old_cout);

        auto t1 = std::chrono::steady_clock::now();
        double algorithm_s = std::chrono::duration<double>(algorithm_t1 - algorithm_t0).count();
        double total_s = std::chrono::duration<double>(t1 - t0).count();

        if (!is_synthetic(config_id)) {
            write_matlab_json(matlab_out_path, input, junctions, total_s);
        }

        print_row(
            "CPU",
            config_id,
            graph.n(),
            secondary_count,
            primary_count,
            graph.directed_edge_count(),
            junctions.size(),
            algorithm_s);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}