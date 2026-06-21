// CUDA GPU version of the junction detector.
// Before running this CUDA program, create the input once using python3 export_graph_input.py.
// Then run (after building the whole project): ./build/junction_detector_gpu

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

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
        throw std::runtime_error("Example: ./build/junction_detector_gpu <config_number>");
    }

    return std::stoi(argv[1]);
}

// detection knobs - keep these in sync with the CPU file
#define MIN_SECONDARY_LINKS 2
#define MIN_BOUNDARY_PRIMARIES 2
#define P_EXPAND_MAX_DEPTH 24
#define P_EXPAND_MAX_ROUNDS 0     // p-expansion not implemented on the gpu path

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
    return "exports/gpu_config" + std::to_string(config_id) + ".json";
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

static std::vector<long long> idx_to_link_ids(const DualGraph& graph, const std::unordered_set<int>& idxs) {
    std::vector<long long> ids;
    ids.reserve(idxs.size());
    for (int i : idxs) {
        ids.push_back(graph.link_ids[i]);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

class DisjointSet {
public:
    explicit DisjointSet(int n) : parent_(n) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
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

// ---- CUDA part ----
// Two stages run on the device: (1) label connected secondary components by
// iterated min-label propagation, (2) the pairwise merge test. Candidate
// building and the final union/grouping stay on the host - they're cheap and
// fiddly with the hash sets.
//
// p-expansion (the CPU file's expand step) isn't ported. We run with
// P_EXPAND_MAX_ROUNDS=0 anyway so it doesn't matter for the reported numbers.

// CUDA error wrapper, from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#define CC(call)                                                             \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void init_labels_kernel(int n,
                                   const int* link_type,
                                   int* labels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (link_type[i] == SECONDARY) {
        labels[i] = i;
    } else {
        labels[i] = -1;
    }
}

// one relaxation sweep: each secondary link takes the smallest label among
// itself and its secondary neighbours. Repeat from the host until nothing
// changes -> every component ends up labelled by its min index. Simple, and
// good enough since the components are shallow.
__global__ void propagate_secondary_labels_kernel(int n,
                                                  const int* row_offsets,
                                                  const int* col_indices,
                                                  const int* link_type,
                                                  int* labels,
                                                  int* changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (link_type[v] != SECONDARY) return;

    int best = labels[v];

    for (int e = row_offsets[v]; e < row_offsets[v + 1]; ++e) {
        int u = col_indices[e];
        if (link_type[u] != SECONDARY) continue;

        int lu = labels[u];
        if (lu >= 0 && lu < best) {
            best = lu;
        }
    }

    if (best < labels[v]) {
        labels[v] = best;
        *changed = 1;
    }
}

static std::vector<JunctionCandidate> candidates_from_labels(
    const DualGraph& graph,
    const std::vector<int>& labels,
    int min_secondary_links,
    int min_boundary_primaries) {

    std::unordered_map<int, std::vector<int>> secondary_by_label;
    std::unordered_map<int, std::unordered_set<int>> primary_by_label;

    for (int i = 0; i < graph.n(); ++i) {
        if (graph.link_type[i] != SECONDARY || labels[i] < 0) {
            continue;
        }

        int label = labels[i];
        secondary_by_label[label].push_back(i);

        for (int e = graph.row_offsets[i]; e < graph.row_offsets[i + 1]; ++e) {
            int u = graph.col_indices[e];
            if (graph.link_type[u] == PRIMARY) {
                primary_by_label[label].insert(u);
            }
        }
    }

    std::vector<int> roots;
    roots.reserve(secondary_by_label.size());
    for (const auto& kv : secondary_by_label) {
        roots.push_back(kv.first);
    }
    std::sort(roots.begin(), roots.end());

    std::vector<JunctionCandidate> junctions;

    for (int root : roots) {
        std::unordered_set<int> sec;
        std::unordered_set<int> pri;

        for (int idx : secondary_by_label[root]) {
            sec.insert(idx);
        }

        auto pit = primary_by_label.find(root);
        if (pit != primary_by_label.end()) {
            pri = std::move(pit->second);
        }

        if (static_cast<int>(sec.size()) < min_secondary_links ||
            static_cast<int>(pri.size()) < min_boundary_primaries) {
            continue;
        }

        JunctionCandidate j;
        j.junction_id = static_cast<int>(junctions.size()) + 1;
        j.seed_link_id = graph.link_ids[root];
        j.secondary_link_ids = idx_to_link_ids(graph, sec);
        j.primary_boundary_link_ids = idx_to_link_ids(graph, pri);
        junctions.push_back(std::move(j));
    }

    return junctions;
}

static std::vector<JunctionCandidate> detect_junctions_gpu(
    const DualGraph& graph,
    int min_secondary_links,
    int min_boundary_primaries,
    int p_expand_max_depth,
    int p_expand_max_rounds,
    double& gpu_ms,
    int& label_iterations) {

    (void)p_expand_max_depth;

    if (p_expand_max_rounds != 0) {
        std::cout << "[WARN] GPU detector currently uses secondary-component expansion only. "
                  << "P_EXPAND_MAX_ROUNDS is ignored in the GPU path.\n";
    }

    int n = graph.n();
    if (n == 0) {
        gpu_ms = 0.0;
        label_iterations = 0;
        return {};
    }

    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    int* d_link_type = nullptr;
    int* d_labels = nullptr;
    int* d_changed = nullptr;
    std::vector<int> labels(n, -1);

    CC(cudaMalloc(&d_row_offsets, graph.row_offsets.size() * sizeof(int)));
    CC(cudaMalloc(&d_col_indices, graph.col_indices.size() * sizeof(int)));
    CC(cudaMalloc(&d_link_type, graph.link_type.size() * sizeof(int)));
    CC(cudaMalloc(&d_labels, n * sizeof(int)));
    CC(cudaMalloc(&d_changed, sizeof(int)));

    CC(cudaMemcpy(d_row_offsets, graph.row_offsets.data(),
                          graph.row_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_col_indices, graph.col_indices.data(),
                          graph.col_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_link_type, graph.link_type.data(),
                          graph.link_type.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CC(cudaEventCreate(&ev_start));
    CC(cudaEventCreate(&ev_stop));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CC(cudaEventRecord(ev_start));

    init_labels_kernel<<<blocks, threads>>>(n, d_link_type, d_labels);
    CC(cudaGetLastError());

    label_iterations = 0;
    for (int iter = 0; iter < n; ++iter) {
        int zero = 0;
        CC(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));

        propagate_secondary_labels_kernel<<<blocks, threads>>>(
            n, d_row_offsets, d_col_indices, d_link_type, d_labels, d_changed);
        CC(cudaGetLastError());

        int changed = 0;
        CC(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

        ++label_iterations;
        if (changed == 0) {
            break;
        }
    }

    CC(cudaEventRecord(ev_stop));
    CC(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CC(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    gpu_ms = static_cast<double>(elapsed_ms);

    CC(cudaMemcpy(labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));

    CC(cudaEventDestroy(ev_start));
    CC(cudaEventDestroy(ev_stop));

    CC(cudaFree(d_row_offsets));
    CC(cudaFree(d_col_indices));
    CC(cudaFree(d_link_type));
    CC(cudaFree(d_labels));
    CC(cudaFree(d_changed));

    return candidates_from_labels(
        graph, labels, min_secondary_links, min_boundary_primaries);
}

__global__ void candidate_centroid_kernel(int num_candidates,
                                          const int* sec_offsets,
                                          const int* sec_indices,
                                          const int* point_offsets,
                                          const double* point_x,
                                          const double* point_y,
                                          double* centroid_x,
                                          double* centroid_y) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_candidates) return;

    double sx = 0.0;
    double sy = 0.0;
    int count = 0;

    for (int a = sec_offsets[c]; a < sec_offsets[c + 1]; ++a) {
        int link_idx = sec_indices[a];

        for (int p = point_offsets[link_idx]; p < point_offsets[link_idx + 1]; ++p) {
            sx += point_x[p];
            sy += point_y[p];
            ++count;
        }
    }

    if (count > 0) {
        centroid_x[c] = sx / static_cast<double>(count);
        centroid_y[c] = sy / static_cast<double>(count);
    } else {
        centroid_x[c] = 1.0e30;
        centroid_y[c] = 1.0e30;
    }
}

__device__ bool contains_device(const int* values, int begin, int end, int x) {
    for (int i = begin; i < end; ++i) {
        if (values[i] == x) return true;
    }
    return false;
}

__device__ int shared_primary_count_device(const int* primary_offsets,
                                           const int* primary_indices,
                                           int a,
                                           int b) {
    int count = 0;

    int a0 = primary_offsets[a];
    int a1 = primary_offsets[a + 1];
    int b0 = primary_offsets[b];
    int b1 = primary_offsets[b + 1];

    for (int i = a0; i < a1; ++i) {
        if (contains_device(primary_indices, b0, b1, primary_indices[i])) {
            ++count;
        }
    }

    return count;
}

__device__ bool primary_sets_touch_one_hop_device(const int* primary_offsets,
                                                  const int* primary_indices,
                                                  const int* row_offsets,
                                                  const int* col_indices,
                                                  const int* link_type,
                                                  int a,
                                                  int b) {
    int a0 = primary_offsets[a];
    int a1 = primary_offsets[a + 1];
    int b0 = primary_offsets[b];
    int b1 = primary_offsets[b + 1];

    for (int i = a0; i < a1; ++i) {
        int p = primary_indices[i];

        for (int e = row_offsets[p]; e < row_offsets[p + 1]; ++e) {
            int u = col_indices[e];
            if (link_type[u] == PRIMARY &&
                contains_device(primary_indices, b0, b1, u)) {
                return true;
            }
        }
    }

    return false;
}

__device__ double min_secondary_distance_device(const int* sec_offsets,
                                                const int* sec_indices,
                                                const int* point_offsets,
                                                const double* point_x,
                                                const double* point_y,
                                                int a,
                                                int b) {
    double best = 1.0e30;

    for (int ia = sec_offsets[a]; ia < sec_offsets[a + 1]; ++ia) {
        int link_a = sec_indices[ia];

        for (int ib = sec_offsets[b]; ib < sec_offsets[b + 1]; ++ib) {
            int link_b = sec_indices[ib];

            for (int pa = point_offsets[link_a]; pa < point_offsets[link_a + 1]; ++pa) {
                double ax = point_x[pa];
                double ay = point_y[pa];

                for (int pb = point_offsets[link_b]; pb < point_offsets[link_b + 1]; ++pb) {
                    double dx = ax - point_x[pb];
                    double dy = ay - point_y[pb];
                    double d2 = dx * dx + dy * dy;

                    if (d2 < best) {
                        best = d2;
                    }
                }
            }
        }
    }

    return sqrt(best);
}

__global__ void pairwise_merge_kernel(int num_candidates,
                                      const int* sec_offsets,
                                      const int* sec_indices,
                                      const int* primary_offsets,
                                      const int* primary_indices,
                                      const int* row_offsets,
                                      const int* col_indices,
                                      const int* link_type,
                                      const int* point_offsets,
                                      const double* point_x,
                                      const double* point_y,
                                      const double* centroid_x,
                                      const double* centroid_y,
                                      int min_shared_primary,
                                      int primary_boundary_hops,
                                      double max_secondary_distance_m,
                                      double centroid_distance_m,
                                      int* merge_flags) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // only do the upper triangle, one thread per (i,k) pair
    if (i >= num_candidates || k >= num_candidates || i >= k) return;

    bool merge = false;

    int shared = shared_primary_count_device(
        primary_offsets, primary_indices, i, k);

    bool has_primary_relation = (shared >= min_shared_primary);

    if (!has_primary_relation && primary_boundary_hops > 0) {
        has_primary_relation = primary_sets_touch_one_hop_device(
            primary_offsets, primary_indices, row_offsets, col_indices, link_type, i, k);
    }

    if (has_primary_relation) {
        if (max_secondary_distance_m > 0.0) {
            double d = min_secondary_distance_device(
                sec_offsets, sec_indices, point_offsets, point_x, point_y, i, k);

            if (d <= max_secondary_distance_m) {
                merge = true;
            }
        } else {
            merge = true;
        }
    }

    if (!merge && centroid_distance_m > 0.0) {
        double dx = centroid_x[i] - centroid_x[k];
        double dy = centroid_y[i] - centroid_y[k];
        double d = sqrt(dx * dx + dy * dy);

        if (d <= centroid_distance_m) {
            merge = true;
        }
    }

    if (merge) {
        merge_flags[i * num_candidates + k] = 1;
        merge_flags[k * num_candidates + i] = 1;
    }
}

static void build_candidate_index_arrays(
    const DualGraph& graph,
    const std::vector<JunctionCandidate>& junctions,
    std::vector<std::unordered_set<int>>& secondary_sets,
    std::vector<std::unordered_set<int>>& primary_sets,
    std::vector<int>& sec_offsets,
    std::vector<int>& sec_indices,
    std::vector<int>& primary_offsets,
    std::vector<int>& primary_indices) {

    auto link_to_idx = link_id_index(graph);

    secondary_sets.clear();
    primary_sets.clear();
    sec_offsets.clear();
    sec_indices.clear();
    primary_offsets.clear();
    primary_indices.clear();

    sec_offsets.push_back(0);
    primary_offsets.push_back(0);

    for (const auto& j : junctions) {
        std::unordered_set<int> sec = ids_to_idx_set(link_to_idx, j.secondary_link_ids);
        std::unordered_set<int> pri = ids_to_idx_set(link_to_idx, j.primary_boundary_link_ids);

        std::vector<int> sec_sorted(sec.begin(), sec.end());
        std::vector<int> pri_sorted(pri.begin(), pri.end());
        std::sort(sec_sorted.begin(), sec_sorted.end());
        std::sort(pri_sorted.begin(), pri_sorted.end());

        sec_indices.insert(sec_indices.end(), sec_sorted.begin(), sec_sorted.end());
        primary_indices.insert(primary_indices.end(), pri_sorted.begin(), pri_sorted.end());

        sec_offsets.push_back(static_cast<int>(sec_indices.size()));
        primary_offsets.push_back(static_cast<int>(primary_indices.size()));

        secondary_sets.push_back(std::move(sec));
        primary_sets.push_back(std::move(pri));
    }
}

static void build_point_arrays(const DualGraph& graph,
                               std::vector<int>& point_offsets,
                               std::vector<double>& point_x,
                               std::vector<double>& point_y) {
    point_offsets.clear();
    point_x.clear();
    point_y.clear();

    point_offsets.push_back(0);

    for (const auto& pts : graph.link_points) {
        for (const auto& p : pts) {
            point_x.push_back(p.x);
            point_y.push_back(p.y);
        }
        point_offsets.push_back(static_cast<int>(point_x.size()));
    }
}

static std::vector<JunctionCandidate> merge_junction_candidates_gpu(
    const DualGraph& graph,
    const std::vector<JunctionCandidate>& junctions,
    int min_shared_primary,
    int primary_boundary_hops,
    double max_secondary_distance_m,
    double centroid_distance_threshold_m,
    double& gpu_ms) {

    if (junctions.size() <= 1) {
        gpu_ms = 0.0;
        return junctions;
    }

    if (primary_boundary_hops > 1) {
        std::cout << "[WARN] GPU merge supports exact primary sharing and one-hop "
                  << "primary adjacency. Values above 1 are treated as 1.\n";
        primary_boundary_hops = 1;
    }

    std::vector<std::unordered_set<int>> secondary_sets;
    std::vector<std::unordered_set<int>> primary_sets;
    std::vector<int> sec_offsets;
    std::vector<int> sec_indices;
    std::vector<int> primary_offsets;
    std::vector<int> primary_indices;

    build_candidate_index_arrays(
        graph, junctions, secondary_sets, primary_sets,
        sec_offsets, sec_indices, primary_offsets, primary_indices);

    std::vector<int> point_offsets;
    std::vector<double> point_x;
    std::vector<double> point_y;
    build_point_arrays(graph, point_offsets, point_x, point_y);

    int num_candidates = static_cast<int>(junctions.size());
    std::vector<int> merge_flags(num_candidates * num_candidates, 0);

    int *d_sec_offsets = nullptr, *d_sec_indices = nullptr;
    int *d_primary_offsets = nullptr, *d_primary_indices = nullptr;
    int *d_row_offsets = nullptr, *d_col_indices = nullptr, *d_link_type = nullptr;
    int *d_point_offsets = nullptr, *d_merge_flags = nullptr;
    double *d_point_x = nullptr, *d_point_y = nullptr;
    double *d_centroid_x = nullptr, *d_centroid_y = nullptr;

    CC(cudaMalloc(&d_sec_offsets, sec_offsets.size() * sizeof(int)));
    CC(cudaMalloc(&d_sec_indices, std::max<std::size_t>(1, sec_indices.size()) * sizeof(int)));
    CC(cudaMalloc(&d_primary_offsets, primary_offsets.size() * sizeof(int)));
    CC(cudaMalloc(&d_primary_indices, std::max<std::size_t>(1, primary_indices.size()) * sizeof(int)));
    CC(cudaMalloc(&d_row_offsets, graph.row_offsets.size() * sizeof(int)));
    CC(cudaMalloc(&d_col_indices, graph.col_indices.size() * sizeof(int)));
    CC(cudaMalloc(&d_link_type, graph.link_type.size() * sizeof(int)));
    CC(cudaMalloc(&d_point_offsets, point_offsets.size() * sizeof(int)));
    CC(cudaMalloc(&d_point_x, std::max<std::size_t>(1, point_x.size()) * sizeof(double)));
    CC(cudaMalloc(&d_point_y, std::max<std::size_t>(1, point_y.size()) * sizeof(double)));
    CC(cudaMalloc(&d_centroid_x, num_candidates * sizeof(double)));
    CC(cudaMalloc(&d_centroid_y, num_candidates * sizeof(double)));
    CC(cudaMalloc(&d_merge_flags, merge_flags.size() * sizeof(int)));

    CC(cudaMemcpy(d_sec_offsets, sec_offsets.data(), sec_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (!sec_indices.empty()) {
        CC(cudaMemcpy(d_sec_indices, sec_indices.data(), sec_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CC(cudaMemcpy(d_primary_offsets, primary_offsets.data(), primary_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (!primary_indices.empty()) {
        CC(cudaMemcpy(d_primary_indices, primary_indices.data(), primary_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CC(cudaMemcpy(d_row_offsets, graph.row_offsets.data(), graph.row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_col_indices, graph.col_indices.data(), graph.col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_link_type, graph.link_type.data(), graph.link_type.size() * sizeof(int), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_point_offsets, point_offsets.data(), point_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (!point_x.empty()) {
        CC(cudaMemcpy(d_point_x, point_x.data(), point_x.size() * sizeof(double), cudaMemcpyHostToDevice));
        CC(cudaMemcpy(d_point_y, point_y.data(), point_y.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    CC(cudaMemset(d_merge_flags, 0, merge_flags.size() * sizeof(int)));

    cudaEvent_t ev_start, ev_stop;
    CC(cudaEventCreate(&ev_start));
    CC(cudaEventCreate(&ev_stop));

    CC(cudaEventRecord(ev_start));

    int threads = 128;
    int blocks = (num_candidates + threads - 1) / threads;

    candidate_centroid_kernel<<<blocks, threads>>>(
        num_candidates, d_sec_offsets, d_sec_indices,
        d_point_offsets, d_point_x, d_point_y, d_centroid_x, d_centroid_y);
    CC(cudaGetLastError());

    dim3 block2(16, 16);
    dim3 grid2((num_candidates + block2.x - 1) / block2.x,
               (num_candidates + block2.y - 1) / block2.y);

    pairwise_merge_kernel<<<grid2, block2>>>(
        num_candidates,
        d_sec_offsets, d_sec_indices,
        d_primary_offsets, d_primary_indices,
        d_row_offsets, d_col_indices, d_link_type,
        d_point_offsets, d_point_x, d_point_y,
        d_centroid_x, d_centroid_y,
        min_shared_primary,
        primary_boundary_hops,
        max_secondary_distance_m,
        centroid_distance_threshold_m,
        d_merge_flags);
    CC(cudaGetLastError());

    CC(cudaEventRecord(ev_stop));
    CC(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CC(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    gpu_ms = static_cast<double>(elapsed_ms);

    CC(cudaMemcpy(merge_flags.data(), d_merge_flags,
                          merge_flags.size() * sizeof(int), cudaMemcpyDeviceToHost));

    CC(cudaEventDestroy(ev_start));
    CC(cudaEventDestroy(ev_stop));

    CC(cudaFree(d_sec_offsets));
    CC(cudaFree(d_sec_indices));
    CC(cudaFree(d_primary_offsets));
    CC(cudaFree(d_primary_indices));
    CC(cudaFree(d_row_offsets));
    CC(cudaFree(d_col_indices));
    CC(cudaFree(d_link_type));
    CC(cudaFree(d_point_offsets));
    CC(cudaFree(d_point_x));
    CC(cudaFree(d_point_y));
    CC(cudaFree(d_centroid_x));
    CC(cudaFree(d_centroid_y));
    CC(cudaFree(d_merge_flags));

    DisjointSet dsu(num_candidates);
    for (int i = 0; i < num_candidates; ++i) {
        for (int k = i + 1; k < num_candidates; ++k) {
            if (merge_flags[i * num_candidates + k] != 0) {
                dsu.unite(i, k);
            }
        }
    }

    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < num_candidates; ++i) {
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

        double gpu_component_ms = 0.0;
        int label_iterations = 0;

        std::vector<JunctionCandidate> junctions = detect_junctions_gpu(
            graph,
            MIN_SECONDARY_LINKS,
            MIN_BOUNDARY_PRIMARIES,
            P_EXPAND_MAX_DEPTH,
            P_EXPAND_MAX_ROUNDS,
            gpu_component_ms,
            label_iterations);

        double gpu_merge_ms = 0.0;

        if (MERGE_CANDIDATES) {
            junctions = merge_junction_candidates_gpu(
                graph,
                junctions,
                MERGE_MIN_SHARED_PRIMARY,
                MERGE_PRIMARY_BOUNDARY_HOPS,
                MERGE_MAX_SECONDARY_DISTANCE_M,
                MERGE_CENTROID_DISTANCE_M,
                gpu_merge_ms);
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
            "GPU",
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