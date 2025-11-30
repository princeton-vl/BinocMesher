#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"
#include "bisection.h"

// get the grid ID in the hypercube for one of the endpoint of the bipolar edge
gridID edge_key_fn(queriedEdge &edge, int index=0) {
    auto L = edge.e.L;
    int *coords = &edge.e.coords[0];
    int coords_key[3];
    for (int j = 0; j < 3; j++) coords_key[j] = coords[j];
    coords_key[edge.e.dir] += index;
    while (L > 0 && coords_key[0] % 2 == 0 && coords_key[1] % 2 == 0 && coords_key[2] % 2 == 0) {
        for (int j = 0; j < 3; j++) coords_key[j] /= 2;
        L--;
    }
    return mp(make_array3(coords_key[0], coords_key[1], coords_key[2]), L);
}

// These are functions exposed to Python
extern "C" {
    // Initialize bisection module
    void bisection_init(int bisection_group) {
        bisection::bisection_group = bisection_group;
    }

    // Initialize bisection for group t
    void bisection_init_t(int t) {
        using namespace dual_contouring;
        using namespace bisection;
        last_head = 0;
        load_group(t, 0, 1);
        load_graph(t);
        load_bip_edges(t, 1);
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        auto &edges = *edges_;
        vertex_map.resize(head.size());
        query_cnt.resize(head.size());
        computed_vertices.clear();
        // compute how many different vertices to query for each hypercube
        OMP_PRAGMA(omp parallel for schedule(dynamic))
        for (ll i = 0; i < head.size(); i++) {
            if (head[i] >= 0) {
                set<gridID> query_verts;
                for (int current = head[i]; current != -1; current = nxt[current]) {
                    int edge_p = edge_pointer[current];
                    auto key = edge_key_fn(edges[edge_p], 0);
                    if (query_verts.count(key) == 0) {
                        query_verts.insert(key);
                    }
                    key = edge_key_fn(edges[edge_p], 1);
                    if (query_verts.count(key) == 0) {
                        query_verts.insert(key);
                    }
                }
                query_cnt[i] = query_verts.size();
            }
            else {
                query_cnt[i] = 0;
            }
        }
    }

    // Get the number of vertices in the current batch to query
    // For each hypercube, we keep shrinking the cube and query the shrinked bipolar edges until no bipolar edge exists
    // In this way we can find a vertex on the isosurface that is quite close to the geometric center of the hypercube
    int bisection_hypermesh_verts(int t, int *cnts, int *center_cnts) {
        using namespace dual_contouring;
        using namespace bisection;
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        auto &edges = *edges_;
        for (int ele = 0; ele < params::n_elements; ele++) {
            cnts[ele] = 0;
            center_cnts[ele] = 0;
        }
        int total_cnt = 0, total_center_cnt = 0;
        // the current range is from bisection_group to bisection_group + bisection_group
        for (current_head = last_head; current_head < head.size(); current_head++) {
            auto tmp = query_cnt[current_head];
            int ele = current_head % params::n_elements;
            if (tmp > 0) {
                cnts[ele] += tmp;
                total_cnt += tmp;
                center_cnts[ele]++;
                total_center_cnt++;
            }
            if (total_cnt >= bisection_group && ele == params::n_elements-1) break;
        }
        if (current_head == head.size()) current_head = head.size() - 1;
        lefts.resize(total_center_cnt);
        rights.resize(total_center_cnt);
        lefts.fill(0);
        rights.fill(1);
        if (total_center_cnt > 0) {
            starts.resize(int(current_head - last_head + 1));
            center_indices.resize(int(current_head - last_head + 1));
            int cnt = 0, center_cnt = 0;
            for (int ele = 0; ele < params::n_elements; ele++) {
                for (ll i = last_head + ele; i <= current_head; i += params::n_elements) {
                    if (head[i] >= 0) {
                        starts[int(i - last_head)] = cnt;
                        center_indices[int(i - last_head)] = center_cnt;
                        cnt += query_cnt[i];
                        center_cnt++;
                    }
                }
            }
        }
        return total_center_cnt > 0;
    }

    // Output the query center vertices to Python
    void bisection_hypermesh_verts_output_center(T *center_xyz) {
        using namespace dual_contouring;
        using namespace bisection;
        auto &active_nodes = fine::active_nodes;
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        auto &edges = *edges_;
        for (int ele = 0; ele < params::n_elements; ele++) {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (ll i = last_head + ele; i <= current_head; i += params::n_elements) {
                if (head[i] >= 0) {
                    int center_index = center_indices[int(i - last_head)];
                    int n = i / params::n_elements;
                    T center[3];
                    compute_center(center, active_nodes[n].c);
                    for (int j = 0; j < 3; j++) {
                        center_xyz[center_index * 3 + j] = center[j];
                    }
                }
            }
        }
    }

    // Output the query cube-surface vertices to Python, called repeated for the number of iterations
    void bisection_hypermesh_verts_output(T *xyz, int finishing) {
        using namespace dual_contouring;
        using namespace bisection;
        auto &active_nodes = fine::active_nodes;
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        auto &edges = *edges_;
        for (int ele = 0; ele < params::n_elements; ele++) {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (ll i = last_head + ele; i <= current_head; i += params::n_elements) {
                if (head[i] >= 0) {
                    int center_index = center_indices[int(i - last_head)];
                    T center[3];
                    T middle;
                    // in the non finishing iteration, always check the midpoint
                    // in the final round, specially compute the right point where the sign change happens
                    if (finishing) middle = rights[center_index];
                    else middle = (lefts[center_index] + rights[center_index]) / 2;
                    int coords[3];
                    T ecoords[3];
                    int n = i / params::n_elements;
                    compute_center(center, active_nodes[n].c);
                    int cnt = starts[int(i - last_head)];
                    set<gridID> query_verts;
                    for (int current = head[i]; current != -1; current = nxt[current]) {
                        int edge_p = edge_pointer[current];
                        int L = edges[edge_p].e.L;
                        auto key = edge_key_fn(edges[edge_p]);
                        if (query_verts.count(key) == 0) {
                            query_verts.insert(key);
                            for (int j = 0; j < 3; j++) coords[j] = edges[edge_p].e.coords[j];
                            compute_coords(ecoords, coords, L);
                            for (int j = 0; j < 3; j++) {
                                xyz[cnt * 3 + j] = center[j] * (1 - middle) + ecoords[j] * middle;
                            }
                            cnt++;
                        }
                        key = edge_key_fn(edges[edge_p], 1);
                        if (query_verts.count(key) == 0) {
                            query_verts.insert(key);
                            for (int j = 0; j < 3; j++) coords[j] = edges[edge_p].e.coords[j];
                            coords[edges[edge_p].e.dir]++;
                            compute_coords(ecoords, coords, L);
                            for (int j = 0; j < 3; j++) {
                                xyz[cnt * 3 + j] = center[j] * (1 - middle) + ecoords[j] * middle;
                            }
                            cnt++;
                        }
                    }
                }
            }
        }
    }

    // According to queried occupancy values, decide to the left/right bounds of the cube size for next iteration
    void bisection_hypermesh_verts_iter(sdfT *sdfs, sdfT *center_sdfs) {
        using namespace dual_contouring;
        using namespace bisection;
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        auto &edges = *edges_;
        
        for (int ele = 0; ele < params::n_elements; ele++) {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (ll i = last_head + ele; i <= current_head; i += params::n_elements) {
                if (head[i] >= 0) {
                    int center_index = center_indices[int(i - last_head)];
                    int cnt = starts[int(i - last_head)];
                    T middle = (lefts[center_index] + rights[center_index]) / 2;
                    int n = i / params::n_elements;
                    bool diff = 0;
                    for (int j = 0; j < query_cnt[i]; j++) {
                        assert(!std::isnan(sdfs[cnt + j]));
                        diff |= (sdfs[cnt + j] >= 0) != (center_sdfs[center_index] >= 0);
                    }
                    // If any query vertex has different sign than the center, we move the right bound to middle
                    if (diff) rights[center_index] = middle;
                    else lefts[center_index] = middle;
                }
            }
        }
    }

    // Compute the final center vertex after all iterations are done
    void bisection_hypermesh_verts_finishing(int t, sdfT *sdfs, sdfT *center_sdfs) {
        using namespace dual_contouring;
        using namespace bisection;
        auto &active_nodes = fine::active_nodes;
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        auto &edges = *edges_;
        int prev_verts_size = computed_vertices.size();
        assert((ll)prev_verts_size + lefts.size() <= INT_MAX);
        computed_vertices.resize(prev_verts_size + lefts.size());
        for (int ele = 0; ele < params::n_elements; ele++) {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (ll i = last_head + ele; i <= current_head; i += params::n_elements) {
                if (head[i] >= 0) {
                    int n = i / params::n_elements;
                    int center_index = center_indices[int(i - last_head)];
                    int cnt = starts[int(i - last_head)];
                    T center[3];
                    compute_center(center, active_nodes[n].c);
                    int intersected = -1;
                    for (int j = 0; j < query_cnt[i]; j++) {
                        assert(!std::isnan(sdfs[cnt + j]));
                        if ((sdfs[cnt + j] >= 0) != (center_sdfs[center_index] >= 0)) {
                            intersected = j;
                            break;
                        }
                    }
                    assert(intersected != -1);
                    int coords[3];
                    T ecoords[3];
                    T intersected_coords[3];
                    set<gridID> query_verts;
                    for (int current = head[i]; current != -1; current = nxt[current]) {
                        int edge_p = edge_pointer[current];
                        int L = edges[edge_p].e.L;
                        auto key = edge_key_fn(edges[edge_p]);
                        int flag = 0;
                        if (query_verts.count(key) == 0) {
                            if (query_verts.size() == intersected) {
                                flag = 1;
                            }
                            query_verts.insert(key);
                        }
                        key = edge_key_fn(edges[edge_p], 1);
                        if (query_verts.count(key) == 0) {
                            if (query_verts.size() == intersected) {
                                flag = 2;
                            }
                            query_verts.insert(key);
                        }
                        // In this final step, we compute the vertices location with the right point hypercube size, therefore there must be a sign change from geometric center vertex to some vertex
                        // We find any of it and use it as the center vertex location
                        if (flag) {
                            for (int j = 0; j < 3; j++) coords[j] = edges[edge_p].e.coords[j];
                            coords[edges[edge_p].e.dir] += flag - 1;
                            compute_coords(ecoords, coords, L);
                            T r = rights[center_index];
                            for (int j = 0; j < 3; j++) {
                                intersected_coords[j] = center[j] * (1 - r) + ecoords[j] * r;
                            }
                            break;
                        }
                    }
                    int index = prev_verts_size + center_index;
                    auto &v = computed_vertices[index];
                    // At the same time, construct the vertex_map for later use
                    vertex_map[i] = index;
                    for (int j = 0; j < 3; j++) v.first.first[j] = intersected_coords[j];
                    v.first.second[1] = 1 << (params::max_tL - active_nodes[n].c.tL);
                    v.first.second[0] = (active_nodes[n].c.tcoord << (1 + params::max_tL - active_nodes[n].c.tL)) + v.first.second[1];
                    v.second = inview_tag[n];
                }
            }
        }
        last_head = current_head + 1;
    }
}

// First, we write the computed center vertices to disk
void write_computed_vertices(int t) {
    using namespace bisection;
    std::stringstream filename;
    filename << params::output_path << "/computed_vertices/" << t << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    write_vec(outfile, vertex_map);
    write_vec(outfile, computed_vertices);
    fclose(outfile);
}

// Later we compute the 4D mesh consisting of polyhedra, and we extend the stored vertices of group t to include vertices on other groups that is contained by polyhedra found in group t
// We load computed vertices from disk for related groups and combined them to a new lookup table "hypervertices"
void load_computed_vertices(int t) {
    using namespace bisection;
    std::stringstream filename;
    filename << params::output_path << "/computed_vertices/" << t << ".bin";
    FILE *infile = fopen(filename.str().c_str() , "rb" );
    assert(infile != NULL);
    read_vec(infile, vertex_map);
    read_vec(infile, computed_vertices);
    fclose(infile);
}

// Write the new lookup table "hypervertices" for group t
void write_vertices(int t) {
    using namespace bisection;
    std::stringstream filename;
    filename << params::output_path << "/hypervertices/" << t << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    hypervertices_vec.a.assign(hypervertices.begin(), hypervertices.end());
    hypervertices_vec._size = hypervertices.size();
    write_vec(outfile, hypervertices_vec);
    fclose(outfile);
    CLS(hypervertices_vec);
    CL(hypervertices);
}

// Load the new lookup table for updating
void load_vertices(int t) {
    using namespace bisection;
    std::stringstream filename;
    filename << params::output_path << "/hypervertices/" << t << ".bin";
    FILE *infile = fopen(filename.str().c_str() , "rb" );
    MEASURE_TIME("load vec", 0, {
        read_vec(infile, hypervertices_vec);
    });

    MEASURE_TIME("convert vec to map", 0, {
        hypervertices.clear();
        for (int i = 0; i < hypervertices_vec.size(); i++) {
            hypervertices[hypervertices_vec[i].first] = hypervertices_vec[i].second;
        }
        PRINT_RSS_MEMORY("");
        CLS(hypervertices_vec);
    });
    fclose(infile);
}

// Write the 4D polyhedra found in group t
void write_hyperpolys(int t) {
    using namespace bisection;
    std::stringstream filename;
    filename << params::output_path << "/hyperpolys/" << t << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    write_vec(outfile, hyperpolys);    
    fclose(outfile);
    hyperpolys.clear();
}

// Load the 4D polyhedra for updating
void load_hyperpolys(int t) {
    using namespace bisection;
    std::stringstream filename;
    filename << params::output_path << "/hyperpolys/" << t << ".bin";
    FILE *infile = fopen(filename.str().c_str() , "rb" );
    assert(infile != NULL);
    read_vec(infile, hyperpolys);
    fclose(infile);
}

// Given bipolar edges and its neighbors, look for those in group t, construct the index of the 4D polyhedra by vertex_map
// Also use 1 << params::max_tL as marker for time boundary vertices
void process_hyperpolys(int t) {
    const int N_THREAD2 = 4;
    using namespace dual_contouring;
    using namespace bisection;
    auto &edges = *edges_;
    OMP_PRAGMA(omp parallel for schedule(dynamic))
    for (int g = 0; g < N_THREAD2; g++) {
        int start_g = ((ll)edges.size() + N_THREAD2 - 1) / N_THREAD2 * g;
        int end_g = smin(((ll)edges.size() + N_THREAD2 - 1) / N_THREAD2 * (g+1), (ll)edges.size());
        for (int index = start_g; index < end_g; index++) {
            auto &ed = edges[index];
            auto &hyperpoly = hyperpolys[index];
            hyperpoly.second = ed.e.ele;
            for (int i = 0; i < 8; i++) {
                bool pass = ed.vertices_gid[i] % (1<<params::max_tL) == t;
                bool begining = ed.vertices_gid[i] / (1<<params::max_tL) == 1;
                bool ending = ed.vertices_gid[i] / (1<<params::max_tL) == 2;
                if (pass) {
                    int A = vertex_map[(ll)ed.vertices_nid[i] * params::n_elements + ed.e.ele];
                    hyperpoly.first[i].first = A;
                    int B = hyperpoly.first[i].second = ed.vertices_gid[i];
                    auto tmp = mp(mp(A, B), computed_vertices[A]);
                    if (begining || ending) tmp.second.first.second[1] = 0;
                    if (begining) tmp.second.first.second[0] = 0;
                    if (ending) tmp.second.first.second[0] = 2 << params::max_tL;
                    hypervertices_g[g][tmp.first] = tmp.second;
                }
            }
        }
    }
}

// process_hyperpolys is called in parallel threads, here we collect the results
void collect_hypervertices() {
    using namespace bisection;
    const int N_THREAD2 = 4;
    for (int g = 0; g < N_THREAD2; g++) {
        for (auto &tmp: hypervertices_g[g]) {
            hypervertices[tmp.first] = tmp.second;
        }
        PRINT_RSS_MEMORY("");
        CL(hypervertices_g[g]);
    }
}

// These are functions exposed to Python
extern "C" {
    // Write final 4D mesh of group t to disk
    void write_final_hypermesh(int t) {
        using namespace dual_contouring;
        using namespace bisection;
        auto &edges = *edges_;
        CLS(fine::active_nodes);
        // We write the computed center vertices strictly in group t first
        MEASURE_TIME("write_final_hypermesh part 1", 1, {
            write_computed_vertices(t);
        });
        MEASURE_TIME("write_final_hypermesh part 2", 1, {
            hyperpolys.resize(edges.size());
            for (int i = 0; i < hyperpolys.size(); i++) {
                auto &verts= hyperpolys[i];
                for (auto &val: verts.first) {
                    val.first = -1;
                    val.second = -1;
                }
            }
        });
        PRINT_RSS_MEMORY("");
        assert(hypervertices_vec.empty());
        assert(hypervertices.empty());
        HVTable hypervertices_g[N_THREAD];
        // process and write the 4D polydra of current group
        MEASURE_TIME("write_final_hypermesh part 3", 1, {
            process_hyperpolys(t);
            collect_hypervertices();
        });
        MEASURE_TIME("write_final_hypermesh part 4", 1, {
            write_hyperpolys(t);
            write_vertices(t);
        });
        vec<int, int> restricted_set;
        restricted_set.clear();
        for (int i = t, j = 0; i != 0; i >>= 1, j++) {
            restricted_set.push_back((i-1) << j);
        }
        // load related groups' polyhedra, for those of them containing neighbors in the current group,
        // we update them according the computed vertices in the current group
        for (auto t_group: restricted_set) {
            MEASURE_TIME("write_final_hypermesh part 5a", 1, {
                load_bip_edges(t_group, 1);
                load_hyperpolys(t_group);
                load_vertices(t_group);
                process_hyperpolys(t);
                collect_hypervertices();
            });
            MEASURE_TIME("write_final_hypermesh part 5", 1, {
                write_hyperpolys(t_group);
                write_vertices(t_group);
            });
        }
        // On the other hand, for the current group's polyhedra, if it contain neighbors in the related groups,
        // we also update them according to the computed vertices in those related groups
        MEASURE_TIME("write_final_hypermesh part 6", 1, {
            load_bip_edges(t, 1);
            load_hyperpolys(t);
            load_vertices(t);
            for (auto t_group: restricted_set) {
                load_computed_vertices(t_group);
                process_hyperpolys(t_group);
            }
            collect_hypervertices();
        });
        MEASURE_TIME("write_final_hypermesh part 7", 1, {
            write_hyperpolys(t);
            write_vertices(t);
        });
        MEASURE_TIME("write_final_hypermesh part 8", 1, {
            clean_cache();
            CLS(hyperpolys);
            CLS(vertex_map);
            CLS(computed_vertices);
            CLS(inview_tag);
        });
    }

    // Function to get the number of vertices in the current loaded group for statistics
    int verts_count(int t) {
        using namespace dual_contouring;
        using namespace bisection;
        load_group(t, 0, 0);
        load_graph(t);
        load_bip_edges(t, 1);
        auto &head = *head_;
        int cnt = 0;
        for (ll i = 0; i < head.size(); i++) {
            if (head[i] >= 0) {
                cnt++;
            }
        }
        return cnt;
    }

    // Clean up bisection module
    void bisection_clean_up() {
        {
            using namespace coarse;
            CLS(nodes);
            CLS(leaf_nodes_vector);
            CLS(nodemap);
        }
        {
            using namespace medium;
            CLS(visible_cubes);
        }
        {
            using namespace fine;
            CLS(visible_cubes_nodes_size);
        }
        {
            using namespace rearrange;
            CLS(nodes);
            CLS(group_id_start);
            CLS(group_id_end);
            CLS(tree_sizes);
        }
        {
            using namespace bisection;
            CLS(lefts);
            CLS(rights);
            CLS(starts);
            CLS(center_indices);
        }
    }
}