#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"

// These are functions exposed to Python
extern "C" {
    // Initialize dual contouring step
    void dual_contouring_init() {
        using namespace dual_contouring;
        unfinalized_edges.clear();
        unfinalized_edges_sorted_cnt = 0;
    }
}

// Load the compressed fine node information
void load_rearranged_fine_nodes(int t) {
    using namespace dual_contouring;
    {
        std::stringstream filename;
        filename << params::output_path << "/rearranged_fine_nodes/" << t << ".bin";
        FILE *infile = fopen(filename.str().c_str() , "rb");
        vec<int, int> flatten_nodes_compressed;
        read_vec(infile, flatten_nodes_compressed);
        fclose(infile);
        rearrange::flatten_nodes.clear();
        for (int j = 0; j < flatten_nodes_compressed.size(); j++) {
            for (int i = 0; i < 16; i++) {
                rearrange::flatten_nodes.push_back((flatten_nodes_compressed[j] >> (i*2)) & 3);
            }
        }
        CLS(flatten_nodes_compressed);
    }
    {
        using namespace rearrange;
        std::stringstream filename;
        filename << params::output_path << "/rearranged_nodemap/" << t << ".bin";
        FILE *infile = fopen(filename.str().c_str() , "rb");
        read_vec(infile, info_offset_rearranged_vec);
        fclose(infile);
        info_offset_rearranged.clear();
        for (int i = 0; i < info_offset_rearranged_vec.size(); i++) {
            info_offset_rearranged.resize(info_offset_rearranged_vec[i].first + 1);
            info_offset_rearranged[info_offset_rearranged_vec[i].first] = info_offset_rearranged_vec[i].second;
        }
        CLS(info_offset_rearranged_vec);
    }
}

// uncompress the information into the current active tree
void unzip_nodes(int root, int subtree_st, int subtree_ed, int l, vec<int, int> &nodes_queue) {
    using namespace dual_contouring;
    int info_offset = info_offset_rearranged[l];
    medium_cube mc = medium::visible_cubes[l];
    int ind = coarse::leaf_nodes_vector[mc.first];
    hypercube c = coarse::nodes[ind].c;
    int s = grid_node_level(coarse::nodes[ind]);
    c.L += s;
    int dcoords[3];
    for (int p = 0; p < 3; p++) dcoords[p] = mc.second[p];
    for (int j = 0; j < 3; j++) assign_check(c.coords[j], c.coords[j], 1<<s, dcoords[j]);
    mark_leaf_node(fine::active_nodes[root]);
    int qh = 0, qt;
    if (rearrange::flatten_nodes[info_offset] != 0) {
        nodes_queue.resize(1);
        nodes_queue[0] = root;
        qt = 0;
    }
    else {
        qt = -1;
    }
    int cur_subtree_st = subtree_st;
    while (qh <= qt) {
        auto front = nodes_queue[qh];
        bool split_time;
        if (front == root) split_time = rearrange::flatten_nodes[info_offset] == 1;
        else split_time = rearrange::flatten_nodes[info_offset - subtree_st + 1 + front] == 1;
        assert(cur_subtree_st < subtree_ed);
        int n_child = split_node(fine::active_nodes, front, split_time, 1, cur_subtree_st);
        for (int ii = 0; ii < n_child; ii++) {
            if (rearrange::flatten_nodes[info_offset - subtree_st + 1 + cur_subtree_st + ii] != 0) {
                nodes_queue.resize(++qt + 1);
                nodes_queue[qt] = cur_subtree_st + ii;
            }
            else mark_leaf_node(fine::active_nodes[cur_subtree_st + ii]);
        }
        qh++;
        cur_subtree_st += n_child;
    }
}

// common function to output vertices to Python to query indivisual elements
void output_vertices_common(node &nii, int L0, int maxL, int g, int i) {
    using namespace rearrange;
    using namespace dual_contouring;
    auto &vertices_g = vertices[g];
    // only leaf node is a hypercube
    if (leaf_node(nii)) {
        int mask = (1 << (nii.c.L - L0)) - 1;
        int offset = maxL - nii.c.L;
        vertex v;
        v.L = nii.c.L;
        int *coords = nii.c.coords;
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++) {
                    v.coords[0] = ((coords[0] & mask) + x) << offset;
                    v.coords[1] = ((coords[1] & mask) + y) << offset;
                    v.coords[2] = ((coords[2] & mask) + z) << offset;
                    int index = cube_index(v.coords[0], v.coords[1], v.coords[2], (1<<(maxL-L0)) + 1);
                    auto &vi = INDEX(vertices_g, index);
                    if (!vi) {
                        vi = 1;
                        assign_check(v.coords[0], coords[0], 1, x);
                        assign_check(v.coords[1], coords[1], 1, y);
                        assign_check(v.coords[2], coords[2], 1, z);
                        output_vertices_buffer[g].push_back(v);
                        output_vertices_index_buffer[g].push_back(mp(i, index));
                    }
                }
    }
}

// These are functions exposed to Python
extern "C" {
    // Load the active group t, if it is the first time query_vertices is true and it returns to python the number of vertices to query individual occupancy;
    // For later calls (see Alg.2 in the paper), query_vertices is false and it just loads the tree into memory and compute neighbors
    int load_group(int T, bool query_vertices, bool compute_inview_tag) {
        using namespace rearrange;
        using namespace dual_contouring;
        auto &active_nodes = fine::active_nodes;
        auto &mask = fine::mask;
        t_group = T;
        int n;
        MEASURE_TIME("load_group part 1", 1, {
            output_vertices.clear();
            output_vertices_index.clear();
            active_nodes.clear();
            active_nodes.push_back(nodes[0]);
            mark_leaf_node(active_nodes[0]);
            mask.resize(1);
            load_group_medium(nodes, 0, 0, T);
        });
        int max_load_id = 0;
        // visit the tree up to the medium level
        MEASURE_TIME("load_group part 2", 1, {
            n = active_nodes.size();
            node_labels.resize(n);
            node_labels.fill(-1);
            for (int i = 0; i < n; i++) {
                if (leaf_node(active_nodes[i])) {
                    if (mask[i]) {
                        int l = get_node_label(active_nodes[i]);
                        node_labels[i] = l;
                        if (l < 0 && query_vertices) {
                            // only output vertices if query_vertices is true
                            vertex v;
                            v.L = active_nodes[i].c.L;
                            int *coords = active_nodes[i].c.coords;
                            for (int x = 0; x < 2; x++)
                                for (int y = 0; y < 2; y++)
                                    for (int z = 0; z < 2; z++) {
                                        assign_check(v.coords[0], coords[0], 1, x);
                                        assign_check(v.coords[1], coords[1], 1, y);
                                        assign_check(v.coords[2], coords[2], 1, z);
                                        output_vertices.push_back(v);
                                    }
                        }
                    }
                }
            }
        });
        // prealocate space for the tree so later we can unzip the fine nodes in parallel
        MEASURE_TIME("load_group part 3", 1, {
            subtree_st.resize(n);
            subtree_st.fill(0);
            subtree_ed.resize(n);
            subtree_ed.fill(0);
            for (int g = 0; g < N_THREAD; g++) {
                output_vertices_buffer[g].clear();
                output_vertices_index_buffer[g].clear();
            }
            int local_nodes_size = active_nodes.size();
            for (int i = 0; i < n; i++) {
                if (leaf_node(active_nodes[i])) {
                    if (mask[i]) {
                        int l = node_labels[i];
                        if (l >= 0) {
                            subtree_st[i] = local_nodes_size;
                            subtree_ed[i] = local_nodes_size = subtree_st[i] + fine::visible_cubes_nodes_size[l] - 1;
                        }
                    }
                }
            }
            active_nodes.resize(local_nodes_size);
            mask.resizefill(local_nodes_size, 1);
            load_rearranged_fine_nodes(T);
        });
        // expand the tree in parallel
        MEASURE_TIME("load_group part 4", 1, {
            if (compute_inview_tag) {
                assert(inview_tag.empty());
                inview_tag.resizefill(active_nodes.size(), 0);
            }
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                int start_g = (n + N_THREAD - 1) / N_THREAD * g;
                int end_g = smin(n, (n + N_THREAD - 1) / N_THREAD * (g+1));
                for (int i = start_g; i < end_g; i++) {
                    if (leaf_node(active_nodes[i])) {
                        if (mask[i]) {
                            int l = node_labels[i];
                            if (l >= 0) {
                                unzip_nodes(i, subtree_st[i], subtree_ed[i], l, nodes_queue[g]);
                                if (compute_inview_tag) {
                                    inview_tag[i] = 1;
                                    for (int ii = subtree_st[i]; ii < subtree_ed[i]; ii++) inview_tag[ii] = 1;
                                }
                                // only output vertices if query_vertices is true
                                if (!query_vertices) continue;
                                lT maxL = 0;
                                auto L0 = active_nodes[i].c.L;
                                if (leaf_node(active_nodes[i])) {
                                    maxL = smax(maxL, L0);
                                }
                                for (int ii = subtree_st[i]; ii < subtree_ed[i]; ii++) {
                                    if (compute_inview_tag) inview_tag[ii] = 1;
                                    auto &nii = INDEX(active_nodes, ii);
                                    if (leaf_node(nii)) {
                                        maxL = smax(maxL, nii.c.L);
                                    }
                                }
                                auto &vertices_g = vertices[g];
                                vertices_g.resize(cubex((1<<(maxL-L0)) + 1));
                                vertices_g.fill(0);
                                // check both the current node and the subtree nodes, but note only leaf nodes are hypercubes
                                output_vertices_common(active_nodes[i], L0, maxL, g, i);
                                for (int ii = subtree_st[i]; ii < subtree_ed[i]; ii++) {
                                    auto &nii = INDEX(active_nodes, ii);
                                    output_vertices_common(nii, L0, maxL, g, i);
                                }
                            }
                        }
                    }
                }
            }
            for (int g = 0; g < N_THREAD; g++) {
                output_vertices.insert(output_vertices.end(), output_vertices_buffer[g].begin(), output_vertices_buffer[g].end());
                output_vertices_index.insert(output_vertices_index.end(), output_vertices_index_buffer[g].begin(), output_vertices_index_buffer[g].end());
            }
        });
        // log memory usage at potential peak locations
        PRINT_RSS_MEMORY("");
        // collect vertices to output to Python
        for (int g = 0; g < N_THREAD; g++) {
            CLS(output_vertices_buffer[g]);
            CLS(output_vertices_index_buffer[g]);
        }
        if (query_vertices) return output_vertices.size();
        else return 0;
    }
    
    // It returns to python the vertices to query individual occupancy functions
    void load_active_group_output(T *xyz, int vstart, int vend) {
        using namespace params;
        using namespace dual_contouring;
        MEASURE_TIME("load_active_group_output part 1", 1, {
            for (int i = vstart; i < vend; i++) {
                auto output_vertex = output_vertices.front();
                output_vertices.pop_front();
                compute_coords(xyz + (i-vstart)*3, output_vertex.coords, output_vertex.L);
            }
        });
    }    
}

// a cache mechanism can be potentially used, these function decide which group to offload
// but we do not use it in the current implementation
int min_priority_t0(map<int, int> &priority) {
    using namespace dual_contouring;
    int min_prio=1e9, min_t0;
    for (auto &tp: priority) {
        if (tp.second < min_prio) {
            min_prio = tp.second;
            min_t0 = tp.first;
        }
    }
    return min_t0;
}

int max_priority(map<int, int> &priority) {
    using namespace dual_contouring;
    int max_prio=0;
    for (auto &tp: priority) {
        max_prio = smax(max_prio, tp.second);
    }
    return max_prio;
}

int get_nxt_slot(int *occupied) {
    for (int i = 0; i < CACHECNT; i++) {
        if (occupied[i] == 0) {
            occupied[i] = 1;
            return i;
        }
    }
    return -1;
}

// unify the edge coordinates to compare edges easily
void simplify_edge(edge &e) {
    while ((e.tcoord & 1) == 0 && e.tL > 0) {
        e.tcoord >>= 1;
        e.tL--;
    }
}

// given a leaf node, this function finds bipolar edges out of its 12 potential edges
void find_edges(node &nd, int maxL, int L0, vec<int, int> &vertices, sdeque<edge> &local_edges, int outer_i) {
    using namespace dual_contouring;
    using namespace dual_contouring;
    int ne = params::n_elements;
    if (leaf_node(nd)) {
        int ss = 1 << (maxL-L0);
        int mask = (1 << (nd.c.L - L0)) - 1, offset = maxL - nd.c.L;
        int *coords = nd.c.coords;
        for (int e = 0; e < 12; e++) {
            int vcoords[3], e0 = e / 4, e1 = (e/4+1) % 3, e2 = (e/4+2) % 3;
            vcoords[e0] = ((coords[e0] & mask) + 1) << offset;
            vcoords[e1] = ((coords[e1] & mask) + (e&1)) << offset;
            vcoords[e2] = ((coords[e2] & mask) + ((e>>1)&1)) << offset;
            int index1 = cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1);
            vcoords[e0] = ((coords[e0]) & mask) << offset;
            int index2 = cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1);
            for (int ele = 0; ele < ne; ele++) {
                int sign1 = INDEX(vertices, index1 * ne + ele);
                int sign2 = INDEX(vertices, index2 * ne + ele);
                if (sign1 != sign2) {
                    edge ed;
                    ed.coords[e0] = coords[e0];
                    assign_check(ed.coords[e1], coords[e1], 1, e&1);
                    assign_check(ed.coords[e2], coords[e2], 1, (e>>1)&1);
                    ed.L = nd.c.L;
                    ed.dir = sign1 == 0? (e0 + 1): (-e0 - 1);
                    ed.ele = ele;
                    ed.tL = nd.c.tL;
                    ed.tcoord = nd.c.tcoord;
                    int id1, id2;
                    id1 = local_edges.size();
                    local_edges.push_back(ed);
                    simplify_edge(local_edges[id1]);
                }
            }
        }
    }
}

// write the dual graph of group t to disk
void write_graph(int t) {
    using namespace dual_contouring;
    int slot = cache_map[t];
    auto &head = head_cache[slot];
    auto &nxt = nxt_cache[slot];
    auto &edge_pointer = edge_pointer_cache[slot];
    std::stringstream filename;
    filename << params::output_path << "/graphs/" << t << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    write_vec(outfile, head);
    write_vec(outfile, nxt);
    write_vec(outfile, edge_pointer);
    fclose(outfile);
}

// load the dual graph of group t from disk
void load_graph(int t) {
    using namespace dual_contouring;
    std::stringstream filename;
    if (cache_map.count(t) == 0) {
        if (cache_map.size() == CACHECNT && CACHECNT > 0) {
            // offload the current graph to disk
            int min_t0 = min_priority_t0(priority);
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "offloading graph %d\n", min_t0);
            fclose(log);
            write_graph(min_t0);
            priority.erase(min_t0);
            occupied[cache_map[min_t0]] = 0;
            cache_map.erase(min_t0);
        }
        filename << params::output_path << "/graphs/" << t << ".bin";
        FILE *infile = fopen(filename.str().c_str() , "rb" );
        int slot = get_nxt_slot(occupied);
        cache_map[t] = slot;
        head_cache[slot].clear();
        nxt_cache[slot].clear();
        edge_pointer_cache[slot].clear();
        head_ = &head_cache[slot];
        nxt_ = &nxt_cache[slot];
        edge_pointer_ = &edge_pointer_cache[slot];
        auto &head = *head_;
        auto &nxt = *nxt_;
        auto &edge_pointer = *edge_pointer_;
        if (infile != NULL) {
            CLS(head);
            CLS(nxt);
            CLS(edge_pointer);
            read_vec(infile, head);
            read_vec(infile, nxt);
            read_vec(infile, edge_pointer);
            fclose(infile);
        }
        else {
            auto cnt_v = (long long)rearrange::tree_sizes[t] * params::n_elements;
            CLS(head);
            head.resizefill(cnt_v, -1);
            CLS(nxt);
            CLS(edge_pointer);
        }
    }
    else {
        int slot = cache_map[t];
        head_ = &head_cache[slot];
        nxt_ = &nxt_cache[slot];
        edge_pointer_ = &edge_pointer_cache[slot];
    }
    priority[t] = max_priority(priority) + 1;
}

// write the edges array of group t to disk
void write_bip_edges(int t) {
    using namespace dual_contouring;
    if (edges_fp_cache[cache_map2[t]] != NULL) {
        fclose(edges_fp_cache[cache_map2[t]]);
        edges_fp_cache[cache_map2[t]] = NULL;
        return;
    }
    auto &edges = edges_cache[cache_map2[t]];
    std::stringstream filename;
    filename << params::output_path << "/bip_edges/" << t << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    write_vec_headless(outfile, edges);
    fclose(outfile);
}

// load the edges array of group t from disk
void load_bip_edges(int t, int to_memory) {
    using namespace dual_contouring;
    if (cache_map2.count(t) == 0) {
        if (cache_map2.size() == CACHECNT && CACHECNT > 0) {
            // offload the current edges to disk
            int min_t0 = min_priority_t0(priority2);
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "offloading bip_edges %d\n", min_t0);
            fclose(log);
            write_bip_edges(min_t0);
            priority2.erase(min_t0);
            occupied2[cache_map2[min_t0]] = 0;
            cache_map2.erase(min_t0);
        }
        int slot = get_nxt_slot(occupied2);
        cache_map2[t] = slot;
        edges_ = &edges_cache[slot];
        auto &edges = *edges_;
        std::stringstream filename;
        filename << params::output_path << "/bip_edges/" << t << ".bin";
        CLS(edges);
        if (to_memory) {
            FILE *infile = fopen(filename.str().c_str() , "rb" );
            if (infile != NULL) {
                read_vec_headless(infile, edges);
                fclose(infile);
            }
            edges_fp_cache[slot] = NULL;
        }
        else {
            FILE *f = edges_fp_cache[slot] = fopen(filename.str().c_str() , "ab");
            edges_fp_ = &edges_fp_cache[slot];
            auto fileSize = std::filesystem::file_size(filename.str());
            if (fileSize == 0) {
                edges_size_cache[slot] = 0;
            }
            else {
                edges_size_cache[slot] = fileSize / sizeof(queriedEdge);
            }
            edges_size_ = &edges_size_cache[slot];
            assert(*edges_size_ >= 0);
        }
    }
    else {
        edges_ = &edges_cache[cache_map2[t]];
        edges_fp_ = &edges_fp_cache[cache_map2[t]];
        edges_size_ = &edges_size_cache[cache_map2[t]];
        assert(*edges_size_ >= 0);
    }
    priority2[t] = max_priority(priority2) + 1;
}

// Write all dual vertices and bipolar edges to disk for group t
// Note that if a bipolar edge contains neighbors in group t, it is wrote to group t, so it can be wrote multiple times
void write_results(int t) {
    using namespace dual_contouring;
    using namespace dual_contouring;
    load_graph(t);
    load_bip_edges(t, 0);
    auto &head = *head_;
    auto &nxt = *nxt_;
    auto &edge_pointer = *edge_pointer_;
    auto &edges_size = *edges_size_;
    auto &edges_fp = *edges_fp_;
    for (int ide = 0; ide < finalized_edges.size(); ide++) {
        auto &de = finalized_edges[ide];
        bool flag = 0;
        for (int i = 0; i < 8; i++)
            if (de.vertices_gid[i] == t) {
                flag = 1;
                break;
            }
        if (flag) {
            fwrite(&de, sizeof(queriedEdge), 1, edges_fp);
            edges_size++;
        }
        int edge_p = edges_size - 1;
        set<int> vset;
        for (int i = 0; i < 8; i++)
            if (de.vertices_gid[i] == t) {
                int v = de.vertices_nid[i];
                if (vset.count(v)) continue;
                vset.insert(v);
                ll id = (ll)v * params::n_elements + de.e.ele;
                int new_cnt = nxt.size();
                nxt.push_back(head[id]);
                assert(edge_p >= 0);
                edge_pointer.push_back(edge_p);
                head[id] = new_cnt;
            }
    }
}

// clean the graph and edge array if not needed
void clean_cache() {
    using namespace dual_contouring;
    for (auto t: priority2) {
        write_bip_edges(t.first);
    }
    for (auto t: priority) {
        write_graph(t.first);
    }
    priority.clear();
    cache_map.clear();
    priority2.clear();
    cache_map2.clear();
    for (int i = 0; i < CACHECNT; i++) {
        occupied[i] = 0;
        occupied2[i] = 0;
        CLS(head_cache[i]);
        CLS(nxt_cache[i]);
        CLS(edge_pointer_cache[i]);
        CLS(edges_cache[i]);
    }
}

// function to compare and make the bipolar edge unique because different adjacent hypercubes may generate the same edge
bool compareEdge(const edge &a, const edge &b) {
    for (int i = 0; i < 3; i++) {
        if (a.coords[i] < b.coords[i]) return 1;
        if (a.coords[i] > b.coords[i]) return 0;
    }
    if (a.L < b.L) return 1;
    if (a.L > b.L) return 0;
    if (a.tcoord < b.tcoord) return 1;
    if (a.tcoord > b.tcoord) return 0;
    if (a.tL < b.tL) return 1;
    if (a.tL > b.tL) return 0;
    if (a.dir < b.dir) return 1;
    if (a.dir > b.dir) return 0;
    return a.ele < b.ele;
}

bool compareQueriedEdge(const queriedEdge &a, const queriedEdge &b) {
    return compareEdge(a.e, b.e);
}

bool equalEdge(const edge &a, const edge &b) {
    for (int i = 0; i < 3; i++) {
        if (a.coords[i] != b.coords[i]) return 0;
    }
    if (a.L != b.L) return 0;
    if (a.tcoord != b.tcoord) return 0;
    if (a.tL != b.tL) return 0;
    if (a.dir != b.dir) return 0;
    return a.ele == b.ele;
}

// sort the unfinalized edges array based on existing sorted part and new ones
void sort_unfinalized_edges(int merge_only=0) {
    using namespace dual_contouring;
    if (!merge_only) {
        sort(unfinalized_edges.begin() + unfinalized_edges_sorted_cnt, unfinalized_edges.end(), compareQueriedEdge);
    }
    std::inplace_merge(unfinalized_edges.begin(), unfinalized_edges.begin() + unfinalized_edges_sorted_cnt, unfinalized_edges.begin() + unfinalized_edges.size(), compareQueriedEdge);
    unfinalized_edges_sorted_cnt = unfinalized_edges.size();
}

// These are functions exposed to Python
extern "C" {
    // see Alg.2 in the paper, this corresponds to line 5 - 12
    // This function calls load_group (with query_vertices being false) and the second part of itself several times to compute the bipolar edge neighbors
    // To clarify, when it is called the first time, it finds bipolar edges and compute neighbors based on occupancy values
    // later it calls (only the second part of) itself without the occupancy values and only computes neighbors
    void constructing_meshes(sdfT *sdf, int debug) {
        using namespace dual_contouring;
        auto &active_nodes = fine::active_nodes;
        auto &mask = fine::mask;
        int ne = params::n_elements;
        // back_eliminating means it is not the first time
        bool back_eliminating = sdf == NULL;
        long long cnt = 0;
        if (!back_eliminating) {
            MEASURE_TIME("constructing_meshes part 1", 1, {
                // with back_eliminating being false, pending edges means those edges discovered from occupancy values
                CLS(pending_edges);
                CLS(propagated_edges);
                CLS(finalized_edges);
                for (int i = 0; i < node_labels.size(); i++) {
                    // first, find bipolar edges in those medium nodes without further subdivision
                    if (leaf_node(active_nodes[i])) {
                        if (mask[i]) {
                            int l = node_labels[i];
                            if (l < 0) {
                                auto nd = active_nodes[i];
                                std::vector<int> vertices(8 * ne);
                                for (int index = 0; index < 8; index++) {
                                    for (int ele = 0; ele < ne; ele++)
                                        vertices[index * ne + ele] = sdf[cnt++] >= 0;
                                }
                                for (int e = 0; e < 12; e++) {
                                    int vcoords[3];
                                    int e0 = e / 4;
                                    int e1 = (e/4+1) % 3;
                                    int e2 = (e/4+2) % 3;
                                    vcoords[e0] = 1;
                                    vcoords[e1] = e&1;
                                    vcoords[e2] = (e>>1)&1;
                                    int index1 = cube_index(vcoords[0], vcoords[1], vcoords[2], 2);
                                    vcoords[e0] = 0;
                                    int index2 = cube_index(vcoords[0], vcoords[1], vcoords[2], 2);
                                        for (int ele = 0; ele < ne; ele++) {
                                            int sign1 = INDEX(vertices, index1 * ne + ele);
                                            int sign2 = INDEX(vertices, index2 * ne + ele);
                                            if (sign1 != sign2) {
                                                edge ed;
                                                ed.coords[e0] = nd.c.coords[e0];
                                                assign_check(ed.coords[e1], nd.c.coords[e1], 1, e&1);
                                                assign_check(ed.coords[e2], nd.c.coords[e2], 1, (e>>1)&1);
                                                ed.L = nd.c.L;
                                                ed.dir = sign1 == 0? (e0 + 1): (-e0 - 1);
                                                ed.ele = ele;
                                                ed.tL = nd.c.tL;
                                                ed.tcoord = nd.c.tcoord;
                                                pending_edges.push_back(ed);
                                                simplify_edge(pending_edges[(int)pending_edges.size() - 1]);
                                            }
                                        }
                                }
                            }
                        }
                    }
                }
            });
            // get pointers to point to the queried occupancy values grouped by medium nodes
            pointers.resize(node_labels.size() + 1);
            pointers[0].first = output_vertices_index.begin();
            pointers[0].second = 0;
            MEASURE_TIME("constructing_meshes part 2", 1, {
                for (int g = 0; g < N_THREAD; g++) {
                    pending_edges_buffer[g].clear();
                }
                for (int i = 0; i < node_labels.size(); i++) {
                    pointers[i+1] = pointers[i];
                    if (node_labels[i] >= 0) {
                        while (pointers[i+1].first != output_vertices_index.end() && pointers[i+1].first->first == i) {
                            pointers[i+1].first++;
                            pointers[i+1].second++;
                        }
                    }
                }
                // find bipolar edges in those medium nodes with further subdivision in parallel
                OMP_PRAGMA(omp parallel for schedule(dynamic))
                for (int g = 0; g < N_THREAD; g++) {
                    int step = (node_labels.size() + N_THREAD - 1) / N_THREAD;
                    int start_g = step * g; 
                    int end_g = smin(start_g + step, (int)node_labels.size());
                    for (int i = start_g; i < end_g; i++) {
                        if (node_labels[i] >= 0) {
                            auto maxL = active_nodes[i].c.L;
                            for (int ii = subtree_st[i]; ii < subtree_ed[i]; ii++) {
                                if (leaf_node(active_nodes[ii])) {
                                    maxL = smax(maxL, active_nodes[ii].c.L);
                                }
                            }
                            int L0 = active_nodes[i].c.L;
                            vertices[g].clear();
                            vertices[g].resizefill(cubex((1<<(maxL-L0)) + 1) * ne, -1);
                            auto p1 = pointers[i].first;
                            auto p2 = pointers[i].second;
                            for (; p1 != pointers[i+1].first; p1++, p2++) {
                                for (int ele = 0; ele < ne; ele++) {
                                    INDEX(vertices[g], p1->second * ne + ele) = sdf[cnt + p2 * ne + ele] >= 0;
                                }
                            }
                            find_edges(active_nodes[i], maxL, L0, vertices[g], pending_edges_buffer[g], i);
                            for (int ii = subtree_st[i]; ii < subtree_ed[i]; ii++) {
                                find_edges(active_nodes[ii], maxL, L0, vertices[g], pending_edges_buffer[g], i);
                            }
                        }
                    }
                }
                for (int g = 0; g < N_THREAD; g++) {
                    pending_edges.insert(pending_edges.end(), pending_edges_buffer[g].begin(), pending_edges_buffer[g].end());
                    PRINT_RSS_MEMORY("");
                    CLS(pending_edges_buffer[g]);
                }
            });
            // initialize absorbed flags (i.e. whether an unfinalized edge is already in pending edges)
            MEASURE_TIME("constructing_meshes part 3", 1, {
                make_unique<edge>(pending_edges, compareEdge, equalEdge);
                absorbed.clear();
                for (int i = 0; i < unfinalized_edges.size(); i++)
                    absorbed.push_back(0);
                backref.clear();
                for (int i = 0; i < pending_edges.size(); i++)
                    backref.push_back(NULL);
            });
            // find whether an unfinalized edge is already in pending edges, set absorbed flag and back reference accordingly
            // The new unfinalized edges become the unabsorbed ones plus the new unfinalized pending edges
            int p1 = 0;
            int p2 = 0;
            sdeque<queriedEdge>::iterator unfinalized_edges_p1=unfinalized_edges.begin();
            sdeque<edge>::iterator pending_edges_p2=pending_edges.begin();
            sdeque<int>::iterator absorbed_p1=absorbed.begin();
            sdeque<queriedEdge*>::iterator backref_p2=backref.begin();
            MEASURE_TIME("constructing_meshes part 3b", 1, {
                while (p1 < unfinalized_edges.size() || p2 < pending_edges.size()) {
                    if (p1 < unfinalized_edges.size() && p2 < pending_edges.size() && equalEdge(unfinalized_edges_p1->e, *pending_edges_p2)) {
                        *absorbed_p1 = 1;
                        *backref_p2 = &(*unfinalized_edges_p1);
                        p1++;
                        unfinalized_edges_p1++;
                        absorbed_p1++;
                        p2++;
                        pending_edges_p2++;
                        backref_p2++;
                    }
                    else if (p2 >= pending_edges.size() || (p1 < unfinalized_edges.size() && compareEdge(unfinalized_edges_p1->e, *pending_edges_p2))) {
                        p1++;
                        unfinalized_edges_p1++;
                        absorbed_p1++;
                    }
                    else {
                        p2++;
                        pending_edges_p2++;
                        backref_p2++;
                    }
                }
            });

        }
        else {
            // with back_eliminating being true, pending edges are those unfinalized edges from the first visit
            MEASURE_TIME("constructing_meshes part 4", 1, {
                pending_edges.clear();
                for (auto &de: unfinalized_edges) {
                    pending_edges.push_back(de.e);
                }
                int l = unfinalized_edges.size();
                auto unfinalized_edges_p = unfinalized_edges.begin();
                absorbed.clear();
                for (int i = 0; i < l; i++) absorbed.push_back(1);
                backref.clear();
                for (int i = 0; i < l; i++) {
                    backref.push_back(&(*unfinalized_edges_p));
                    unfinalized_edges_p++;
                }
            });
            PRINT_RSS_MEMORY("after constructing_meshes part 4");
        }
        // allocate pending_edges_results according to the current pending edges
        MEASURE_TIME("constructing_meshes part 5", 1, {
            int ps = pending_edges.size();
            pending_edges_results.clear();
            auto pending_edges_p = pending_edges.begin();
            auto backref_p = backref.begin();
            for (int i = 0; i < ps; i++) {
                edgeRes tmp;
                tmp.e = *pending_edges_p;
                tmp.detailed_edge_p = *backref_p;
                pending_edges_results.push_back(tmp);
                pending_edges_p++;
                backref_p++;
            };
            PRINT_RSS_MEMORY("after pending_edges_results");
            CLS(backref);
            CLS(pending_edges);
        });
        // query the neighbors in parallel
        MEASURE_TIME("constructing_meshes part 6", 1, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                ll start_g = (pending_edges_results.size() + N_THREAD - 1) / N_THREAD * g;
                ll end_g = smin((pending_edges_results.size() + N_THREAD - 1) / N_THREAD * (g+1), pending_edges_results.size());
                if (start_g >= pending_edges_results.size()) continue;
                auto it=pending_edges_results.begin() + start_g;
                for (; it != pending_edges_results.begin() + end_g; it++) {
                    auto &ed = it->e;
                    bool finalized = 1;
                    bool invalid = 0;
                    int e0 = abs(ed.dir) - 1;
                    int e1 = (e0+1) % 3;
                    int e2 = (e0+2) % 3;
                    std::vector<int> dirs(3);
                    std::vector<int> results(16);
                    timeT &pending_edges_category = it->category;
                    timeT *pending_edges_propagation = &(it->propagation[0]);
                    queriedEdge *backref_p = it->detailed_edge_p;
                    pending_edges_category = 0;
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            for (int t = 0; t < 2; t++) {
                                dirs[e1] = i;
                                dirs[e2] = j;
                                int res = results[2*(i + j*2 + t*4)] = bipolar_edge_neighbor_search(ed.coords, ed.L, &dirs[0], ed.tcoord, ed.tL, t, e0, fine::active_nodes, fine::mask);
                                // a bipolar edge is finalized only if all its neighbors are loaded in the current group (?= 0) or its neighbor is the time boundary
                                finalized &= (res >= 0 || res == TBOUND);
                                // a bipolar edge is invalid if any of its neighbor is invalid (i.e., there are middle points found)
                                invalid |= res == NONE;
                            }
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++) {
                            pending_edges_propagation[i*4+j*2] = -1;
                            int index = &ed - &pending_edges[0];
                            if (backref_p != NULL && (backref_p)->vertices_nid[i + j*2 + 4] >= 0) continue;
                            int node_id = results[2*(i + j*2 + 4)];
                            // To propagate the bipolar edges (see Fig.16 in the paper), we find the time span of this neighbor node
                            // And translate the current edge along the time axis
                            if (node_id >= 0) {
                                pending_edges_propagation[i*4+j*2] = active_nodes[node_id].c.tcoord + 1;
                                pending_edges_propagation[i*4+j*2+1] = active_nodes[node_id].c.tL;
                            }
                        }
                    if (!finalized) {
                        if (!invalid) {
                            // note that though backref_p refers back to an unfinalized edge, but such unfinalized edge can still contain NONE neighbors
                            // (these are kept because they are propagated edges)
                            if (backref_p != NULL && (backref_p)->vertices_nid[0] == NONE) {
                                results[0] = NONE;
                            }
                            else {
                                for (int i = 0; i < 8; i++) {
                                    // If the neighbor is found to be in inactive groups
                                    // Look back to the referenced unfinalized edges to get the neighbor
                                    if (results[i * 2] == OTHER) {
                                        if (backref_p != NULL) {
                                            int *ref_results_nid = (backref_p)->vertices_nid;
                                            timeT *ref_results_gid = (backref_p)->vertices_gid;
                                            if (ref_results_nid[i] != OTHER) {
                                                results[i*2] = ref_results_nid[i];
                                                results[i*2 + 1] = ref_results_gid[i];
                                            }
                                        }
                                    }
                                    else {
                                        results[i*2 + 1] = t_group;
                                    }
                                }
                            }
                            finalized = 1;
                            for (int i = 0; i < 8; i++) finalized &= (results[i * 2] >= 0 || results[i * 2] == TBOUND);
                        }
                    }
                    else {
                        // For finalized edges, assign the group id
                        for (int i = 0; i < 8; i++) results[i*2 + 1] = t_group;
                    }
                    if (!invalid) {
                        if (finalized) {
                            // special offset 1<<params::max_tL is added to indicate that this vertex is from time boundary
                            bool any_tbound = 0;
                            for (int i = 0; i < 8; i++) any_tbound |= results[i*2] == TBOUND;
                            if (any_tbound) {
                                if (results[0] == TBOUND) {
                                    bool all_tbound = 1;
                                    for (int i = 1; i < 4; i++) all_tbound &= results[i*2] == TBOUND;
                                    assert(all_tbound);
                                    bool no_tbound = 1;
                                    for (int i = 4; i < 8; i++) no_tbound &= results[i*2] != TBOUND;
                                    assert(no_tbound);
                                    for (int i = 0; i < 4; i++) {
                                        results[i*2] = results[i*2 + 8];
                                        results[i*2 + 1] = (1<<params::max_tL) + results[i*2 + 9];
                                    }
                                }
                                else {
                                    bool all_tbound = 1;
                                    for (int i = 4; i < 8; i++) all_tbound &= results[i*2] == TBOUND;
                                    assert(all_tbound);
                                    bool no_tbound = 1;
                                    for (int i = 0; i < 4; i++) no_tbound &= results[i*2] != TBOUND;
                                    assert(no_tbound);
                                    for (int i = 4; i < 8; i++) {
                                        results[i*2] = results[i*2 - 8];
                                        results[i*2 + 1] = (1<<params::max_tL) * 2 + results[i*2 - 7];
                                    }
                                }
                            }
                            // category one means finalized edges
                            pending_edges_category = 1;
                            int *results_index_nid = &(it->vertices_nid[0]);
                            timeT *results_index_gid = &(it->vertices_gid[0]);
                            for (int i = 0; i < 8; i++) {
                                results_index_nid[i] = results[i*2];
                                assert(results[i*2+1] <= (1<<15)-1);
                                results_index_gid[i] = results[i*2+1];
                            }
                            // swap order of vertices according to the sign convention
                            if (ed.dir < 0) {
                                auto tmp = results_index_nid[1]; results_index_nid[1] = results_index_nid[2]; results_index_nid[2] = tmp;
                                tmp = results_index_gid[1]; results_index_gid[1] = results_index_gid[2]; results_index_gid[2] = tmp;
                                tmp = results_index_nid[5]; results_index_nid[5] = results_index_nid[6]; results_index_nid[6] = tmp;
                                tmp = results_index_gid[5]; results_index_gid[5] = results_index_gid[6]; results_index_gid[6] = tmp;
                            }
                        }
                        else {
                            invalid = 0;
                            for (int i = 0; i < 8; i++) invalid |= results[i * 2] == NONE;
                            if (!invalid) {
                                // category two means unfinalized edges
                                pending_edges_category = 2;
                                int *results_index_nid = &(it->vertices_nid[0]);
                                timeT *results_index_gid = &(it->vertices_gid[0]);
                                for (int i = 0; i < 8; i++) {
                                    results_index_nid[i] = results[i*2];
                                    assert(results[i*2+1] <= (1<<15)-1);
                                    results_index_gid[i] = results[i*2+1];
                                }
                            }
                        }
                    }
                }
            }
        });
        // sequential processing according to previously marked categories
        MEASURE_TIME("constructing_meshes part 7", 1, {
            auto it=pending_edges_results.begin();
            for (; it != pending_edges_results.end(); it++) {
                auto &ed = it->e;
                timeT *pending_edges_propagation = &(it->propagation[0]);
                timeT &pending_edges_category = it->category;
                int *results_index_nid = &(it->vertices_nid[0]);
                timeT *results_index_gid = &(it->vertices_gid[0]);
                // propagate the edge if needed
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++) {
                        if (pending_edges_propagation[i*4+j*2] >= 0) {
                            auto ed0 = ed;
                            ed0.tcoord = pending_edges_propagation[i*4+j*2];
                            ed0.tL = pending_edges_propagation[i*4+j*2+1];
                            simplify_edge(ed0);
                            propagated_edges.push_back(ed0);
                        }
                    }
                // finalize or add to unfinalized edges accordingly
                if (pending_edges_category == 1) {
                    queriedEdge de;
                    de.e = ed;
                    de.e.dir = abs(de.e.dir) - 1;
                    for (int i = 0; i < 8; i++) {
                        de.vertices_nid[i] = results_index_nid[i];
                        de.vertices_gid[i] = results_index_gid[i];
                    }
                    finalized_edges.push_back(de);
                }
                else if (pending_edges_category == 2) {
                    queriedEdge de;
                    de.e = ed;
                    for (int i = 0; i < 8; i++) {
                        de.vertices_nid[i] = results_index_nid[i];
                        de.vertices_gid[i] = results_index_gid[i];
                    }
                    unfinalized_edges.push_back(de);
                }
            }
            PRINT_RSS_MEMORY("after constructing_meshes part 7");
            CLS(pending_edges_results);
        });
        // sort the new unfinalized edges and remove absorbed ones
        int prev_size = unfinalized_edges.size();
        MEASURE_TIME("constructing_meshes part 8", 1, {
            sdeque<queriedEdge> unfinalized_edges_copy;
            sdeque<queriedEdge>::iterator unfinalized_edges_p = unfinalized_edges.begin();
            sdeque<int>::iterator absorbed_p = absorbed.begin();
            for (; unfinalized_edges_p != unfinalized_edges.end(); unfinalized_edges_p++) {
                if (absorbed_p == absorbed.end() || !(*absorbed_p)) {
                    unfinalized_edges_copy.push_back(*unfinalized_edges_p);
                }
                if (absorbed_p != absorbed.end()) absorbed_p++;
            }
            unfinalized_edges.clear();
            sswap(unfinalized_edges, unfinalized_edges_copy);
            unfinalized_edges_sorted_cnt -= prev_size - unfinalized_edges.size();
            CLS(absorbed);
        });
        PRINT_RSS_MEMORY("");
        MEASURE_TIME("constructing_meshes part 9", 1, {
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "sorting unfinalized_edges size %d\n", int(unfinalized_edges.size()));
            fclose(log);
            sort_unfinalized_edges(1);
        });
        PRINT_RSS_MEMORY("");
        // deduplicate propagated edges
        MEASURE_TIME("constructing_meshes part 10", 1, {
            make_unique<edge>(propagated_edges, compareEdge, equalEdge);
        });
        PRINT_RSS_MEMORY("");
        // merge propagated edges into unfinalized edges and sort them
        int us = unfinalized_edges.size();
        int p1 = 0, p2 = 0;
        sdeque<queriedEdge>::iterator unfinalized_edges_p1=unfinalized_edges.begin();
        sdeque<edge>::iterator propagated_edges_p2=propagated_edges.begin();
        MEASURE_TIME("constructing_meshes part 11", 1, {
            new_unfinalized_edges.clear();
            while (p1 < us || p2 < propagated_edges.size()) {
                if (p1 < us && p2 < propagated_edges.size() && equalEdge(unfinalized_edges_p1->e, *propagated_edges_p2)) {
                    p1++;
                    p2++;
                    unfinalized_edges_p1++;
                    propagated_edges_p2++;
                }
                else if (p2 == propagated_edges.size() || (p1 < us && compareEdge(unfinalized_edges_p1->e, *propagated_edges_p2))) {
                    p1++;
                    unfinalized_edges_p1++;
                }
                else {
                    queriedEdge de;
                    de.e = *propagated_edges_p2;
                    p2++;
                    propagated_edges_p2++;
                    for (int i = 0; i < 8; i++) {
                        de.vertices_nid[i] = OTHER;
                    }
                    new_unfinalized_edges.push_back(de);
                }
            }
            unfinalized_edges.insert(unfinalized_edges.end(), new_unfinalized_edges.begin(), new_unfinalized_edges.end());
            CLS(new_unfinalized_edges);
        });
        PRINT_RSS_MEMORY("");
        MEASURE_TIME("constructing_meshes part 12", 1, {
            CLS(propagated_edges);
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "sorting unfinalized_edges size %d\n", int(unfinalized_edges.size()));
            fclose(log);
            sort_unfinalized_edges();
        });
        PRINT_RSS_MEMORY("");
        // After the first visit, we iterate related previous groups (line 8 in Alg.2 in the paper) to compute remaining neighbors
        if (!back_eliminating) {
            int T0 = t_group;
            vec<int, int> restricted_set;
            restricted_set.clear();
            for (int i = t_group, j = 0; i != 0; i >>= 1, j++) {
                restricted_set.push_back((i-1) << j);
            }
            for (auto group: restricted_set) {
                load_group(group, 0, 0);
                constructing_meshes(NULL, 0);
            }
            // group t_group need to be loaded again because the first time only deals with new in-group edges;
            // the second reound deals with old edges that has neighbor in t_group
            load_group(T0, 0, 0);
            constructing_meshes(NULL, 0);
            PRINT_RSS_MEMORY("");
            CLS(active_nodes);
            CLS(output_vertices);
            CLS(output_vertices_index);            
            CLS(node_labels);
            CLS(info_offset_rearranged);
            CLS(subtree_st);
            CLS(subtree_ed);
            CLS(pointers);
            for (int i = 0; i < N_THREAD; i++) {
                CLS(vertices[i]);
                CLS(nodes_queue[i]);
            }
            PRINT_RSS_MEMORY("");
            // writing results to disk
            MEASURE_TIME("constructing_meshes part writing results", 1, {
                write_results(T0);
                for (auto group: restricted_set) {
                    write_results(group);
                }
                clean_cache();
                CLS(finalized_edges);
            });
        }
    }

    // Save unfinalized edges for later reuse for debug purpose (other data are already on the disk)
    void load_unfinalized_edges() {
        using namespace dual_contouring;
        std::stringstream filename;
        filename << params::output_path << "/unfinalized_edges.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb");
        read_deq(infile, unfinalized_edges);
        fclose(infile);
        unfinalized_edges_sorted_cnt = unfinalized_edges.size();
    }
    
    // Load unfinalized edges from previous saved file
    void save_unfinalized_edges() {
        using namespace dual_contouring;
        clean_cache();
        std::stringstream filename;
        filename << params::output_path << "/unfinalized_edges.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb");
        write_deq(outfile, unfinalized_edges);
        unfinalized_edges.clear();
        fclose(outfile);
    }
}