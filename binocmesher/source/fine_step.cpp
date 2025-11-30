#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"

// These are functions exposed to Python
extern "C" {
    // initalization work based on fine_group - the ACTUAL number of medium cubes per group
    void fine_init(int fine_group) {
        using namespace fine;
        visible_cubes_nodes_size.clear();
        assert(fine_group <= N_FINEGROUP);
        fine::fine_group = fine_group;
        flatten_nodes.clear();
        stored_size = 0;
        for (int j = 0; j < N_FINEGROUP; j++) {
            nodes_array[j].clear();
        }
    }
    
    // split medium cubes from start_mc to end_mc and flood to potential new medium cubes
    int fine_iteration(int start_mc, int end_mc) {
        fine::start_mc = start_mc;
        fine::end_mc = end_mc;
        using namespace fine;
        output_vertices.clear();
        output_vertices_index.clear();
        maxLs.resize(end_mc - start_mc);
        typedef pair<T, bool> tb;        
        MEASURE_TIME("fine_iteration part 1", 1, {
            for (int j = start_mc; j < end_mc; j++) {
                nodes_array[j - start_mc].clear();
            }
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int j = start_mc; j < end_mc; j++) {
                int j0 = j - start_mc;
                node n;
                mark_leaf_node(n);
                int i0 = medium::visible_cubes[j].first;
                auto &nnvi = INDEX(coarse::nodes, INDEX(coarse::leaf_nodes_vector, i0));
                int s = grid_node_level(nnvi);
                auto tmp = medium::visible_cubes[j].second;
                int coords[3];
                for (int p = 0; p < 3; p++) coords[p] = tmp[p];
                n.c = nnvi.c;
                for (int p = 0; p < 3; p++) assign_check(n.c.coords[p], n.c.coords[p], 1<<s, coords[p]);
                n.c.L += s;
                auto ps = projected_size(n.c);
                auto &nq = nodes_queue[j0];
                int qh = 0;
                int qt;
                if (ps.first > 1) {
                    qt = 0;
                    nq.resize(1);
                    nq[0] = mp(0, ps);
                }
                else {
                    qt = -1;
                }
                auto &nodes = nodes_array[j0];
                nodes.push_back(n);
                // BFS split each subtree (instead of max-heap) since we want to expand all
                while (qh <= qt) {
                    auto front = nq[qh];
                    int ind = front.first;
                    T size = front.second.first;
                    bool split_time = front.second.second;
                    int i0 = nodes.size();
                    int n_child = split_node(nodes, ind, split_time);
                    for (int i = 0; i < n_child; i++) {
                        auto &ni = INDEX(nodes, i0 + i);
                        tb ps;
                        if (ni.c.L <= n.c.L + 1) {
                            ps = projected_size(ni.c);
                        }
                        else {
                            ps.first = size / 2;
                            ps.second = false;
                        }
                        if (ps.first > 1) {
                            qt++;
                            nq.resize(qt+1);
                            nq[qt] = mp(i0 + i, ps);
                        }
                        else mark_leaf_node(ni);
                    }
                    qh++;
                }
            }
        });
        // Prepare query vertices to Python
        MEASURE_TIME("fine_iteration part 2", 1, {
            output_vertices.clear();
            output_vertices_index.clear();
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                output_vertices_buffer[g].clear();
                output_vertices_index_buffer[g].clear();
                vertices[g].clear();
            }
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                int start_g = start_mc + (end_mc - start_mc + N_THREAD - 1) / N_THREAD * g;
                int end_g = smin(start_mc + (end_mc - start_mc + N_THREAD - 1) / N_THREAD * (g+1), end_mc);
                for (int j = start_g; j < end_g; j++) {
                    int j0 = j - start_mc;
                    auto &nodes = nodes_array[j0];
                    lT maxL = 0;
                    for (int i = 0; i < nodes.size(); i++) {
                        auto &ni = INDEX(nodes, i);
                        if (leaf_node(ni)) {
                            maxL = smax(maxL, ni.c.L);
                        }
                    }
                    maxLs[j0] = maxL;
                    int L0 = nodes[0].c.L;
                    vertices[g].resize(cubex((1<<(maxL-L0)) + 1));
                    for (int i = 0; i < cubex((1<<(maxL-L0)) + 1); i++) {
                        vertices[g][i] = 0;
                    }
                    for (int i = 0; i < nodes.size(); i++) {
                        auto &ni = INDEX(nodes, i);
                        if (leaf_node(ni)) {
                            int mask = (1 << (ni.c.L - L0)) - 1;
                            int offset = maxL - ni.c.L;
                            vertex v;
                            v.L = ni.c.L;
                            int *coords = ni.c.coords;
                            for (int x = 0; x < 2; x++)
                                for (int y = 0; y < 2; y++)
                                    for (int z = 0; z < 2; z++) {
                                        v.coords[0] = ((coords[0] & mask) + x) << offset;
                                        v.coords[1] = ((coords[1] & mask) + y) << offset;
                                        v.coords[2] = ((coords[2] & mask) + z) << offset;
                                        int index = cube_index(v.coords[0], v.coords[1], v.coords[2], (1<<(maxL-L0)) + 1);
                                        auto &vi = INDEX(vertices[g], index);
                                        if (!vi) {
                                            vi = 1;
                                            assign_check(v.coords[0], coords[0], 1, x);
                                            assign_check(v.coords[1], coords[1], 1, y);
                                            assign_check(v.coords[2], coords[2], 1, z);
                                            output_vertices_buffer[g].push_back(v);
                                            output_vertices_index_buffer[g].push_back(mp(j, index));
                                        }
                                    }
                        }
                    }
                }
            }
            for (int g = 0; g < N_THREAD; g++) {
                output_vertices.insert_end(output_vertices_buffer[g]);
                output_vertices_index.insert_end(output_vertices_index_buffer[g]);
            }
        });
        // Output vertices number first to let Python preallocate
        return output_vertices.size();
    }

    // actual writing of output vertices to Python
    void fine_iteration_output(T *xyz) {
        using namespace params;
        using namespace fine;
        OMP_PRAGMA(omp parallel for schedule(dynamic))
        for (int i = 0; i < output_vertices.size(); i++) {
            auto &ovi = INDEX(output_vertices, i);
            compute_coords(xyz + i*3, ovi.coords, ovi.L);
        }
    }

}

// the compare and equal functions for used to sort and make the surface-intersecting edges (bipolar edges) unique
bool compare_be(const be &a, const be &b) {
    be a0 = a, b0 = b;
    if (a0[0] > a0[1]) sswap(a0[0], a0[1]);
    if (b0[0] > b0[1]) sswap(b0[0], b0[1]);
    if (a[0] < b[0]) return true;
    if (a[0] > b[0]) return false;
    return a[1] < b[1];
}
bool equal_be(const be &a, const be &b) {
    return (a[0] == b[0] && a[1] == b[1]) || (a[0] == b[1] && a[1] == b[0]);
}

// We write fine nodes to disk in a compressed way while we process fine nodes and flood in (propagate)
void write_fine_nodes(int id, vec<int, int> &data, int cnt, bool rearranged=0) {
    std::stringstream filename;
    if (rearranged) filename << params::output_path << "/rearranged_fine_nodes/" << id << ".bin";
    else filename << params::output_path << "/fine_nodes/" << id << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    vec<int, int> compressed_data;
    int buffer = 0;
    for (int i = 0; i < cnt; i++) {
        buffer |= data[i] << (2*(i%16));
        if (i % 16 == 15) {
            compressed_data.push_back(buffer);
            buffer = 0;
        }
    }
    compressed_data.push_back(buffer);
    write_vec(outfile, compressed_data);
    fclose(outfile);
    if (!rearranged) {
        for (int i = 0; i < data.size() - cnt; i++) {
            data.a[i] = data.a[i + cnt];
        }
        data._size -= cnt;
    }
}

// These are functions exposed to Python
extern "C" {
    // Given queried occupancy values from Python, flood to potential new medium cubes
    int fine_iteration_propagate(sdfT *sdf) {
        using namespace fine;
        int pointer1s[end_mc - start_mc], pointer2s[end_mc - start_mc];
        res_coll.clear();
        // organize output vertices by their medium cube
        MEASURE_TIME("fine_iteration_propagate part 1", 1, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int i = 0; i < output_vertices_index.size(); i++) {
                if (i == 0 || output_vertices_index[i].first != output_vertices_index[i-1].first) {
                    pointer1s[output_vertices_index[i].first - start_mc] = i;
                    if (i != 0) {
                        pointer2s[output_vertices_index[i-1].first - start_mc] = i;
                    }
                }
            }
            if (output_vertices_index.size() != 0) {
                pointer2s[output_vertices_index[output_vertices_index.size()-1].first - start_mc] = output_vertices_index.size();
            }
        });
        // spawn threads to process different medium cubes
        // the pointers gives the range of vertices to process
        typedef vec<int, medium_cube> vec_mc;
        MEASURE_TIME("fine_iteration_propagate part 2", 1, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                auto &res = res_coll_buffer[g];
                auto &edges = edges_buffer[g];
                res.clear();
                vertices[g].clear();
                int start_g = start_mc + (end_mc - start_mc + N_THREAD - 1) / N_THREAD * g;
                int end_g = smin(start_mc + (end_mc - start_mc + N_THREAD - 1) / N_THREAD * (g+1), end_mc);
                for (int j = start_g; j < end_g; j++) {
                    int j0 = j - start_mc;
                    auto &nodes = nodes_array[j0];
                    int maxL = maxLs[j0];
                    int L0 = nodes[0].c.L;
                    int ss = 1<<(maxL-L0);
                    vertices[g].resize(cubex((1<<(maxL-L0)) + 1));
                    for (int i = 0; i < cubex((1<<(maxL-L0)) + 1); i++) {
                        vertices[g][i] = 0;
                    }
                    for (int i = pointer1s[j0]; i < pointer2s[j0]; i++)
                        vertices[g][output_vertices_index[i].second] = sdf[i] >= 0;
                    edges.clear();
                    for (int i = 0; i < nodes.size(); i++) {
                        auto &ni = INDEX(nodes, i);
                        if (leaf_node(ni)) {
                            int mask = (1 << (ni.c.L - L0)) - 1;
                            int offset = maxL - ni.c.L;
                            for (int e = 0; e < 12; e++) {
                                int vcoords[3];
                                int e0 = e / 4;
                                int e1 = (e/4+1) % 3;
                                int e2 = (e/4+2) % 3;
                                int *coords = ni.c.coords;
                                vcoords[e0] = ((coords[e0] & mask) + 1) << offset;
                                vcoords[e1] = ((coords[e1] & mask) + (e&1)) << offset;
                                vcoords[e2] = ((coords[e2] & mask) + ((e>>1)&1)) << offset;
                                int index1 = cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1);
                                int sign1 = INDEX(vertices[g], index1);
                                vcoords[e0] = (coords[e0] & mask) << offset;
                                int index2 = cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1);
                                int sign2 = INDEX(vertices[g], index2);
                                if (sign1 != sign2) {
                                    int qcoords[3];
                                    int dirs[3];
                                    int qL = ni.c.L;
                                    qcoords[e0] = coords[e0];
                                    assign_check(qcoords[e1], coords[e1], 1, e&1);
                                    assign_check(qcoords[e2], coords[e2], 1, (e>>1)&1);
                                    be data;
                                    data[0] = index1;
                                    data[1] = index2;
                                    data[2] = qcoords[0];
                                    data[3] = qcoords[1];
                                    data[4] = qcoords[2];
                                    data[5] = qL;
                                    data[6] = ni.c.tcoord;
                                    data[7] = ni.c.tL;
                                    data[8] = e0;
                                    edges.push_back(data);
                                }
                            }
                        }
                    }
                    // make the surface-intersecting edges (bipolar edges) unique
                    sort(edges.begin(), edges.end(), compare_be);
                    edges._size = unique(edges.begin(), edges.end(), equal_be) - edges.begin();
                    int dirs[3];
                    for (int ie = 0; ie < edges.size(); ie++) {
                        int e1 = (edges[ie][8]+1) % 3;
                        int e2 = (edges[ie][8]+2) % 3;
                        for (int c = 0; c < 4; c++) {
                            dirs[e1] = c&1;
                            dirs[e2] = (c>>1)&1;
                            vec_mc res1;
                            // find new surface-intersecting medium cubes through fine edge neighbor search
                            edge_neighbor_search(&edges[ie][2], edges[ie][5], dirs, edges[ie][6], edges[ie][7], 1, edges[ie][8], coarse::nodes, coarse::nodemap, res1);
                            res.insert_end(res1);
                            edge_neighbor_search(&edges[ie][2], edges[ie][5], dirs, edges[ie][6], edges[ie][7], 0, edges[ie][8], coarse::nodes, coarse::nodemap, res1);
                            res.insert_end(res1);
                            edge_neighbor_search(&edges[ie][2], edges[ie][5], dirs, edges[ie][6] + 1, edges[ie][7], 1, edges[ie][8], coarse::nodes, coarse::nodemap, res1);
                            res.insert_end(res1);
                        }
                    }
                }
            }
        });
        // collect results from all threads
        MEASURE_TIME("fine_iteration_propagate part 3", 1, {
            for (int g = 0; g < N_THREAD; g++) {
                res_coll.insert_end(res_coll_buffer[g]);
            }
        });
        // get unique medium cubes in parallel through link list
        MEASURE_TIME("fine_iteration_propagate part 4", 1, {
            nxt.clear();
            head.resize(coarse::leaf_nodes_vector.size());
            head.fill(-1);
            for (int i = 0; i < res_coll.size(); i++) {
                int nindex = res_coll[i].first;
                nxt.push_back(head[nindex]);
                head[nindex] = nxt.size() - 1;
            }
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                int start_g = (coarse::leaf_nodes_vector.size() + N_THREAD - 1) / N_THREAD * g;
                int end_g = smin(coarse::leaf_nodes_vector.size(), (coarse::leaf_nodes_vector.size() + N_THREAD - 1) / N_THREAD * (g+1));
                int max_l = 0;
                for (int i = start_g; i < end_g; i++) {
                    max_l = smax(max_l, grid_node_level(coarse::nodes[coarse::leaf_nodes_vector[i]]));
                }
                max_l = 1 << max_l;
                auto &vertex_flag = vertex_flags[g];
                vertex_flag.resize(cubex(max_l));
                for (int i = 0; i < cubex(max_l); i++) vertex_flag[i] = -1;
                for (int i = start_g; i < end_g; i++) {
                    for (int j = head[i]; j != -1; j = nxt[j]) {
                        auto &subindex = res_coll[j].second;
                        int x = subindex[0];
                        int y = subindex[1];
                        int z = subindex[2];
                        int ci = cube_index(x, y, z, max_l);
                        auto &vfc = vertex_flag[ci];
                        if (vfc != i) {
                            res_coll[j].first = -1 - res_coll[j].first;
                            vfc = i;
                        }
                    }
                }
            }
        });
        // If the new medium cube is already in occluded set, reject, otherwise the propagation might be too much
        // Add new medium cubes to visible set
        MEASURE_TIME("fine_iteration_propagate part 5", 1, {
            for (int i = 0; i < res_coll.size(); i++) {
                auto &rescube = res_coll[i];
                if (rescube.first < 0) {
                    rescube.first = -1 - rescube.first;
                    if (!medium::occluded_set.count(rescube)) {
                        if (!medium::visible_set.count(rescube)) {
                            medium::visible_set.insert(rescube);
                            medium::visible_cubes.push_back(rescube);
                        }
                    }
                }
            }
        });
        int inc;
        using namespace medium;
        MEASURE_TIME("fine_iteration_propagate part 6", 1, {
            // compress the current subtrees
            for (int j = start_mc; j < end_mc; j++) {
                auto mc = visible_cubes[j];
                auto &nodes = nodes_array[j - start_mc];
                visible_cubes_nodes_size.push_back(nodes.size());
                for (int i = 0; i < nodes.size(); i++) {
                    int tag = leaf_node(nodes[i])? 0: (is_tsplit(nodes[i])? 1: 2);
                    flatten_nodes.push_back(tag);
                }
            }
            bool flag = visible_cubes.size() == current_size && end_mc == visible_cubes.size();
            bool finish = 0;
            // Write to disk for every fine_group medium cubes
            if (flag || (end_mc >= stored_size + fine_group)) {
                int new_stored_size = smin(stored_size + fine_group, current_size);
                finish = flag && new_stored_size == current_size;
                int s = 0;
                for (int i = stored_size; i < new_stored_size; i++) s += visible_cubes_nodes_size[i];
                write_fine_nodes(stored_size / fine_group * fine_group, flatten_nodes, s);
                stored_size  = new_stored_size;
            }
            // Sort the remaining medium cubes along with the new medium cubes
            if (end_mc + fine_group >= sorted_size) {
                sort(visible_cubes.begin() + sorted_size, visible_cubes.end());
                sorted_size = visible_cubes.size();
            }
            inc = finish? -1: (visible_cubes.size() - current_size);
            current_size = visible_cubes.size();
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "sorted_size: %d; visible_cubes.size(): %d\n", sorted_size, (int)visible_cubes.size());
            fclose(log);
        });
        return inc;
    }

    // Save all results for later reuse (for debug purpose)
    void fine_dump() {
        using namespace medium;
        using namespace fine;
        std::stringstream filename;
        filename << params::output_path << "/fine.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb" );
        write_vec(outfile, visible_cubes_nodes_size);
        write_vec(outfile, visible_cubes);
        fclose(outfile);
    }

    // Load previously saved results from disk
    void fine_load() {
        using namespace medium;
        using namespace fine;
        std::stringstream filename;
        filename << params::output_path << "/fine.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb" );
        read_vec(infile, visible_cubes_nodes_size);
        read_vec(infile, visible_cubes);
        fclose(infile);
    }
}

// framing_cubes are those medium cubes that are instantiated
// Compute hypercube coordinates given a medium cube, and add it to the framing_cubes
void framing_mc(medium_cube &mc, int i) {
    using namespace fine;
    using namespace rearrange;
    int ind = coarse::leaf_nodes_vector[mc.first];
    hypercube c = coarse::nodes[ind].c;
    int s = grid_node_level(coarse::nodes[ind]);
    c.L += s;
    int dcoords[3];
    for (int p = 0; p < 3; p++) dcoords[p] = mc.second[p];
    for (int j = 0; j < 3; j++) assign_check(c.coords[j], c.coords[j], 1<<s, dcoords[j]);
    framing_cubes.push_back(mp(c, mp(i, 0)));
}

// Given the range of framing_cubes [l, r), split the current subtree rooted at r_ind
void rearrange_nodes_func(int l, int r, int r_ind) {
    using namespace fine;
    using namespace rearrange;
    if (l == r) {
        mark_leaf_node(nodes[r_ind]);
        label_node(nodes[r_ind], -1);
        return;
    }
    bool split_time = true, split_space = false;
    auto &root = nodes[r_ind];
    // If there is any framing cube span the entire time range, we do not split in time
    for (int i = l; i < r; i++) {
        hypercube c = framing_cubes[i].first;
        if (c.tcoord == root.c.tcoord && c.tL == root.c.tL) {
            split_time = false;
            // If such framing cube exists, we just need to check its spatial extent
            split_space = c.L != root.c.L;
            for (int p = 0; p < 3; p++) if (c.coords[p] != root.c.coords[p]) split_space = true;
            break;
        }
    }
    if (split_time) {
        // If we split in time, we sort the framing _cubes in time
        int i = l, j = r - 1;
        split_node(nodes, r_ind, 1);
        auto &root = nodes[r_ind];
        int A = nodes[root.nxts[1]].c.tcoord, B = nodes[root.nxts[1]].c.tL;
        while (i <= j) {
            while (i < r && compare(framing_cubes[i].first.tcoord, framing_cubes[i].first.tL, A, B)) i++;
            while (j >= l && !compare(framing_cubes[j].first.tcoord, framing_cubes[j].first.tL, A, B)) j--;
            if (i < j) {
                sswap(framing_cubes[i], framing_cubes[j]);
                i++;
                j--;
            }
        }
        // recurse on the two time subtrees
        rearrange_nodes_func(l, i, root.nxts[0]);
        rearrange_nodes_func(i, r, nodes[r_ind].nxts[1]);
        nodes[r_ind].nxts[2] = -1;
    }
    else if (split_space) {
        // If we split in space, we sort the framing _cubes in the order of 8 octants
        split_node(nodes, r_ind, 0);
        auto &root = nodes[r_ind];
        for (int i = l; i < r; i++) {
            hypercube c = framing_cubes[i].first;
            int &o = framing_cubes[i].second.second;
            o = 0;
            for (int p = 0; p < 3; p++) o += (!compare(c.coords[p], c.L, nodes[root.nxts[7]].c.coords[p], root.c.L+1)) << p;
        }
        int i = l, j = r - 1;
        int last_o = -1;
        for (int o = 0; o < 7; o++) {
            while (i <= j) {
                while (i < r && framing_cubes[i].second.second == o) i++;
                while (j >= l && framing_cubes[j].second.second != o) j--;
                if (i < j) {
                    sswap(framing_cubes[i], framing_cubes[j]);
                    i++;
                    j--;
                }
            }
            rearrange_nodes_func(l, i, nodes[r_ind].nxts[o]);
            last_o = o;
            l = i;
            j = r - 1;
            if (i == r) break;
        }
        // recurse on octants
        for (int o = last_o + 1; o < 8; o++)
            rearrange_nodes_func(i, r, nodes[r_ind].nxts[o]);
    }
    else {
        #ifdef DEBUG
        assert(l == r - 1);
        assert(framing_cubes[l].second.first < 1e9 && framing_cubes[l].second.first >= -1);
        #endif
        mark_leaf_node(nodes[r_ind]);
        label_node(nodes[r_ind], framing_cubes[l].second.first);
    }
}

// These are functions exposed to Python
extern "C" {
    // rearrange step to instantiate and sort the visible medium cubes
    void rearrange_nodes() {
        using namespace medium;
        using namespace fine;
        using namespace rearrange;
        nodes.clear();
        nodes.push_back(coarse::nodes[0]);
        mark_leaf_node(nodes[0]);
        int vsize = (int)visible_cubes.size();
        MEASURE_TIME("rearrange_nodes part 2", 1, {
            framing_cubes.clear();
            int i = 0;
            for (auto &mc: visible_cubes) {
                framing_mc(mc, i++);
            }
            for (auto &mc: occluded_cubes) {
                framing_mc(mc, -1);
            }
        });
        MEASURE_TIME("rearrange_nodes part 3", 1, {
            rearrange_nodes_func(0, framing_cubes.size(), 0);
            visited_subtree.resize(nodes.size());
            group_id_start.resize(nodes.size());
            group_id_start.fill(-1);
            group_id_end.resize(nodes.size());
            group_id_end.fill(-1);
        });    
    }

}

// function to visit the tree of group id = T
void mark_group_id(vec<int, node> &nodes, int index, int T) {
    using namespace rearrange;
    // group_id_start to group_id_end is group id span, different from time span
    if (!visited_subtree[index])
        if (leaf_node(nodes[index])) {
            visited_subtree[index] = 1;
            group_id_start[index] = T;
            group_id_end[index] = T;
        }
        else {
            bool tsp = is_tsplit(nodes[index]);
            if (tsp) {
                int tcoord = nodes[index].c.tcoord * 2 + 1, tL = nodes[index].c.tL + 1;
                int t_index = T >= tcoord << (params::max_tL - tL);
                mark_group_id(nodes, nodes[index].nxts[t_index], T);
            }
            else {
                for (int i = 0; i < 8; i++)
                    mark_group_id(nodes, nodes[index].nxts[i], T);
            }
            int n_child = tsp? 2: 8;
            for (int i = 0; i < n_child; i++) {
                visited_subtree[index] &= visited_subtree[nodes[index].nxts[i]];
                if (group_id_start[nodes[index].nxts[i]] != -1) {
                    if (group_id_start[index] == -1) {
                        group_id_start[index] = group_id_start[nodes[index].nxts[i]];
                        group_id_end[index] = group_id_end[nodes[index].nxts[i]];
                    }
                    else {
                        group_id_start[index] = smin(group_id_start[index], group_id_start[nodes[index].nxts[i]]);
                        group_id_end[index] = smax(group_id_end[index], group_id_end[nodes[index].nxts[i]]);
                    }
                }
            }
        }
}

// These are functions exposed to Python
extern "C" {
    // visit in group id order for the first time
    void pre_tree_building(int t) {
        using namespace rearrange;
        mark_group_id(nodes, 0, t);
    }
}

// function to write the accumulated subtree sizes to disk after rearrangement
void write_info_offset_rearranged(int t) {
    using namespace rearrange;
    std::stringstream filename;
    filename << params::output_path << "/rearranged_nodemap/" << t << ".bin";
    FILE *outfile = fopen(filename.str().c_str() , "wb" );
    _sort(info_offset_rearranged_vec);
    write_vec(outfile, info_offset_rearranged_vec);
    fclose(outfile);
}

// We group fine nodes every fine_group medium cubes
// This function loads those of group id
void load_fine_nodes(int id) {
    using namespace medium;
    using namespace fine;
    vec<int, int> flatten_nodes_uncompressed;
    MEASURE_TIME("load_fine_nodes part 1", 0, {
        std::stringstream filename;
        filename << params::output_path << "/fine_nodes/" << id << ".bin";
        FILE *infile = fopen(filename.str().c_str() , "rb");
        read_vec(infile, flatten_nodes_uncompressed);
        fclose(infile);
        flatten_nodes.clear();
        for (int j = 0; j < flatten_nodes_uncompressed.size(); j++) {
            for (int i = 0; i < 16; i++) {
                flatten_nodes.push_back((flatten_nodes_uncompressed[j] >> (i*2)) & 3);
            }
        }
    });
    CLS(flatten_nodes_uncompressed);
    MEASURE_TIME("load_fine_nodes part 2", 0, {
        info_offset[0] = 0;
        for (int i = id; i+1 < smin(id + fine_group, int(visible_cubes.size())); i++) {
            info_offset[i+1-id] = info_offset[i-id] + visible_cubes_nodes_size[i];
        }
    });
}

// load the subtree of group id = T into active_nodes
// but up to medium nodes, later load_fine_nodes is called to load fine nodes
void load_group_medium(vec<int, node> &nodes, int index, int to_index, int T) {
    using namespace fine;
    using namespace rearrange;
    if (group_id_start[index] > T || group_id_end[index] < T) {
        mark_leaf_node(active_nodes[to_index]);
        mask[to_index] = 0;
    }
    else if (leaf_node(nodes[index])) {
        mark_leaf_node(active_nodes[to_index]);
        int l = get_node_label(nodes[index]);
        label_node(active_nodes[to_index], l);
        mask[to_index] = 1;
    }
    else {
        int tsp = is_tsplit(nodes[index]);
        int n_child = tsp? 2: 8;
        split_node(active_nodes, to_index, tsp);
        for (int i = 0; i < n_child; i++) mask.push_back(0);
        for (int i = 0; i < n_child; i++)
            load_group_medium(nodes, nodes[index].nxts[i], active_nodes[to_index].nxts[i], T);
    }
}

// These are functions exposed to Python
extern "C" {
    // rewrite the compressed node trees to disk
    void rearrange_fine_nodes(int t) {
        using namespace fine;
        using namespace rearrange;
        rearrange::flatten_nodes.clear();
        info_offset_rearranged_vec.clear();
        set<int> load_ids;
        int vsize = (int)medium::visible_cubes.size();
        // load the medium tree first
        MEASURE_TIME("rearrange_fine_nodes part 1", 1, {
            active_nodes.clear();
            active_nodes.push_back(nodes[0]);
            mark_leaf_node(active_nodes[0]);
            fine::mask.resize(1);
            load_group_medium(nodes, 0, 0, t);
        });
        int tree_size = active_nodes.size();
        // load the fine nodes from disk (stored in old order)
        MEASURE_TIME("rearrange_fine_nodes part 2", 1, {
            for (int i = 0; i < active_nodes.size(); i++) {
                if (leaf_node(active_nodes[i])) {
                    if (fine::mask[i]) {
                        int l = get_node_label(active_nodes[i]);
                        if (l >= 0) {
                            load_ids.insert(l / fine_group);
                            tree_size += fine::visible_cubes_nodes_size[l] - 1;
                        }
                    }
                }
            }
        });
        MEASURE_TIME("rearrange_fine_nodes part 3", 1, {
            for (auto load_id: load_ids) {
                load_fine_nodes(load_id * fine_group);
                for (int i = 0; i < active_nodes.size(); i++) {
                    if (leaf_node(active_nodes[i])) {
                        if (fine::mask[i]) {
                            int l = get_node_label(active_nodes[i]);
                            if (l >= 0 && l / fine_group == load_id) {
                                int st = fine::info_offset[l - load_id * fine_group];
                                int ed = st + fine::visible_cubes_nodes_size[l];
                                int tmp = rearrange::flatten_nodes.size();
                                rearrange::flatten_nodes.resize(tmp + ed - st);
                                std::copy(fine::flatten_nodes.begin() + st, fine::flatten_nodes.begin() + ed, rearrange::flatten_nodes.begin() + tmp);
                                info_offset_rearranged_vec.push_back(mp(l, tmp));
                            }
                        }
                    }
                }
            }
        });
        tree_sizes.resize(t+1);
        tree_sizes[t] = tree_size;
        // write the rearranged fine nodes to disk
        MEASURE_TIME("rearrange_fine_nodes part 4", 1, {
            write_fine_nodes(t, rearrange::flatten_nodes, rearrange::flatten_nodes.size(), 1);
            write_info_offset_rearranged(t);
        });
    }

    // write tree size for debug purpose
    void write_tree_size() {
        using namespace rearrange;
        std::stringstream filename;
        filename << params::output_path << "/tree_sizes.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb" );
        fwrite(&tree_sizes[0], sizeof(int), tree_sizes.size(), outfile);
        fclose(outfile);
    }

    // load tree size for debug purpose
    void load_tree_size() {
        using namespace rearrange;
        std::stringstream filename;
        filename << params::output_path << "/tree_sizes.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb");
        tree_sizes.resize(1 << params::max_tL);
        fread(&tree_sizes[0], sizeof(int), tree_sizes.size(), infile);
        fclose(infile);
    }

    // Save all results for later reuse (for debug purpose)
    void rearrange_dump() {
        using namespace rearrange;
        std::stringstream filename;
        filename << params::output_path << "/rearrange.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb" );
        write_vec(outfile, nodes);
        write_vec(outfile, group_id_start);
        write_vec(outfile, group_id_end);
        fclose(outfile);
    }

    // Load previously saved results from disk
    void rearrange_load() {
        using namespace rearrange;
        std::stringstream filename;
        filename << params::output_path << "/rearrange.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb" );
        read_vec(infile, nodes);
        read_vec(infile, group_id_start);
        read_vec(infile, group_id_end);
        fclose(infile);
    }

    // Clean up all fine and rearrange step related variables to save memory
    void fine_clean_up() {
        {
            using namespace medium;
            visible_set.clear();
            occluded_set.clear();
            CLS(occluded_cubes);
        }
        {
            using namespace fine;
            CLS(active_nodes);
            CLS(mask);
            CLS(output_vertices_index);
            CLS(output_vertices);
            CLS(res_coll);
            CLS(head);
            CLS(nxt);
            CLS(flatten_nodes);
            CLS(maxLs);
            for (int i = 0; i < N_FINEGROUP; i++) {
                CLS(nodes_array[i]);
                CLS(nodes_queue[i]);
            }
            for (int i = 0; i < N_THREAD; i++) {
                CLS(edges_buffer[i]);
                CLS(output_vertices_index_buffer[i]);
                CLS(output_vertices_buffer[i]);
                CLS(vertices[i]);
                CLS(res_coll_buffer[i]);
                CLS(vertex_flags[i]);
            }
        }
        {
            using namespace rearrange;
            CLS(framing_cubes);
            CLS(visited_subtree);
            CLS(flatten_nodes);
            CLS(info_offset_rearranged_vec);
        }
    }
}