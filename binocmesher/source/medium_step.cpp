#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"

// compare two medium_cubes based on their coarse node indices
bool compareCube(const medium_cube &a, const medium_cube &b) {
    return a.first < b.first;
}

// Same, but for medium_cubes with occupancy informations
bool compareCubeExt(const medium_cube_ext &a, const medium_cube_ext &b)
{
    return a.first.first < b.first.first;
}

// Each thread g has a boolean lookup table to check whether a medium_cube is visited
// For reuse purpose, we keep track of all nonzero entries
void medium_assign_cubes(int g, int ind) {
    using namespace medium;
    cubes[g][ind] = 1;
    cubes_nonzero[g].push_back(ind);
}

// We reset all visited medium_cubes for the next batch
void restore_cubes() {
    using namespace medium;
    OMP_PRAGMA(omp parallel for)
    for (int g = 0; g < N_THREAD; g++) {
        auto &cubes_nonzero_g = cubes_nonzero[g];
        for (int i = 0; i < cubes_nonzero_g.size(); i++) {
            cubes[g][cubes_nonzero_g[i]] = 0;
        }
        cubes_nonzero_g.clear();
    }
}

// Similarly, each thread g has an array to record occupancy values of cube vertices
// We also keep track of all nonzero entries for reuse
void restore_vertices() {
    using namespace medium;
    OMP_PRAGMA(omp parallel for)
    for (int g = 0; g < N_THREAD; g++) {
        auto &vertices_nonzero_g = vertices_nonzero[g];
        for (int i = 0; i < vertices_nonzero_g.size(); i++) {
            vertices[g][vertices_nonzero_g[i]] = -1;
        }
        vertices_nonzero_g.clear();
    }
}

// This is the common part of medium_iteration_init and medium_iteration_regular
// timing is whether to record time
int medium_iteration_common(int timing) {
    using namespace coarse;
    using namespace medium;
    // for thread g, it only considers intra-coarse-node flooding for nodes in [start_node[g], start_node[g+1])
    // if cubes_queue of all threads are empty, then we are done
    bool finish = 1;
    for (int g = 0; g < N_THREAD; g++) {
        if (!cubes_queue[g].empty()) finish = 0;
    }
    if (finish) return -1;
    // cleans up output vertices buffers to write to Python
    MEASURE_TIME("medium_iteration_common part 1", timing, {
        output_vertices.clear();
        output_vertices_index.clear();
        OMP_PRAGMA(omp parallel for schedule(dynamic))
        for (int g = 0; g < N_THREAD; g++) {
            output_vertices_buffer[g].clear();
            output_vertices_index_buffer[g].clear();
        }
    });
    // Given the list of medium cubes cubes_queue for each thread g
    // if the occupancy value of any vertex is unknown, write to output_vertices_buffer to give to Python
    MEASURE_TIME("medium_iteration_common part 2", timing, {
        OMP_PRAGMA(omp parallel for schedule(dynamic))
        for (int g = 0; g < N_THREAD; g++) {
            for (int j = 0; j < cubes_queue[g].size(); j++) {
                auto &cqj = INDEX(cubes_queue[g], j);
                int i = cqj.first;
                int nvi = INDEX(leaf_nodes_vector, i);
                auto &nnvi = INDEX(nodes, nvi);
                int s = grid_node_level(nnvi);
                int ss = 1<<s;
                auto cqjs = cqj.second;
                for (int dx = 0; dx < 2; dx++)
                for (int dy = 0; dy < 2; dy++)
                for (int dz = 0; dz < 2; dz++) {
                    int coords[3];
                    coords[0] = cqjs[0] + dx;
                    coords[1] = cqjs[1] + dy;
                    coords[2] = cqjs[2] + dz;
                    int ci = cube_index(coords[0], coords[1], coords[2], ss+1);
                    // ici is the total index of the vertex
                    // with vertices_index being the preprocessed base index for each coarse node
                    int ici = INDEX(vertices_index[g], i-start_node[g]) + ci;
                    int ind = (j << 3) + cube_index(dx, dy, dz, 2);
                    int cap;
                    int &vi = INDEX(vertices[g], ici);
                    if (vi == -1) {
                        vi = 0;
                        vertex v;
                        hypercube c = nnvi.c;
                        for (int p = 0; p < 3; p++) assign_check(v.coords[p], c.coords[p], 1<<s, coords[p]);
                        v.L = c.L + s;
                        output_vertices_buffer[g].push_back(v);
                        output_vertices_index_buffer[g].push_back(mp(g, ici));
                        vertices_nonzero[g].push_back(ici);
                    }
                }
            }
        }
    });
    // collect all vertices to output_vertices to return to Python
    MEASURE_TIME("medium_iteration_common part 3", timing, {
        for (int g = 0; g < N_THREAD; g++) {
            output_vertices.insert_end(output_vertices_buffer[g]);
            output_vertices_index.insert_end(output_vertices_index_buffer[g]);
        }
    });
    return output_vertices.size();
}

// These are functions exposed to Python
extern "C" {
    // Initialize medium_cubes seeds for the flooding algorithm
    int medium_seeding(int stride) {
        using namespace coarse;
        using namespace medium;
        int nvs = leaf_nodes_vector.size();
        vec<int, int> base(nvs + 1);
        base[0] = 0;
        for (int i = 0; i < nvs; i++) {
            int ind = INDEX(leaf_nodes_vector, i);
            int s = grid_node_level(INDEX(nodes, ind));
            assert(s >= 0);
            ll nb = base[i] + (sqr(((1 << s) + stride - 1) / stride) << s);
            assert(nb < INT_MAX);
            base[i+1] = nb;
        }
        cubes_entry.resize(base[nvs]);
        all_cubes.clear();
        all_cubes_size = 0;
        for (int g = 0; g < N_THREAD; g++) {
            cubes_entry_buffer[g].clear();
            all_cubes_buffer[g].clear();
            cubes_queue[g].clear();
            cubes[g].clear();
            vertices[g].clear();
            cubes_nonzero[g].clear();
            vertices_nonzero[g].clear();
            vertices_index[g].clear();
            cubes_index[g].clear();
            output_vertices_index_buffer[g].clear();
            output_vertices_buffer[g].clear();
        }
        // Initial seed cubes with sparse sampling in X-Y plane but dense Z coverage to ensure intersection with the surface
        OMP_PRAGMA(omp parallel for)
        for (int i = 0; i < nvs; i++) {
            int ind = INDEX(leaf_nodes_vector, i);
            int s = grid_node_level(INDEX(nodes, ind));
            int cnt = 0;
            for (int x = 0; x < 1 << s; x += stride)
                for (int y = 0; y < 1 << s; y += stride)
                    for (int z = 0; z < 1 << s; z++)
                        cubes_entry[base[i] + cnt++] = mp(i, make_array3(int8_t(x), int8_t(y), int8_t(z)));
        }
        return cubes_entry.size();
    }

    // The entire medium step is composed of multiple "medium_loop"s and "medium_iteration_end"s
    // medium_loop only propagates within each coarse node, while medium_iteration_end does the cross-node propagation
    // A medium_loop function (core.cpp) will call medium_iteration_init once, and call medium_iteration_output and medium_iteration_regular multiple times
    // See the core.py for the detailed logic
    int medium_iteration_init(int start_node, int end_node, int *update, int timing) {
        using namespace coarse;
        using namespace medium;
        int cubes_size[N_THREAD]={0}, vertices_size[N_THREAD]={0}, starts[N_THREAD], ends[N_THREAD];
        MEASURE_TIME("medium_iteration_init part 1", timing, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                // divide the coarse nodes among threads
                starts[g] = start_node + (end_node - start_node + N_THREAD - 1) / N_THREAD * g;
                ends[g] = smin(end_node, start_node + (end_node - start_node + N_THREAD - 1) / N_THREAD * (g + 1));
                medium::start_node[g] = starts[g];
                vertices_index[g].clear();
                cubes_index[g].clear();
                for (int i = starts[g]; i < ends[g]; i++) {
                    int ind = INDEX(leaf_nodes_vector, i);
                    int s = grid_node_level(INDEX(nodes, ind));
                    // get the base index of medium cubes within each coarse node
                    cubes_index[g].push_back(cubes_size[g]);
                    cubes_size[g] += cubex(1<<s);
                    // get the base index of medium cube vertices within each coarse node
                    vertices_index[g].push_back(vertices_size[g]);
                    vertices_size[g] += cubex((1<<s)+1);
                }
            }
        });
        // Reset visit flag and occupancy values of medium cubes
        MEASURE_TIME("medium_iteration_init part 2", timing, {
            restore_cubes();
            restore_vertices();
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                cubes[g].resizefill(cubes_size[g], 0);
                vertices[g].resizefill(vertices_size[g], -1);
            }
        });
        // Find medium cubes from all_cubes within each thread's range and mark them as visited
        // Note that due to the large amount of medium cubes, we have to process and reset in multiple chunks
        MEASURE_TIME("medium_iteration_init part 3", timing, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                medium_cube_ext cestart;
                medium_cube_ext ceend;
                cestart.first.first = starts[g];
                ceend.first.first = ends[g];
                int cube_start = lower_bound(all_cubes.begin(), all_cubes.begin() + all_cubes_size, cestart, compareCubeExt) - all_cubes.begin();
                int cube_end = lower_bound(all_cubes.begin(), all_cubes.begin() + all_cubes_size, ceend, compareCubeExt) - all_cubes.begin();
                for (int j = cube_start; j < cube_end; j++) {
                    int i = INDEX(all_cubes, j).first.first;
                    int s = grid_node_level(INDEX(nodes, INDEX(leaf_nodes_vector, i)));
                    int ss = 1<<s;
                    auto cqj = INDEX(all_cubes, j).first.second;
                    int ici = cube_index(cqj[0], cqj[1], cqj[2], ss) + INDEX(cubes_index[g], i-starts[g]);
                    medium_assign_cubes(g, ici);
                }
            }
        });
        MEASURE_TIME("medium_iteration_init part 4", timing, {
            *update = 0;
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                medium_cube cstart;
                medium_cube cend;
                cstart.first = starts[g];
                cend.first = ends[g];
                // cubes_entry is those medium cubes after the cross-node flooding the the previous loop
                // find those within thread g's range and put them to cubes_queue
                int cube_start = lower_bound(cubes_entry.begin(), cubes_entry.end(), cstart, compareCube) - cubes_entry.begin();
                int cube_end = lower_bound(cubes_entry.begin(), cubes_entry.end(), cend, compareCube) - cubes_entry.begin();
                assert(cubes_queue[g].empty());
                for (int j = cube_start; j < cube_end; j++) {
                    auto &cej = INDEX(cubes_entry, j);
                    int i = cej.first;
                    int nvi = INDEX(leaf_nodes_vector, i);
                    auto &nnvi = INDEX(nodes, nvi);
                    int s = grid_node_level(nnvi);
                    int ss = 1<<s;
                    hypercube cubei = nnvi.c;
                    auto cqj = cej.second;
                    int coords[3];
                    for (int p = 0; p < 3; p++) coords[p] = cqj[p];
                    int flag = 0;
                    int ici = cube_index(coords[0], coords[1], coords[2], ss) + INDEX(cubes_index[g], i-starts[g]);
                    if (INDEX(cubes[g], ici) == 0) {
                        medium_assign_cubes(g, ici);
                        cubes_queue[g].push_back(cej);
                    }
                }
                // "update" counts the total number of medium cubes to query in Python
                *update += cubes_queue[g].size();
            }
        });
        // calls the common part to get the size of vertices array to allocate output arrays in Python\
        // because the python side has better memory management
        int ret;
        MEASURE_TIME("medium_iteration_init part 5", timing, {
            ret = medium_iteration_common(timing);
        });
        return ret;
    }

    // As mentioned, a medium_loop function (core.cpp) will call medium_iteration_init once, and call medium_iteration_output and medium_iteration_regular multiple times
    int medium_iteration_regular(sdfT *sdf, int timing) {
        using namespace coarse;
        using namespace medium;
        // In implementation, we use sdf values to determine the occupancy values, but note the sdf is not strict distance
        // We write the occupancy values to the table of each thread "vertices[]"
        MEASURE_TIME("medium_iteration_regular part 1", timing, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int i = 0; i < output_vertices.size(); i++) {
                assert(!std::isnan(sdf[i]));
                auto &ind = INDEX(output_vertices_index, i);
                INDEX(vertices[ind.first], ind.second) = sdf[i]>=0? 1: 2;
            }
        });
        MEASURE_TIME("medium_iteration_regular part 2", timing, {
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                all_cubes_buffer[g].clear();
                int cqs = cubes_queue[g].size();
                // check each medium cube in current cubes_queue
                for (int j = 0; j < cqs; j++) {
                    int i = INDEX(cubes_queue[g], j).first;
                    auto &nnvi = INDEX(nodes, INDEX(leaf_nodes_vector, i));
                    int s = grid_node_level(nnvi);
                    int ss = 1<<s;
                    hypercube cubei = nnvi.c;
                    auto cqj = INDEX(cubes_queue[g], j).second;
                    int coords[3];
                    for (int p = 0; p < 3; p++) coords[p] = cqj[p];
                    int flag = 0;
                    int ici = cube_index(coords[0], coords[1], coords[2], ss) + INDEX(cubes_index[g], i-start_node[g]);
                    assert(INDEX(cubes[g], ici));
                    // check each edge
                    for (int e = 0; e < 12; e++) {
                        int vcoords[3];
                        vcoords[e/4] = coords[e/4] + 1;
                        vcoords[(e/4+1) % 3] = coords[(e/4+1) % 3] + (e&1);
                        vcoords[(e/4+2) % 3] = coords[(e/4+2) % 3] + ((e>>1)&1);
                        int base = INDEX(vertices_index[g], i-start_node[g]);
                        int index = base + cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1);
                        int sign1 = INDEX(vertices[g], index);
                        vcoords[e/4] = coords[e/4];
                        index = base + cube_index(vcoords[0], vcoords[1], vcoords[2], ss+1);
                        int sign2 = INDEX(vertices[g], index);
                        // if the two vertices have different occupancy values, then we need to flood to the neighboring cubes and add them to cubes_queue
                        // Note that here we only do intra-coarse-node flooding
                        if (sign1 != sign2) {
                            flag |= 1 << e;
                            // each edge has 4 neighboring cubes
                            for (int c = 0; c < 4; c++) {
                                int8_t ccoords[3];
                                ccoords[e/4] = vcoords[e/4];
                                int tmp1 = ccoords[(e/4+1) % 3] = vcoords[(e/4+1) % 3] + (c&1) - 1;
                                int tmp2 = ccoords[(e/4+2) % 3] = vcoords[(e/4+2) % 3] + ((c>>1)&1) - 1;
                                if (tmp1 < 0 || tmp1 >= ss || tmp2 < 0 || tmp2 >= ss) continue;
                                int ci = cube_index(ccoords[0], ccoords[1], ccoords[2], ss);
                                int ici = INDEX(cubes_index[g], i-start_node[g])+ ci;
                                if (INDEX(cubes[g], ici)) continue;
                                medium_assign_cubes(g, ici);
                                cubes_queue[g].push_back(mp(i, make_array3(ccoords[0], ccoords[1], ccoords[2])));
                            }
                        }
                    }
                    // at the same time, we add the medium cubes containing different values to all_cubes (each thread's buffer first)
                    if (flag) {
                        all_cubes_buffer[g].push_back(mp(INDEX(cubes_queue[g], j), flag));
                    }
                }
                cubes_queue[g].erase(0, cqs);
            }
        });
        // collect all_cubes from all threads' buffers
        MEASURE_TIME("medium_iteration_regular part 3", timing, {
            for (int g = 0; g < N_THREAD; g++) {
                auto tmp = all_cubes.size();
                all_cubes.resize(all_cubes.size() + all_cubes_buffer[g]._size);
                std::copy(all_cubes_buffer[g].begin(), all_cubes_buffer[g].end(), all_cubes.begin() + tmp);
            }
        });
        // for the new cubes in cubes_queue, we need to query their vertices again
        return medium_iteration_common(timing);
    }

    // After we allocate the vertices array in Python, we write their 3D coordinates here
    void medium_iteration_output(T *xyz) {
        using namespace params;
        using namespace medium;
        OMP_PRAGMA(omp parallel for schedule(dynamic))
        for (int i = 0; i < output_vertices.size(); i++) {
            auto &ovi = INDEX(output_vertices, i);
            compute_coords(xyz + i*3, ovi.coords, ovi.L);
        }
    }

    // Save all medium_cubes to disk for later reuse (for debug purpose)
    void medium_dump() {
        using namespace medium;
        std::stringstream filename;
        filename << params::output_path << "/medium_cubes.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb" );
        write_vec(outfile, all_cubes);
        fclose(outfile);
    }

    // Load previously saved medium_cubes from disk
    void medium_load() {
        using namespace medium;
        std::stringstream filename;
        filename << params::output_path << "/medium_cubes.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb" );
        read_vec(infile, all_cubes);
        fclose(infile);
    }

    // In the end of medium_loop, we do inter-coarse-node flooding
    bool medium_iteration_end(int timing) {
        using namespace coarse;
        using namespace medium;
        // If no new medium_cubes are added, we are done
        if (all_cubes_size == all_cubes.size()) return true;
        if (timing) {
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "all_cubes.size()=%d\n", int(all_cubes.size()));
            fclose(log);
        }
        typedef vec<int, medium_cube> vec_mc;
        MEASURE_TIME("medium_iteration_end part 1", timing, {
            cubes_entry.clear();
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                cubes_entry_buffer[g].clear();
                // For simplicity, we take all medium cubes and divide among threads
                int step = (all_cubes.size() - all_cubes_size + N_THREAD - 1) / N_THREAD;
                int start_g = all_cubes_size + step * g; 
                int end_g = smin(start_g + step, (int)all_cubes.size());
                for (int j = start_g; j < end_g; j++) {
                    auto &acjf = INDEX(all_cubes, j).first;
                    auto cqj = acjf.second;
                    int i = acjf.first;
                    auto &nnvi = INDEX(nodes, INDEX(leaf_nodes_vector, i));
                    hypercube cubei = nnvi.c;
                    int s = grid_node_level(nnvi);
                    int ss = 1<<s;
                    int coords[3];
                    for (int p = 0; p < 3; p++) coords[p] = cqj[p];
                    for (int e = 0; e < 12; e++) {
                        if ((INDEX(all_cubes, j).second >> e) & 1) {
                            int vcoords[3];
                            vcoords[e/4] = coords[e/4];
                            vcoords[(e/4+1) % 3] = coords[(e/4+1) % 3] + (e&1);
                            vcoords[(e/4+2) % 3] = coords[(e/4+2) % 3] + ((e>>1)&1);
                            for (int c = 0; c < 4; c++) {
                                int ccoords[3];
                                ccoords[e/4] = vcoords[e/4];
                                int tmp1 = ccoords[(e/4+1) % 3] = vcoords[(e/4+1) % 3] + (c&1) - 1;
                                int tmp2 = ccoords[(e/4+2) % 3] = vcoords[(e/4+2) % 3] + ((c>>1)&1) - 1;
                                // Check if the neighboring cube is outside the coarse node
                                if (tmp1 < 0 || tmp1 >= ss || tmp2 < 0 || tmp2 >= ss) {
                                    int qcoords[3];
                                    int dirs[3];
                                    int qL = cubei.L + s;
                                    for (int p = 0; p < 3; p++) assign_check(qcoords[p], cubei.coords[p], 1<<s, vcoords[p]);
                                    dirs[(e/4+1) % 3] = c&1;
                                    dirs[(e/4+2) % 3] = (c>>1)&1;
                                    vec_mc res;
                                    // For inter-coarse-node flooding, a neighbor node can have a different time range,
                                    // So we call edge_neighbor_search with different time parameters
                                    edge_neighbor_search(qcoords, qL, dirs, cubei.tcoord, cubei.tL, 1, e/4, coarse::nodes, coarse::nodemap, res);
                                    cubes_entry_buffer[g].insert_end(res);
                                    edge_neighbor_search(qcoords, qL, dirs, cubei.tcoord, cubei.tL, 0, e/4, coarse::nodes, coarse::nodemap, res);
                                    cubes_entry_buffer[g].insert_end(res);
                                    edge_neighbor_search(qcoords, qL, dirs, cubei.tcoord + 1, cubei.tL, 1, e/4, coarse::nodes, coarse::nodemap, res);
                                    cubes_entry_buffer[g].insert_end(res);
                                }
                            }
                        }
                    }
                }
            }
        });
        // Collect new medium cubes from all threads' buffers
        MEASURE_TIME("medium_iteration_end part 2", timing, {
            for (int g = 0; g < N_THREAD; g++) {
                cubes_entry.insert_end(cubes_entry_buffer[g]);
            }
        });
        // Sort all medium cubes again by sorting the new ones and merging
        MEASURE_TIME("medium_iteration_end part 3", timing, {
            sort(all_cubes.begin() + all_cubes_size, all_cubes.end(), compareCubeExt);
        });
        MEASURE_TIME("medium_iteration_end part 4", timing, {
            std::inplace_merge(all_cubes.begin(), all_cubes.begin()+all_cubes_size, all_cubes.end(), compareCubeExt);
            all_cubes_size = all_cubes.size();
        });
        if (timing) {
            FILE *log = fopen(params::log_path.c_str(), "a");
            fprintf(log, "cubes_entry.size()=%d\n", int(cubes_entry.size()));
            fclose(log);
        }
        // If no new medium cubes are added, we are done
        if (cubes_entry.size() == 0) return true;
        MEASURE_TIME("medium_iteration_end part 5", timing, {
            make_unique(cubes_entry);
        });
        return false;
    }
}

// This function is used to sort interval end-points (see later)
bool compareEnds(const pair<T, int> &a, const pair<T, int> &b) {
    if (a.first < b.first) return 1;
    if (a.first > b.first) return 0;
    return a.second > b.second;
}

// These are functions exposed to Python
extern "C" {
    void visibility_filter(bool simplify_occluded, int relax_margin, int boundary_margin, int relax_iters) {
        using namespace params;
        using namespace coarse;
        using namespace medium;
        // We take all time span (start point + end point) of all medium cubes and sort them
        // We put end point before start point if they are the same
        typedef pair<T, int> ie;
        vec<int, ie> interval_ends;
        MEASURE_TIME("visibility_filter part 2", 1, {
            visible.resize(all_cubes.size());
            visible.fill(0);
            for (int j = 0; j < all_cubes.size(); j++) {
                int i = all_cubes[j].first.first;
                hypercube c = INDEX(nodes, INDEX(leaf_nodes_vector, i)).c;
                interval_ends.push_back(mp(c.tcoord * 1.0 / (1 << c.tL) * tsize, j));
                interval_ends.push_back(mp((c.tcoord + 1.0) / (1 << c.tL) * tsize, -j));
            }
        });
        MEASURE_TIME("visibility_filter part 3", 1, {
            _sort(interval_ends, compareEnds);
        });
        int maxH = 0, maxW = 0;
        T factor = params::pixels_per_cube * params::inv_scale;
        typedef pair<pair<II, int>, T> iiiT;
        typedef vec<int, iiiT> vec_iiiT;
        MEASURE_TIME("visibility_filter part 4", 1, {
            // Set up a canvas for each camera for z-buffering with resolution reduction "factor"
            // Usually all H and W are the same, but we still allocate based on the maximum H and W
            for (int k = 0; k < cams.size(); k++) {
                camera current_cam = cams[k];
                int H = current_cam.H / factor;
                int W = current_cam.W / factor;
                maxH = smax(maxH, H);
                maxW = smax(maxW, W);
            }
            // Divide the cameras among threads
            OMP_PRAGMA(omp parallel for schedule(dynamic))
            for (int g = 0; g < N_THREAD; g++) {
                T canvas[(maxH + 2*boundary_margin) * (maxW + 2*boundary_margin)];
                int start_g = (cams.size() + N_THREAD - 1) / N_THREAD * g;
                int end_g = smin((cams.size() + N_THREAD - 1) / N_THREAD * (g+1), cams.size());
                int pointer = 0;
                set<int> sliced_cubes;
                for (int k = start_g; k < end_g; k++) {
                    vec_iiiT projected_coords_j;
                    camera current_cam = cams[k];
                    int H = current_cam.H / factor;
                    int W = current_cam.W / factor;
                    int new_pointer = pointer;
                    int Hp = H + 2*boundary_margin;
                    int Wp = W + 2*boundary_margin;
                    for (int i = 0; i < Hp * Wp; i++)
                        canvas[i] = std::numeric_limits<T>::infinity();
                    // Find the set of medium cubes intersecting with the current camera's timestamp
                    // Within each thread, the camera is in sequential order, so we reuse the pointer of the previous camera
                    while (new_pointer != interval_ends.size() && interval_ends[new_pointer].first <= current_cam.t) new_pointer++;
                    for (int i = pointer; i < new_pointer; i++) {
                        if (interval_ends[i].second > 0) sliced_cubes.insert(interval_ends[i].second);
                        else sliced_cubes.erase(-interval_ends[i].second);
                    }
                    pointer = new_pointer;
                    for (auto &sliced_cube: sliced_cubes) {
                        T image_coords[3];
                        int i = all_cubes[sliced_cube].first.first;
                        auto tmp = all_cubes[sliced_cube].first.second;
                        int coords[3];
                        for (int p = 0; p < 3; p++) coords[p] = tmp[p];
                        auto &nnvi = INDEX(nodes, INDEX(leaf_nodes_vector, i));
                        hypercube c = nnvi.c;
                        int s = grid_node_level(nnvi);
                        for (int p = 0; p < 3; p++) assign_check(c.coords[p], c.coords[p], 1<<s, coords[p]);
                        c.L += s;
                        // Project the center of the medium cube to the image plane
                        projected_coords(c, k, image_coords, NULL, NULL);
                        if (image_coords[2] >= 0) {
                            int x = floor(image_coords[0] / factor);
                            int y = floor(image_coords[1] / factor);
                            bool indexable = (x >= -boundary_margin && y >= -boundary_margin && x < W + boundary_margin && y < H + boundary_margin);
                            // If the projected coordinate is within image bounds
                            if (indexable) {
                                // If we consider occluison among medium cubes, we first store all projected coordinates and depth
                                // and maintain the min depth in the z-buffer
                                if (simplify_occluded) {
                                    projected_coords_j.push_back(mp(mp(mp(x, y), sliced_cube), image_coords[2]));
                                    int x0 = x + boundary_margin;
                                    int y0 = y + boundary_margin;
                                    if (image_coords[2] <= canvas[x0 * Hp + y0]) {
                                        canvas[x0 * Hp + y0] = image_coords[2];
                                    }
                                }
                                // Other wise, we directly mark the medium cube as visible
                                else visible[sliced_cube] = 1;
                            }
                        }
                    }
                    // We revisit stored projected coordinates and depth, and compare with the z-buffer in a neighborhood to determine visibility
                    if(simplify_occluded) {
                        for (int j = 0; j < projected_coords_j.size(); j++) {
                            int i = projected_coords_j[j].first.second;
                            int x = projected_coords_j[j].first.first.first;
                            int y = projected_coords_j[j].first.first.second;
                            T z = projected_coords_j[j].second;
                            for (int dx = -relax_margin; dx <= relax_margin; dx++)
                            for (int dy = -relax_margin; dy <= relax_margin; dy++) {
                                int nx = x + dx + boundary_margin;
                                int ny = y + dy + boundary_margin;
                                if (nx >= 0 && ny >= 0 && nx < W+2*boundary_margin && ny < H+2*boundary_margin) {
                                    if (z <= canvas[nx * Hp + ny]) {
                                        visible[i] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        set<medium_cube> new_visible_set, old_visible_set;
        typedef vec<int, medium_cube> vec_mc;
        MEASURE_TIME("visibility_filter part 5", 1, {
            visible_set.clear();
            occluded_set.clear();
            for (int i = 0; i < visible.size(); i++) {
                if (visible[i]) visible_set.insert(all_cubes[i].first);
                else occluded_set.insert(all_cubes[i].first);
            }
            // We do several iterations of relaxation to mark more cubes as visible
            for (int it = 0; it < relax_iters; it++) {
                for (auto viscube: visible_set) {
                    auto cqj = viscube.second;
                    int i = viscube.first;
                    hypercube cubei = nodes[leaf_nodes_vector[i]].c;
                    int s = grid_node_level(nodes[leaf_nodes_vector[i]]);
                    int coords[3];
                    coords[0] = cqj[0];
                    coords[1] = cqj[1];
                    coords[2] = cqj[2];
                    hypercube c;
                    c.L = cubei.L + s;
                    c.tcoord = cubei.tcoord;
                    c.tL = cubei.tL;
                    for (int e = 0; e < 12; e++) {
                        int vcoords[3];
                        vcoords[e/4] = coords[e/4];
                        vcoords[(e/4+1) % 3] = coords[(e/4+1) % 3] + (e&1);
                        vcoords[(e/4+2) % 3] = coords[(e/4+2) % 3] + ((e>>1)&1);
                        for (int c = 0; c < 4; c++) {
                            int qcoords[3];
                            int dirs[3];
                            int qL = cubei.L + s;
                            for (int p = 0; p < 3; p++) assign_check(qcoords[p], cubei.coords[p], 1<<s, vcoords[p]);
                            dirs[(e/4+1) % 3] = c&1;
                            dirs[(e/4+2) % 3] = (c>>1)&1;
                            vec_mc res;
                            edge_neighbor_search(qcoords, qL, dirs, cubei.tcoord, cubei.tL, 1, e/4, coarse::nodes, coarse::nodemap, res);
                            for (auto rescube: res) {
                                // But only surface-intersecting cubes (originally in occluded_set) will become visible again
                                if (occluded_set.count(rescube)) {
                                    occluded_set.erase(rescube);
                                    new_visible_set.insert(rescube);
                                }
                            }
                        }
                    }
                }
                old_visible_set.insert(visible_set.begin(), visible_set.end());
                visible_set = new_visible_set;
                new_visible_set.clear();
            }
            visible_set.insert(old_visible_set.begin(), old_visible_set.end());
            old_visible_set.clear();
            new_visible_set.clear();
            visible_cubes.clear();
            for (auto cube: visible_set) {
                visible_cubes.push_back(cube);
            }
            occluded_cubes.clear();
            for (auto cube: occluded_set) {
                occluded_cubes.push_back(cube);
            }
            // Sort the final visible and occluded cubes for later fine steps
            _sort(visible_cubes);
            _sort(occluded_cubes);
        });
    }

    // Save visibility filtering result to disk for later reuse (for debug purpose)
    void visibility_filter_dump() {
        using namespace medium;
        std::stringstream filename;
        filename << params::output_path << "/visibility.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb" );
        write_vec(outfile, visible_cubes);
        write_vec(outfile, occluded_cubes);
        fclose(outfile);
    }
    void visibility_filter_load() {
        using namespace medium;
        std::stringstream filename;
        filename << params::output_path << "/visibility.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb");
        read_vec(infile, visible_cubes);
        read_vec(infile, occluded_cubes);
        fclose(infile);
    }
    
    // We cannot write an stl::set to disk directly, so we need some post processing
    int visibility_filter_ending() {
        using namespace medium;
        FILE *log = fopen(params::log_path.c_str(), "a");
        MEASURE_TIME("visibility_filter_ending part 1", 1, {
            visible_set.clear();
            occluded_set.clear();
            for (auto &cube: visible_cubes) {
                visible_set.insert(visible_set.end(), cube);
            }
            for (auto &cube: occluded_cubes) {
                occluded_set.insert(occluded_set.end(), cube);
            }
        });
        fprintf(log, "visible_cubes size: %d\n", (int)visible_cubes.size());
        fprintf(log, "occluded_cubes size: %d\n", (int)occluded_cubes.size());
        MEASURE_TIME("visibility_filter_ending part 2", 1, {
            current_size = visible_set.size();
            sorted_size = visible_set.size();
        });
        fclose(log);
        return visible_set.size();
    }

    // Clean up all medium step related variables to save memory
    void medium_clean_up() {
        using namespace medium;
        CLS(cubes_entry);
        CLS(output_vertices_index);
        CLS(output_vertices);
        CLS(visible);
        CLS(all_cubes);
        for (int g = 0; g < N_THREAD; g++) {
            CLS(cubes_entry_buffer[g]);
            CLS(cubes_queue[g]);
            CLS(cubes[g]);
            CLS(vertices[g]);
            CLS(cubes_nonzero[g]);
            CLS(vertices_nonzero[g]);
            CLS(vertices_index[g]);
            CLS(cubes_index[g]);
            CLS(output_vertices_index_buffer[g]);
            CLS(output_vertices_buffer[g]);
            CLS(all_cubes_buffer[g]);
        }
    }

}