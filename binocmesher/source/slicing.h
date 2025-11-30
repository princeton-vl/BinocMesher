#ifndef SLICING_H
#define SLICING_H

// There must be both external declaration in previous headers and definition here

namespace params {
    int n_elements;
    T *center;
    T size, tsize;
    T pixels_per_cube, inv_scale, outview_scale;
    T min_dist;
    T fading_time, deltaT;
    int max_tL;
    vec<int, camera> cams;
    vec<int, int> cam_lookup;
    std::string output_path, log_path;
}

namespace coarse {
    vec<int, node> nodes;
    vec<int, int> leaf_nodes_vector;
    vec<int, int> nodemap;
}

namespace medium {
    int all_cubes_size;
    vec<int, medium_cube_ext> all_cubes;
    vec<int, medium_cube> cubes_entry;
    vec<int, vertex> output_vertices;
    vec<int, II> output_vertices_index;
    vec<int, medium_cube_ext> all_cubes_buffer[N_THREAD];
    vec<int, medium_cube> cubes_entry_buffer[N_THREAD], cubes_queue[N_THREAD];
    vec<int, int> cubes[N_THREAD], vertices[N_THREAD], cubes_nonzero[N_THREAD], vertices_nonzero[N_THREAD];
    vec<int, int> vertices_index[N_THREAD], cubes_index[N_THREAD];
    vec<int, II> output_vertices_index_buffer[N_THREAD];
    vec<int, vertex> output_vertices_buffer[N_THREAD];
    int start_node[N_THREAD];
    vec<int, int> visible;
    vec<int, medium_cube> visible_cubes, occluded_cubes;
    set<medium_cube> visible_set, occluded_set;
    int current_size, sorted_size;
}

namespace fine {
    int fine_group;
    int start_mc, end_mc;
    vec<int, int> visible_cubes_nodes_size;
    vec<int, node> nodes_array[N_FINEGROUP];
    vec<int, pair<int, pair<T, bool> > > nodes_queue[N_FINEGROUP];
    vec<int, medium_cube> res_coll;
    vec<int, II> output_vertices_index, output_vertices_index_buffer[N_THREAD];
    vec<int, vertex> output_vertices, output_vertices_buffer[N_THREAD];
    vec<int, int> vertices[N_THREAD];
    vec<int, medium_cube> res_coll_buffer[N_THREAD];
    vec<int, be> edges_buffer[N_THREAD];
    vec<int, int> head, nxt;
    vec<int, int> vertex_flags[N_THREAD];
    vec<int, int> flatten_nodes;
    int stored_size;
    vec<int, int> maxLs;
    int info_offset[N_FINEGROUP];
    vec<int, node> active_nodes;
    vec<int, int> mask;
}

namespace rearrange {
    vec<int, node> nodes;
    vec<int, pair<hypercube, II> > framing_cubes;
    vec<int, int> visited_subtree, group_id_start, group_id_end;
    vec<int, int> flatten_nodes;
    vec<int, II> info_offset_rearranged_vec;
    vec<int, int> tree_sizes;
}

namespace dual_contouring {
    int t_group;
    vec<int, int> restricted_set;
    vec<int, int> node_labels, subtree_st, subtree_ed;
    vec<int, int> vertices[N_THREAD];
    vec<int, int> info_offset_rearranged;
    vec<int, int> nodes_queue[N_THREAD];
    vec<int, pair<sdeque<II>::iterator, long long> > pointers;
    sdeque<vertex> output_vertices, output_vertices_buffer[N_THREAD];
    sdeque<II> output_vertices_index, output_vertices_index_buffer[N_THREAD];
    sdeque<queriedEdge> unfinalized_edges, finalized_edges, new_unfinalized_edges;
    int unfinalized_edges_sorted_cnt;
    sdeque<edge> pending_edges, pending_edges_buffer[N_THREAD], propagated_edges;
    sdeque<int> absorbed;
    sdeque<queriedEdge*> backref;
    sdeque<edgeRes> pending_edges_results;
    vec<ll, int> head_cache[CACHECNT];
    vec<int, int> nxt_cache[CACHECNT], edge_pointer_cache[CACHECNT];
    vec<int, queriedEdge> edges_cache[CACHECNT];
    FILE *edges_fp_cache[CACHECNT];
    int edges_size_cache[CACHECNT];
    map<int, int> priority;
    map<int, int> cache_map;
    int occupied[CACHECNT];
    map<int, int> priority2;
    map<int, int> cache_map2;
    int occupied2[CACHECNT];
    vec<ll, int> *head_;
    vec<int, int> *nxt_, *edge_pointer_;
    vec<int, queriedEdge> *edges_;
    FILE **edges_fp_;
    int *edges_size_;
    vec<int, int8_t> inview_tag;
}

namespace bisection {
    int bisection_group;
    vec<ll, int> query_cnt;
    ll last_head, current_head;
    vec<int, T> lefts, rights;
    vec<int, int> starts;
    vec<int, int> center_indices;
    vec<int, HP> hyperpolys;
    vec<ll, int> vertex_map;
    vec<int, HV> computed_vertices;
    HVTable hypervertices;
    HVTable hypervertices_g[N_THREAD];
    vec<int, pair<HVID, HV> > hypervertices_vec;
}

// A vertex in the mesh is from cutting an edge connecting two center vertices, therefore ID for a mesh vertex consists of two HVIDs
typedef array<HVID, 2> VID;
// map from VID to the actual 3D coordinates; the VID is used for merging identical vertices (extracted from the same edge)
typedef pair<VID, pair<int, pair<bool, array<spaceT, 3> > > > VUnmerged;

namespace slicing {
    // The maximum number of elements
    #define N_ELE 5
    // unmerged vertex data of the mesh of each element
    vec<int, VUnmerged> unmerged_vertices[N_ELE];
    // merging map from VID to the final vertex index
    vec<int, int> merging_map;
    // the face data of the mesh of each element
    vec<int, array<int, 3> > faces[N_ELE];
    // the vertex data of the mesh of each element
    vec<int, array<spaceT, 3> > vertices[N_ELE];
    // visibility tag of each vertex
    vec<int, int> vertices_inview[N_ELE];
    // auxiliary arrays for bucket sort
    vec<int, VUnmerged> vertices_tmp;
    vec<int, int> head, nxt, ind;
}

// These are functions exposed to Python
extern "C" {
    // The preprocessing function that prebuild a lookup table for which edge to cut given a timestamp
    void slicing_preprocess();
    // the main slicing function that returns the number of vertices and faces per element
    // extra_smooth indicates whether "Extension to Ameliorate Popping Artifacts" in Sec. 7 is turned On
    void run_slicing(T t0, int *v_cnts, int *f_cnts, bool extra_smooth=true);
    // actually output the mesh data
    void slicing_output(int ele, T *output_verts, int *output_faces, int *output_inview);
    // clean up slicing data
    void slicing_clean_up();
}

#endif // SLICING_H
