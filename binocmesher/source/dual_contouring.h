#ifndef DUAL_CONTOURING_H
#define DUAL_CONTOURING_H

// To explain first: the scene is repsented as a union of several scene elements (mountain, ground, etc) for a good reason
// We treat the overall occupancy function as their "OR" results in all previous steps
// But in this step, we look into individual elements and do dual contouring each element
// This is a data structure for an edge including its coordinates and the the element id
struct edge {
    int coords[3];
    lT L, tL;
    eleT ele;
    timeT tcoord;
    int8_t dir;
};

namespace dual_contouring {
    // the current group id
    extern int t_group;
    // the restricted set for each big loop need to be tracked when loading different groups within the loop because t_group changes
    extern vec<int, int> restricted_set;
    // variables to preallocate memory for the active tree nodes
    extern vec<int, int> node_labels, subtree_st, subtree_ed;
    extern vec<int, int> vertices[N_THREAD];
    // variables for querying the occupancy function value of individual elements
    extern vec<int, int> info_offset_rearranged;
    extern vec<int, int> nodes_queue[N_THREAD];
    extern vec<int, pair<sdeque<II>::iterator, long long> > pointers;
    // note that sdeque is used as a container without contiguous requirement to reduce memory usage
    extern sdeque<vertex> output_vertices, output_vertices_buffer[N_THREAD];
    extern sdeque<II> output_vertices_index, output_vertices_index_buffer[N_THREAD];
}

// This is a data structure containing an edge and queried neighbor cubes
struct queriedEdge {
    edge e;
    int vertices_nid[8];
    timeT vertices_gid[8];
};

// This is a data structure storing pending new bipolar edges
// Given a new bipolar edge, it might have already been queried before; detailed_edge_p points to that
// propagation also indicates whether new edges need to be propagated (see Fig.16 in the paper)
struct edgeRes {
    edge e;
    int vertices_nid[8];
    timeT vertices_gid[8];
    timeT propagation[8];
    int8_t category;
    queriedEdge* detailed_edge_p;
};

namespace dual_contouring {
    // unfinalized_edges stored all bipolar edges that has unknown neighbors because the entire tree is load group by group; finalized_edges stores those with all neighbors known; new_unfinalized_edges stores and unfinalized_edges_sorted_cnt are temporary variables to update finalized edges
    extern sdeque<queriedEdge> unfinalized_edges, finalized_edges, new_unfinalized_edges;
    extern int unfinalized_edges_sorted_cnt;
    // pending_edges are newly found bipolar edges in the current group; propagated_edges are propageted edges (see Fig.16 in the paper)
    extern sdeque<edge> pending_edges, pending_edges_buffer[N_THREAD], propagated_edges;
    // absorbed is 1 when an old unfinalized edge is absorbed into newly found edges
    extern sdeque<int> absorbed;
    // backref will be made into detailed_edge_p field of pending_edges_results
    extern sdeque<queriedEdge*> backref;
    // pending_edges_results stores query results of pending_edges
    extern sdeque<edgeRes> pending_edges_results;
    // static adjacency list of the dual graph of the tree
    extern vec<ll, int> head_cache[CACHECNT];
    extern vec<int, int> nxt_cache[CACHECNT], edge_pointer_cache[CACHECNT];
    // edges array storing the edge neighbors
    extern vec<int, queriedEdge> edges_cache[CACHECNT];
    extern FILE *edges_fp_cache[CACHECNT];
    extern int edges_size_cache[CACHECNT];
    // a cache mechanism can be potentially used but we do not use it in the current implementation and set CACHECNT to 1
    extern map<int, int> priority;
    extern map<int, int> cache_map;
    extern int occupied[CACHECNT];
    extern map<int, int> priority2;
    extern map<int, int> cache_map2;
    extern int occupied2[CACHECNT];
    // alias for the current dual graph and edge array
    extern vec<ll, int> *head_;
    extern vec<int, int> *nxt_, *edge_pointer_;
    extern vec<int, queriedEdge> *edges_;
    extern FILE **edges_fp_;
    extern int *edges_size_;
    // Mark whether the nodes are in view or not, which indicates whether the dual vertices are in view or not
    // used for an interpolation trick in the slicing step
    extern vec<int, int8_t> inview_tag;
}

// Load the dual graph of group t (the tree of nodes of group id t)
void load_graph(int t);
// load the edges array of group t, to_memory indicates whether to load into memory or just open the file pointer
// the later is used when the only thing todo is to write edges to this group t
void load_bip_edges(int t, int to_memory);
// clean all current loaded dual graphs and edges arrays
void clean_cache();

// These are functions exposed to Python
extern "C" {
    // Initialize dual contouring step
    void dual_contouring_init();
    // Load the active group t, if it is the first time query_vertices is true and it returns to python the number of vertices to query individual occupancy;
    // For later calls (see Alg.2 in the paper), query_vertices is false and it just loads the tree into memory and compute neighbors
    int load_group(int T, bool query_vertices, bool compute_inview_tag);
    // It returns to python the vertices to query individual occupancy functions
    void load_active_group_output(T *xyz, int vstart, int vend);
    // This function calls load_group (with query_vertices being false) and the second part of itself several times to compute the bipolar edge neighbors
    void constructing_meshes(sdfT *sdf, int debug);
    // Save unfinalized edges for later reuse for debug purpose (other data are already on the disk)
    void save_unfinalized_edges();
    // Load unfinalized edges from previous saved file
    void load_unfinalized_edges();
}

#endif // DUAL_CONTOURING_H