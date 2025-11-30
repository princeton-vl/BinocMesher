#ifndef FINE_STEP_H
#define FINE_STEP_H

// The MAXIMUM number of medium cubes to be processed in each fine step group
#define N_FINEGROUP 10000
// Data structure to store discovered bipolar edges in the fine tree
typedef array<int, 9> be;

// The fine step is to split medium_cubes and do potential flooding iteratively
namespace fine {
    // The ACTUAL number of medium cubes to be processed in each fine step group
    extern int fine_group;
    // The current medium cube ids being processed
    extern int start_mc, end_mc;
    // The number of nodes of the subtree of each visible medium node
    extern vec<int, int> visible_cubes_nodes_size;
    // array of subtree leaves
    extern vec<int, node> nodes_array[N_FINEGROUP];
    // array of subtree nodes in priority queue order
    extern vec<int, pair<int, pair<T, bool> > > nodes_queue[N_FINEGROUP];
    // collection of new medium cubes found through propagation
    // (that may have sign change on inner vertices instead of 8 vertices)
    extern vec<int, medium_cube> res_coll;
    // These are variables used in multi-thread propagation
    extern vec<int, II> output_vertices_index, output_vertices_index_buffer[N_THREAD];
    extern vec<int, vertex> output_vertices, output_vertices_buffer[N_THREAD];
    extern vec<int, int> vertices[N_THREAD];
    extern vec<int, medium_cube> res_coll_buffer[N_THREAD];
    extern vec<int, be> edges_buffer[N_THREAD];
    extern vec<int, int> head, nxt;
    extern vec<int, int> vertex_flags[N_THREAD];
    // compressed way of storing trees on disk - only store split tag
    extern vec<int, int> flatten_nodes;
    // current number of stored medium cubes
    extern int stored_size;
    // list of maximum level of subtrees in current medium cubes
    extern vec<int, int> maxLs;
    // accumulated subtree sizes so we can read into a bunch of stored subtrees
    extern int info_offset[N_FINEGROUP];
    // Active node tree that only expands the active medium cube given the current timestamp
    extern vec<int, node> active_nodes;
    // Mask to indicate whether a node is active or not
    extern vec<int, int> mask;
}

// These are functions exposed to Python
extern "C" {
    // initalization work based on fine_group - the ACTUAL number of medium cubes per group
    void fine_init(int fine_group);
    // split medium cubes from start_mc to end_mc and flood to potential new medium cubes
    int fine_iteration(int start_mc, int end_mc);
    // output vertices to Python to query the occupancy values
    void fine_iteration_output(T *xyz);
    // flood to potential new medium cubes based on queried occupancy values
    int fine_iteration_propagate(sdfT *sdf);
    // Save all results for later reuse (for debug purpose)
    void fine_dump();
    void fine_load();
}


// after the fine step, there is a rearrange step to instantiate and sort the visible medium cubes in temporal order
// and them the medium nodes are assigned group ids
namespace rearrange {
    // the tree up to the medium visible cubes
    extern vec<int, node> nodes;
    // the visible medium cubes serve as a sort of framing cubes, to be sorted and processed
    extern vec<int, pair<hypercube, II> > framing_cubes;
    // for simplicity, the node group id (see paper) is based on medium nodes
    // we visit the tree in increasing order of group id
    // visited_subtree indicates whether a subtree has been visited
    // group_id_start and group_id_end indicate the group id range of each subtree (note this is different from time range)
    extern vec<int, int> visited_subtree, group_id_start, group_id_end;
    // same compressed way of storing trees on disk - but after the rearrangement, so it coexists with fine::flatten_nodes in order to read from there
    extern vec<int, int> flatten_nodes;
    // same as info_offset before to accumulate subtree sizes, but it is map from medium node id to offset map, stored as a vector
    extern vec<int, II> info_offset_rearranged_vec;
    // computed tree size at different time for debug purpose
    extern vec<int, int> tree_sizes;
}

// after we visit the tree in group id order, we call the below function to directly get the active nodes for a given group id
void load_group_medium(vec<int, node> &nodes, int index, int to_index, int T);

// These are functions exposed to Python
extern "C" {
    // rearrange step to instantiate and sort the visible medium cubes
    void rearrange_nodes();
    // visit in group id order for the first time
    void pre_tree_building(int t);
    // then rewrite the compressed node trees to disk
    void rearrange_fine_nodes(int t);
    // write tree size for debug purpose
    void write_tree_size();
    void load_tree_size();
    // Save all results for later reuse (for debug purpose)
    void rearrange_dump();
    void rearrange_load();
    // Clean up all fine and rearrange step related variables to save memory
    void fine_clean_up();
}

#endif // FINE_STEP_H
