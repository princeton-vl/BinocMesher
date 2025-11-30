#ifndef COARSE_STEP_H
#define COARSE_STEP_H

namespace coarse {
    // array of all tree nodes in the coarse step
    extern vec<int, node> nodes;
    // array of all leaf node indices in the coarse step
    extern vec<int, int> leaf_nodes_vector;
    // map from node index to leaf node index
    extern vec<int, int> nodemap;
}

// A coarse node in the priority-queue contains: angular diameter, whether to split in time, and node index
typedef pair<pair<T, int>, int> coarse_node;

// Sort coarse leaf nodes by time coordinate after the coarse step is done
// So that the fine_step can process nodes in temporal order
inline bool compareCoarse(int a, int b) {
    return coarse::nodes[a].c.tcoord * 1.0 / (1 << coarse::nodes[a].c.tL) < coarse::nodes[b].c.tcoord * 1.0 / (1 << coarse::nodes[b].c.tL);
}

// A hypercube containing any camera will have very large angular diameter, making the split process unbalanced
// In practice, we pre-split all hypercubes until those containing the cameras are small enough (see details in the implementation)
int pre_split_check(hypercube &c);

// These are functions exposed to Python
extern "C" {
    // Initialize algorithm parameters from Python input
    int load_parameters(
        T *center, T size, T tsize,
        int n_cams, T *cams,
        T fading_time,
        T pixels_per_cube,
        T pixels_per_cube_coarse,
        T pixels_per_cube_outview,
        T min_dist,
        int n_elements,
        char *output_path
    );
    // Main coarse step function
    int run_coarse(int n_coarse_nodes);
    // Save coarse tree structure to disk for later reuse (for debug purpose)
    void coarse_dump();
    // Load previously saved coarse tree structure from disk
    int coarse_load();
}

#endif // COARSE_STEP_H
