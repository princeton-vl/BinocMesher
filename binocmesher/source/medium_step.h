#ifndef MEDIUM_STEP_H
#define MEDIUM_STEP_H

// A medium_cube is the subcube in the virtual grid technique
// The medium step is to find all medium_cubes that intersect the surface through propagation
namespace medium {
    // all_cubes_size is the number of medium_cubes in the current iteration
    extern int all_cubes_size;
    // all_cubes stores all medium_cubes up to the current iteration
    extern vec<int, medium_cube_ext> all_cubes;
    // cubes_entry maintain the current active list of medium_cubes
    extern vec<int, medium_cube> cubes_entry;
    // The c++ side need to generate an array of vertices to python to query their occupancy function values
    extern vec<int, vertex> output_vertices;
    // The index to write back the queried occupancy function values into the virtual grid
    extern vec<int, II> output_vertices_index;
    // These are variables used in multi-thread propagation
    extern vec<int, medium_cube_ext> all_cubes_buffer[N_THREAD];
    extern vec<int, medium_cube> cubes_entry_buffer[N_THREAD], cubes_queue[N_THREAD];
    extern vec<int, int> cubes[N_THREAD], vertices[N_THREAD], cubes_nonzero[N_THREAD], vertices_nonzero[N_THREAD];
    extern vec<int, int> vertices_index[N_THREAD], cubes_index[N_THREAD];
    extern vec<int, II> output_vertices_index_buffer[N_THREAD];
    extern vec<int, vertex> output_vertices_buffer[N_THREAD];
    extern int start_node[N_THREAD];
    // The array visible indicate whether each medium_cube is visible (1) or occluded (0) to any camera within its time window
    extern vec<int, int> visible;
    // The array/set of visible and occluded medium_cubes after visibility filtering
    extern vec<int, medium_cube> visible_cubes, occluded_cubes;
    extern set<medium_cube> visible_set, occluded_set;
    // However, the visible set may still expand in the later fine step's propagation
    // current_size and sorted_size track the size of the visible set
    extern int current_size, sorted_size;
}

// These are functions exposed to Python
extern "C" {
    // Initialize medium_cubes seeds for the flooding algorithm
    int medium_seeding(int stride);
    // The entire medium step is composed of multiple "medium_loop"s and "medium_iteration_end"s
    // medium_loop only propagates within each coarse node, while medium_iteration_end does the cross-node propagation
    // A medium_loop function (core.cpp) will call medium_iteration_init once, and call medium_iteration_output and medium_iteration_regular multiple times
    // See the core.py for the detailed logic
    int medium_iteration_init(int start_node, int end_node, int *update, int timing);
    void medium_iteration_output(T *xyz);
    int medium_iteration_regular(sdfT *sdf, int timing);
    bool medium_iteration_end(int timing);
    // Save all medium_cubes to disk for later reuse (for debug purpose)
    void medium_dump();
    void medium_load();
    // Find whether each medium_cube is visible (1) or occluded (0) to any camera within its time window
    // Simplify_occluded indicates whether consider cubes behind other cubes as occluded (usually true for opaque materials)
    void visibility_filter(bool simplify_occluded, int relax_margin, int boundary_margin, int relax_iters);
    // Save visibility filtering result to disk for later reuse (for debug purpose)
    void visibility_filter_dump();
    void visibility_filter_load();
    // We cannot write an stl::set to disk directly, so we need some post processing
    int visibility_filter_ending();
    // Clean up all medium step related variables to save memory
    void medium_clean_up();
}

#endif // MEDIUM_STEP_H
