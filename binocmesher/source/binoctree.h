#ifndef BINOCTREE_H
#define BINOCTREE_H

// Type aliases for different coordinates and data types
typedef double T;
typedef int8_t eleT;
typedef int8_t timeT;
typedef int8_t lT;
typedef float sdfT;
typedef float spaceT;

// Node marking macros - use negative values in nxts[0] and nxts[1] to encode extra information
#define mark_leaf_node(node) (node).nxts[0] = -2                    // Mark node as leaf
#define mark_grid_node(node, gl) (node).nxts[0] = -2-gl            // Mark node as virual grid at level gl
#define label_node(node, l) (node).nxts[1] = -2-l;                 // Assign node ID l for sequential storage of fine structures

// Node query macros
#define leaf_node(node) ((node).nxts[0] <= -2)                     // Check if node is leaf
#define grid_node_level(node) (-(node).nxts[0] - 2)                // Extract grid level
#define get_node_label(node) (-(node).nxts[1] - 2)                 // Extract node ID

// Safe assignment with overflow protection
#define assign_check(x, y, a, b) {assert((y) < (INT_MAX-(smax(0,b)))/(a)); x=(y)*(a)+(b);}

// Temporal split check
#define is_tsplit(node) ((node).nxts[0] >= 0 && (node).nxts[2] < 0)

// Return values indicating the state of the neighboring nodes of a queried edge - see function bipolar_edge_neighbor_search
#define NONE -1
#define OTHER -2
#define TBOUND -3

// In normalized coordinates, 4D hypercube (3D space + time) spanning [coords[i]/2^L, (coords[i]+1)/2^L] for i=0,1,2 and [tcoord/2^tL, (tcoord+1)/2^tL]
struct hypercube {
    int coords[3];
    timeT tcoord;
    lT L, tL;
};

// 3D spatial vertex at (coords[i]/2^L, (coords[i]+1)/2^L) for i=0,1,2
struct vertex {
    int coords[3];
    lT L;
};

// Binoctree node: hypercube + child indices (8 used for spatial, 2 used for temporal splits)
struct node {
    hypercube c;
    int nxts[8];
};

// Camera iparameters, though location can be infered from extrinsics
struct camera {
    T extrinsics[12], intrinsics[9], t, location[3];
    int H, W;
};

// A medium_cube in the virtual grid technique is represented by the node index and the coordinate within the virtual grid
typedef pair<int, array<int8_t, 3> > medium_cube;
// A medium_cube_ext stores extra information of the occupancy function value
typedef pair<medium_cube, short> medium_cube_ext;

// Parameters for the binoctree and mesh extraction
namespace params {
    extern int n_elements; // number of scene elements
    extern T *center; // center of the scene
    extern T size, tsize; // spatial and temporal scene size
    extern T pixels_per_cube, inv_scale, outview_scale; // angular resolution, invisible parts coarser by inv_scale, out-of-view parts by outview_scale
    extern T min_dist; // distances cutoff to prevent extreme subdivision, angular resolution = node size / max(distance, min_dist)
    // A node spans at least fading_time in time dimension, resulting in a total max_tL temporal levels, and the actual smallest node spans deltaT (>fading_time)
    extern T fading_time, deltaT;
    extern int max_tL;
    extern vec<int, camera> cams; // camera parameters
    extern vec<int, int> cam_lookup; // preprocessed table to look up which cameras are in a specified time range
    extern std::string output_path, log_path; // output and log paths
}

// Compares two cameras' temporal coordinates
inline bool compareCamera(const camera &a, const camera &b) {
    return a.t < b.t;
}

// Compares two coordinates a1 / 2^L1 and a2 / 2^L2
bool compare(int a1, int L1, int a2, int L2);

// Convert normalized integer coordinates to world coordinates
void compute_coords(T *coords, int *icoords, int L);

// Compute world coordinates of the center of a hypercube
void compute_center(T *coords, hypercube &v);

// Check whether the camera is contained in the hypercube c
int contain_in(hypercube &c, camera &cam);

// Project hypercube c onto the k'th camera and compute image coordinates and other informations
void projected_coords(hypercube &c, int k, T *icoords0, T *r, int *inview, T pix_ang=0, int *contain=NULL);

// Given hypercube c, considering all the cameras contained in its time span and decide whether it needs time spliting and its maximum projected size
pair<T, bool> projected_size(hypercube c);

// Split nodes[index] according to whether_split_time, whether_prepared indicates whether nodes is preallocated (when reloading the tree)
int split_node(vec<int, node> &nodes, int index, int whether_split_time, int whether_prepared=0, int base_size=0);

// Given a bipolar edge in the binoctree, this function checks whether this is really an edge without middle vertices
// and returns one of its 8 neighbors neighbor nodes (specified by dirs and tdir); if part of the tree is unloaded, it returns a status indicator
int bipolar_edge_neighbor_search(int *coords, int L, int *dirs, int tcoord, int tL, int tdir, int edge_dir, vec<int, node> &nodes, vec<int, int> &mask);

// Given an arbitrary edge in the binoctree, this function finds all neighboring "medium_cube"s (see definitions above)
void edge_neighbor_search(int *coords, int L, int *dirs, int tcoord, int tL, int tdir, int edge_dir, vec<int, node> &nodes, vec<int, int> &nodemap, vec<int, medium_cube> &res);

#endif // BINOCTREE_H
