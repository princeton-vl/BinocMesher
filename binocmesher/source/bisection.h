#ifndef BISECTION_H
#define BISECTION_H

// ID for center vertex of hypercube consists of node id and group id
typedef pair<int, int8_t> HVID;

// A hyper cube has several bipolar edges on it, we bisect by shrinking the cube until all bipolar edges disappear
// gridID is to identify the unique bipolar edge vertices on it
typedef pair<array<int, 3>, int> gridID;

// Hash function for HVTable
struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);

        if (hash1 != hash2) {
            return hash1 ^ hash2;
        }
        return hash1;
    }
};

// A dual polyhedra in 4D space consists of 8 center vertices (and the element id)
typedef pair<array<HVID, 8>, eleT> HP;

namespace bisection {
    // We bisect by shrinking the cube until all bipolar edges disappear
    // Because the number of queried vertices are too many, we process in batches (groups) of size bisection_group
    extern int bisection_group;
    extern vec<ll, int> query_cnt;
    extern ll last_head, current_head;
    extern vec<int, T> lefts, rights;
    extern vec<int, int> starts;
    extern vec<int, int> center_indices;
    // array of 4D polyhedra
    extern vec<int, HP> hyperpolys;
    // map from node ID to its center vertex ID
    extern vec<ll, int> vertex_map;
}

// computed center vertex of hypercube, consists of 4D position, time span of the hypercube, and its inview tag
// The later two are used for slicing the edge in non-linear way
typedef pair<pair<array<spaceT, 3>, array<timeT, 2> >, int8_t> HV;

namespace bisection {
    // computed center vertices of hypercube
    extern vec<int, HV> computed_vertices;
}

// map from ID to data of center vertices of hypercube
typedef unordered_map<HVID, HV, hash_pair> HVTable;
namespace bisection {
    extern HVTable hypervertices;
    extern HVTable hypervertices_g[N_THREAD];
    // vector version
    extern vec<int, pair<HVID, HV> > hypervertices_vec;
}

// Functions to load the 4D polyhedra and center vertices indices
void load_hyperpolys(int t);

// Functions to load center vertices information
void load_vertices(int t);

// These are functions exposed to Python
extern "C" {
    // Initialize bisection module
    void bisection_init(int bisection_group);
    // Initialize bisection for group t
    void bisection_init_t(int t);
    // Get the number of vertices in the current batch to query
    // For each hypercube, we keep shrinking the cube and query the shrinked bipolar edges until no bipolar edge exists
    // In this way we can find a vertex on the isosurface that is quite close to the geometric center of the hypercube
    int bisection_hypermesh_verts(int t, int *cnts, int *center_cnts);
    // Output the query center vertices to Python
    void bisection_hypermesh_verts_output_center(T *center_xyz);
    // Output the query cube-surface vertices to Python, called repeated for the number of iterations
    void bisection_hypermesh_verts_output(T *xyz, int finishing);
    // According to queried occupancy values, decide to the left/right bounds of the cube size for next iteration
    void bisection_hypermesh_verts_iter(sdfT *sdfs, sdfT *center_sdfs);
    // Compute the final center vertex after all iterations are done
    void bisection_hypermesh_verts_finishing(int t, sdfT *sdfs, sdfT *center_sdfs);
    // Write final 4D mesh of group t to disk
    void write_final_hypermesh(int t);
    // Function to get the number of vertices in the current loaded group for statistics
    int verts_count(int t);
    // Clean up bisection module
    void bisection_clean_up();
}

#endif // BISECTION_H
