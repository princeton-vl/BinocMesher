#include "utils.h"
#include "binoctree.h"

// Compares two coordinates a1 / 2^L1 and a2 / 2^L2
bool compare(int a1, int L1, int a2, int L2) {
    if (L1 <= L2) return a1 << (L2-L1) < a2;
    return a1 < a2 << (L1-L2);
}

// Convert normalized integer coordinates to world coordinates
void compute_coords(T *coords, int *icoords, int L) {
    for (int j = 0; j < 3; j++)
        coords[j] = params::center[j] - params::size / 2 + params::size * icoords[j] / (1 << L);
}

// Compute world coordinates of the center of a hypercube
void compute_center(T *coords, hypercube &v) {
    for (int j = 0; j < 3; j++)
        coords[j] = params::center[j] - params::size / 2 + params::size * (v.coords[j] + 0.5) / (1 << v.L);
}

// Check whether the camera is contained in the hypercube c, and also returns which quadrant it is in if contained
int contain_in(hypercube &c, camera &cam) {
    T *p = cam.location;
    int quad = 0;
    for (int j = 0; j < 3; j++) {
        T lb = params::center[j] - params::size / 2 + params::size * c.coords[j] / (1 << c.L);
        if (p[j] < lb) return -1;
        T ub = lb +  params::size / (1 << c.L);
        if (p[j] >= ub) return -1;
        quad += (p[j] >= (lb + ub) / 2) << j;
    }
    return quad;
}

// Function used in projected_coords(hypercube &c, int k, T *icoords0, T *r, int *inview, T pix_ang, int *contain)
// Compute closest point in hypercube to camera (not necessarily a vertex)
void compute_closest(T *coords, hypercube &v, camera &cam) {
    compute_center(coords, v);
    T step = params::size * 0.5 / (1 << v.L);
    for (int j = 0; j < 3; j++) {
        if (cam.location[j] >= coords[j] + step) coords[j] += step;
        else if (cam.location[j] <= coords[j] - step) coords[j] -= step;
        else {
            coords[j] = cam.location[j];
        }
    }
}

// Project hypercube c onto the kth camera and compute image coordinates and other informations
void projected_coords(hypercube &c, int k, T *icoords0, T *r, int *inview, T pix_ang, int *contain) {
    using namespace params;
    T Pw[3], Pc[3];
    camera &current_cam = cams[k];
    int ct = contain_in(c, current_cam) != -1;
    if (contain != NULL) *contain = ct;
    // If hypercube c contains the kth camera, then use the center; otherwise compute closest point
    if (ct) compute_center(Pw, c);
    else compute_closest(Pw, c, current_cam);
    // Multiply by extrinsics
    for (int i = 0; i < 3; i++) {
        Pc[i] = current_cam.extrinsics[i * 4 + 3];
        for (int j = 0; j < 3; j++) {
            Pc[i] += Pw[j] * current_cam.extrinsics[i * 4 + j];
        }
    }
    T *icoords;
    T icoords1[3];
    if (icoords0 != NULL) icoords = icoords0;
    else icoords = icoords1;
    // Multiply by intrinsics
    for (int i = 0; i < 3; i++) {
        icoords[i] = 0;
        for (int j = 0; j < 3; j++) {
            icoords[i] += Pc[j] * current_cam.intrinsics[i * 3 + j];
        }
    }
    icoords[0] /= icoords[2];
    icoords[1] /= icoords[2];
    // Compute the distance and inview status
    if (r != NULL) {
        *r = sqrt(Pc[0] * Pc[0] + Pc[1] * Pc[1] + Pc[2] * Pc[2]);
        // distances cutoff to prevent extreme subdivision
        *r = smax(*r, min_dist);
        T margin = size * 1.0 / (1<<c.L) / *r / pix_ang;
        // a hypercube is in view if it is entirely in front of the camera and within the image bounds
        *inview = icoords[2] >= 0 && icoords[0] >= -margin && icoords[0] <= current_cam.W + margin && icoords[1] >= -margin && icoords[1] <= current_cam.H + margin;
    }
}

// Function used in projected_size(hypercube c)
// Given hypercube c, compute its projected size with out-of-frustum adjustment
T projected_size(hypercube &c, int k) {
    using namespace params;
    camera &current_cam = cams[k];
    T r;
    int inview;
    T W = current_cam.W;
    // get the angular diameter of a pixel
    T pix_ang = atan(W / 2 / current_cam.intrinsics[0]) * 2 / W;
    // get the target angular diameter specified by target cube size in pixels
    T ang = pix_ang * pixels_per_cube;
    int contain;
    // call the projected_coords function to get r, inview, contain (coordinates are not needed here)
    projected_coords(c, k, NULL, &r, &inview, pix_ang, &contain);
    // To make tree splitting balanced, we have a preprocessing step (in coarse_step.cpp) 
    // that sufficiently splits all hypercubes containing any camera.
    // Therefore, such cubes containing cameras do not need further splitting 
    // and we can directly return 0 here.
    if (contain) return 0;
    // Relative angular diameter = cube size / distance / target angular diameter
    T res = size / (1 << c.L) / r / ang;
    // For out-of-frustum cubes, we use a coarser criterion by a factor of outview_scale
    if (!inview) res /= params::outview_scale;
    return res;
}

// Given hypercube c, considering all the cameras contained in its time span and decide whether it needs time spliting and its maximum projected size
pair<T, bool> projected_size(hypercube c) {
    bool split_time = 0;
    T max_size = 0;
    // We have sorted the camera and can quickly find those in the time span of c from the lookup table
    int k1 = params::cam_lookup[(2<<c.tL) + 2*c.tcoord - 2];
    int k2 = params::cam_lookup[(2<<c.tL) + 2*c.tcoord - 1];
    vec<int, T> sizes;
    sizes.resize(k2-k1);
    // See Fig.6 in the paper. We compute projected sizes (relative angular diameters) for all these cameras first
    for (int k = 0; k < k2-k1; k++) {
        T res = projected_size(c, k1+k);
        sizes[k] = res;
    }
    // Then find the maximum projected size
    for (int k = k1; k < k2; k++)
        max_size = smax(max_size, sizes[k - k1]);
    // We do not split in time if already at max temporal level (precomputed as max_tL)
    if (c.tL < params::max_tL) {
        // We search for a continuous subsequence where all projected sizes are at most half of the maximum projected size
        // This would avoid equivalent spatial splitting over the whole sequence, saving overall memory.
        T block_start = std::numeric_limits<T>::quiet_NaN();
        for (int i = 0; i < sizes.size(); i++) {
            if (sizes[i] * 2 <= max_size) {
                if (std::isnan(block_start)) block_start = params::cams[k1 + i].t;
                else if (params::cams[k1 + i].t - block_start >= params::fading_time) {
                    split_time = 1;
                    break;
                }
            }
            else {
                block_start = std::numeric_limits<T>::quiet_NaN();
            }
        }
    }
    return mp(max_size, split_time);
}

// Split nodes[index] according to whether_split_time, whether_prepared indicates whether nodes is preallocated (when reloading the tree)
int split_node(vec<int, node> &nodes, int index, int whether_split_time, int whether_prepared, int base_size) {
    assert(leaf_node(nodes[index]));
    if (!whether_prepared) base_size = nodes.size();
    if (!whether_split_time) {
        for (int i = 0; i < 8; i++) {
            node *current = &nodes[index];
            current->nxts[i] = whether_prepared? base_size + i: nodes.size();
            node new_node;
            node &node0 = whether_prepared? nodes[base_size + i]: new_node;
            mark_leaf_node(node0);
            node0.c.L = current->c.L + 1;
            for (int k = 0; k < 3; k++) {
                int offset = (i>>k) & 1;
                assign_check(node0.c.coords[k], current->c.coords[k], 2, offset);
            }
            node0.c.tL = current->c.tL;
            node0.c.tcoord = current->c.tcoord;
            if (!whether_prepared) nodes.push_back(node0);
        }
        if (whether_prepared) return 8;
    }
    else {
        for (int i = 0; i < 2; i++) {
            node *current = &nodes[index];
            current->nxts[i] = whether_prepared? base_size + i: nodes.size();
            node new_node;
            node &node0 = whether_prepared? nodes[base_size + i]: new_node;
            mark_leaf_node(node0);
            node0.c.L = current->c.L;
            for (int k = 0; k < 3; k++)
                node0.c.coords[k] = current->c.coords[k];
            node0.c.tL = current->c.tL + 1;
            assign_check(node0.c.tcoord, current->c.tcoord, 2, i);
            if (!whether_prepared) nodes.push_back(node0);
        }
        node *current = &nodes[index];
        current->nxts[2] = -1;
        if (whether_prepared) return 2;
    }
    // We return the number of children in both cases
    if (!whether_prepared) return nodes.size() - base_size;
}

// Given a bipolar edge in the binoctree, this function checks whether this is really an edge without middle vertices
// and returns one of its 8 neighbors neighbor nodes (specified by dirs and tdir); if part of the tree is unloaded, it returns OTHER
int bipolar_edge_neighbor_search(int *coords, int L, int *dirs, int tcoord, int tL, int tdir, int edge_dir, vec<int, node> &nodes, vec<int, int> &mask) {
    assert(L >= 0 && tL >= 0);
    // If the neighbor is out of spatial bounds, return NONE
    for (int p = 0; p < 3; p++) {
        if (edge_dir != p) {
            if (coords[p] < 0 || (coords[p] == 0 && dirs[p] == 0)) return NONE;
            if (coords[p] > (1<<L) || (coords[p] == (1<<L) && dirs[p] == 1)) return NONE;
        }
        else {
            if (!(coords[p] >= 0 && coords[p] < 1<<L)) return NONE;
        }
    }
    // If the neighbor is out of temporal bounds, return TBOUND
    if (tcoord == 0 && tdir == 0) return TBOUND;
    if (tcoord == (1<<tL) && tdir == 1) return TBOUND;
    // Otherwise, traverse the tree to pin down the neighbor node
    // Resulting in an O(tree depth) complexity
    int current = 0;
    node current_node = nodes[0];
    int cnt = 0;
    for(;;) {
        // If the current node is a leaf, we found the neighbor node
        // (OTHER returned if the leaf node represents an unloaded subtree)
        if (leaf_node(current_node)) break;
        int nxt = 0;
        // Otherwise determine which child to go to
        if (is_tsplit(current_node)) {
            int A, B = current_node.c.tL + 1;
            assign_check(A, current_node.c.tcoord, 2, 1);
            if (!compare(tcoord, tL, A, B)) {
                if (compare(A, B, tcoord, tL)) nxt = 1;
                else nxt = tdir;
            }
        }
        else {
            int A, B = current_node.c.L + 1;
            for (int p = 0; p < 3; p++) {
                assign_check(A, current_node.c.coords[p], 2, 1);
                if (edge_dir != p) {
                    if (!compare(coords[p], L, A, B)) {
                        if (compare(A, B, coords[p], L)) nxt += 1<<p;
                        else {
                            nxt += dirs[p]<<p;
                        }
                    }
                }
            }
            assign_check(A, current_node.c.coords[edge_dir], 2, 1);
            // Compares the time span of the current node at the edge direction
            // Returns NONE if the edge is not contained in either (then there are middle vertices)
            bool flag1 = compare(coords[edge_dir], L, A, B);
            bool flag2 = compare(A, B, coords[edge_dir] + 1, L);
            if (flag2 && !flag1) nxt += 1<<edge_dir;
            if (flag1 && flag2) return NONE;
        }
        current = current_node.nxts[nxt];
        current_node = nodes[current];
    }
    // Return OTHER if the leaf node represents an unloaded subtree
    if (!mask[current]) return OTHER;
    return current;
}

// Given an arbitrary edge in the binoctree, this function finds all neighboring "medium_cube"s (see definitions above)
void edge_neighbor_search(int *coords, int L, int *dirs, int tcoord, int tL, int tdir, int edge_dir, vec<int, node> &nodes, vec<int, int> &nodemap, vec<int, medium_cube> &res) {
    // Similar to bipolar_edge_neighbor_search, but now we need to find all neighboring medium_cubes even if the given edge contains middle vertices
    res.clear();
    assert(L >= 0 && tL >= 0);
    for (int p = 0; p < 3; p++) {
        if (edge_dir != p) {
            if (coords[p] < 0 || (coords[p] == 0 && dirs[p] == 0)) return;
            if (coords[p] > (1<<L) || (coords[p] == (1<<L) && dirs[p] == 1)) return;
        }
        else {
            if (!(coords[p] >= 0 && coords[p] < 1<<L)) return;
        }
    }
    if (tcoord == 0 && tdir == 0) return;
    if (tcoord == (1<<tL) && tdir == 1) return;
    // We use a queue to traverse all possible paths down the tree
    // The time complexity is not guaranteed as previous, but we find the runtime is acceptable in practice
    queue<int> nodes_queue;
    nodes_queue.push(0);
    while (!nodes_queue.empty()) {
        node current = nodes[nodes_queue.front()];
        if (leaf_node(current)) {
            // Unlike bipolar_edge_neighbor_search, the tree may still contain virtual grids of medium_cubes glxglxgl
            int gl = grid_node_level(current), resL = current.c.L + gl;
            int st[3], ed[3], mask = (1<<gl) - 1;
            for (int p = 0; p < 3; p++) {
                if (edge_dir != p) {
                    if (dirs[p] == 0) {
                        if (L <= resL) st[p] = (coords[p] << (resL - L)) - 1;
                        else st[p] = (coords[p] - 1) >> (L - resL);
                    }
                    else {
                        if (L <= resL) st[p] = coords[p] << (resL - L);
                        else st[p] = coords[p] >> (L - resL);
                    }
                    ed[p] = st[p] + 1;
                }
                else {
                    if (L <= resL) {
                        st[p] = coords[p] << (resL - L);
                        ed[p] = (coords[p] + 1) << (resL - L);
                    }
                    else {
                        st[p] = coords[p] >> (L - resL);
                        ed[p] = st[p] + 1;
                    }
                }
                st[p] = smax(st[p], current.c.coords[p]<<gl);
                ed[p] = smin(ed[p], (current.c.coords[p]+1)<<gl);
            }
            for (int x = st[0]; x < ed[0]; x++)
                for (int y = st[1]; y < ed[1]; y++)
                    for (int z = st[2]; z < ed[2]; z++) {
                        res.push_back(mp(nodemap[nodes_queue.front()], make_array3(int8_t(x&mask), int8_t(y&mask), int8_t(z&mask))));
                    }
        }
        else {
            int nxt = 0;
            if (is_tsplit(current)) {
                int A, B = current.c.tL + 1;
                assign_check(A, current.c.tcoord, 2, 1);
                if (!compare(tcoord, tL, A, B)) {
                    if (compare(A, B, tcoord, tL)) nxt = 1;
                    else nxt = tdir;
                }
                nodes_queue.push(current.nxts[nxt]);
            }
            else {
                int A, B = current.c.L + 1;
                for (int p = 0; p < 3; p++) {
                    assign_check(A, current.c.coords[p], 2, 1);
                    if (edge_dir != p) {
                        if (!compare(coords[p], L, A, B)) {
                            if (compare(A, B, coords[p], L)) nxt += 1<<p;
                            else nxt += dirs[p]<<p;
                        }
                    }
                }
                assign_check(A, current.c.coords[edge_dir], 2, 1);
                if (compare(coords[edge_dir], L, A, B)) nodes_queue.push(current.nxts[nxt]);
                if (compare(A, B, coords[edge_dir] + 1, L)) nodes_queue.push(current.nxts[nxt + (1<<edge_dir)]);
            }

        }
        nodes_queue.pop();
    }
    return;
}
