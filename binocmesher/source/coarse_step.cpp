#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"

// Based on the cut_off distance min_dist
// check whether a hypercube c containing a camera needs further subdivision
bool needs_subdivide(hypercube &c, camera &cam){
    using namespace params;
    T W = cam.W;
    T pix_ang = atan(W / 2 / cam.intrinsics[0]) * 2 / W;
    T ang = pix_ang * pixels_per_cube;
    T min_size = min_dist * ang;
    return size / (1<<c.L) / min_size > 1;
}

// A hypercube containing any camera will have very large angular diameter, making the split process unbalanced
// In practice, we pre-split all hypercubes until those containing the cameras are small enough
// 0: no split; 1: time split; 2: spatial split
int pre_split_check(hypercube &c) {
    using namespace params;
    // We have sorted the camera and can quickly find those in the time span of c from the lookup table
    int k1 = params::cam_lookup[(2<<c.tL) + 2*c.tcoord - 2];
    int k2 = params::cam_lookup[(2<<c.tL) + 2*c.tcoord - 1];
    int quads[8]={0};
    int sum = 0;
    bool subdivide = 0;
    for (int k = k1; k < k2; k++) {
        int quad = contain_in(c, cams[k]);
        if (quad != -1) {
            // sum counts non-empty quads
            if (quads[quad] == 0) {
                sum++;
                quads[quad] = 1;
            }
            // If the cube contains any camera and not yet reached min_dist criterion, we need to further subdivide
            subdivide |= needs_subdivide(c, cams[k]);
        }
    }
    // If the cube does not contain any camera or any subdivide flag for any camera, no need to pre-split
    if (sum == 0 || !subdivide) return 0;
    // If there are cameras in multiple quads, we split temporally to allow spatially different subtrees
    if (sum >= 2 && c.tL < max_tL) return 1;
    // Otherwise, we just split spatially first
    return 2;
}

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
    ) {
        params::center = center;
        params::size = size;
        params::tsize = tsize;
        params::pixels_per_cube = pixels_per_cube;
        params::inv_scale = pixels_per_cube_coarse / pixels_per_cube;
        params::outview_scale = pixels_per_cube_outview / pixels_per_cube_coarse;
        params::min_dist = min_dist;
        params::n_elements = n_elements;
        std::stringstream output_path_ss;
        output_path_ss << output_path;
        params::output_path = output_path_ss.str();
        params::cams.clear();
        params::cam_lookup.clear();
        params::log_path = params::output_path + "/log.txt";
        for (int i = 0; i < n_cams; i++) {
            const int CAM_BLOCK_SIZE = 27;
            camera cam;
            for (int j = 0; j < 12; j++) cam.extrinsics[j] = cams[i * CAM_BLOCK_SIZE + j];
            for (int j = 0; j < 9; j++) cam.intrinsics[j] = cams[i * CAM_BLOCK_SIZE + 12 + j];
            cam.H = int(cams[i * CAM_BLOCK_SIZE + 21]);
            cam.W = int(cams[i * CAM_BLOCK_SIZE + 22]);
            cam.t = cams[i * CAM_BLOCK_SIZE + 23];
            for (int j = 0; j < 3; j++) cam.location[j] = cams[i * CAM_BLOCK_SIZE + 24 + j];
            params::cams.push_back(cam);
        }
        // Sort the cameras by time
        sort(params::cams.begin(), params::cams.end(), compareCamera);
        // Get the minimum frame duration (In a regular video the frame duration is constant)
        T frame_duration = std::numeric_limits<T>::infinity();
        for (int i = 1; i < params::cams.size(); i++) {
            if (params::cams[i].t != params::cams[i - 1].t) {
                T duration = params::cams[i].t - params::cams[i - 1].t;
                frame_duration = smin(duration, frame_duration);
            }
        }
        // Ensure fading_time is at least one frame duration
        fading_time = smax(fading_time, frame_duration);
        // Preprocess the cam_lookup table for quickly finding cameras in a time range
        for (int L = 0; ; L++) {
            for (int i = 0; i < 1 << L; i++) {
                camera tmp;
                tmp.t = tsize * i / (1<<L);
                int k1 = lower_bound(params::cams.begin(), params::cams.end(), tmp, compareCamera) - params::cams.begin();
                tmp.t = tsize * (i+1) / (1<<L);
                int k2 = lower_bound(params::cams.begin(), params::cams.end(), tmp, compareCamera) - params::cams.begin();
                params::cam_lookup.push_back(k1);
                params::cam_lookup.push_back(k2);
            }
            // L is the maximum temporal level
            params::max_tL = L;
            params::deltaT = params::tsize / (2<<params::max_tL);
            if (fading_time * (1<<(L+1)) >= tsize) break;
        }
        params::fading_time = fading_time;
        return 1 << params::max_tL;
    }

    // Main coarse step function
    int run_coarse(int n_coarse_nodes) {
        using namespace params;
        using namespace coarse;
        FILE *log = fopen(params::log_path.c_str(), "a");
        auto pre_queue = queue<int>();
        priority_queue<coarse_node> nodes_heap;
        queue<coarse_node> nodes_vector_tmp;
        // Create a root node
        node root;
        mark_leaf_node(root);
        memset(root.c.coords, 0, 3 * sizeof(int));
        root.c.tcoord = 0;
        root.c.L = 0;
        root.c.tL = 0;
        nodes.clear();
        nodes.push_back(root);
        // A hypercube containing any camera will have very large angular diameter, making the split process unbalanced
        // In practice, we pre-split all hypercubes until those containing the cameras are small enough
        // A QUEUE is used for BFS pre-splitting
        MEASURE_TIME("run_coarse part 1", 1, {
            pre_queue.push(0);
            while (!pre_queue.empty()) {
                int front = pre_queue.front();
                int split = pre_split_check(nodes[front].c);
                if (split > 0) {
                    int n_child = split_node(nodes, front, split == 1);
                    for (int i = 0; i < n_child; i++) pre_queue.push((int)nodes.size() - 1 - i);
                }
                pre_queue.pop();
            }
        });
        MEASURE_TIME("run_coarse part 2", 1, {
            // Put all current leaf nodes into a PRIORITY QUEUE for further splitting
            for (int i = 0; i < nodes.size(); i++) {
                if (leaf_node(nodes[i])) {
                    nodes_heap.push(mp(projected_size(nodes[i].c), i));
                }
            }
            fprintf(log, "%d %d\n", (int)nodes_heap.size(), n_coarse_nodes);
            // Further split nodes until reaching the threshold n_coarse_nodes
            // or all nodes are small enough
            while (nodes_heap.size() < n_coarse_nodes) {
                coarse_node top = nodes_heap.top();
                int split_time = top.first.second;
                if (top.first.first <= inv_scale) {
                    break;
                }
                nodes_heap.pop();
                int i0 = nodes.size();
                int n_child = split_node(nodes, top.second, split_time);
                for (int i = 0; i < n_child; i++) nodes_heap.push(mp(projected_size(INDEX(nodes, (int)nodes.size() - 1 - i).c), (int)nodes.size() - 1 - i));
            }
            fprintf(log, "nodes before tcomp: %d\n", int(nodes_heap.size()));
        });
        int max_gl = 0;
        MEASURE_TIME("run_coarse part 3", 1, {
            leaf_nodes_vector.clear();
            while (!nodes_heap.empty()) {
                coarse_node top = nodes_heap.top();
                nodes_vector_tmp.push(top);
                nodes_heap.pop();
            }
            // In virtual grid technique, we mark leaf nodes as virtual grids at level gl
            // To make things easy, we don't split hypercube temporally, i.e., a grid is only glxglxglx1
            // Therefore, we need to finish all remaining temporal splits first
            // (which may create new nodes exceeding the target n_coarse_nodes, but only marginally)
            while (!nodes_vector_tmp.empty()) {
                auto front = nodes_vector_tmp.front();
                auto c = nodes[front.second].c;
                int split_time = front.first.second;
                int gl = int_log(front.first.first / inv_scale);
                max_gl = smax(max_gl, gl);
                if (gl > 1 && split_time) {
                    int i0 = nodes.size();
                    int n_child = split_node(nodes, front.second, 1);
                    for (int i = 0; i < n_child; i++) nodes_vector_tmp.push(mp(projected_size(INDEX(nodes, (int)nodes.size() - 1 - i).c), (int)nodes.size() - 1 - i));
                }
                else {
                    mark_grid_node(INDEX(nodes, front.second), gl);
                    leaf_nodes_vector.push_back(front.second);
                }
                nodes_vector_tmp.pop();
            }
            // Sort coarse leaf nodes by time coordinate after the coarse step is done
            // So that the fine_step can process nodes in temporal order
            sort(leaf_nodes_vector.begin(), leaf_nodes_vector.end(), compareCoarse);
            nodemap.resize(nodes.size());
            for (int i = 0; i < leaf_nodes_vector.size(); i++)
                INDEX(nodemap, INDEX(leaf_nodes_vector, i)) = i;
        });
        fprintf(log, "max_gl: %d\n", max_gl);
        fprintf(log, "nodes after tcomp: %d\n", int(leaf_nodes_vector.size()));
        fclose(log);
        return leaf_nodes_vector.size();
    }

    // Save coarse tree structure to disk for later reuse (for debug purpose)
    void coarse_dump() {
        using namespace coarse;
        std::stringstream filename;
        filename << params::output_path << "/coarse_cubes.bin";
        FILE *outfile = fopen(filename.str().c_str() , "wb" );
        write_vec(outfile, nodes);
        write_vec(outfile, leaf_nodes_vector);
        write_vec(outfile, nodemap);
        fclose(outfile);
    }

    // Load previously saved coarse tree structure from disk
    int coarse_load() {
        using namespace coarse;
        std::stringstream filename;
        filename << params::output_path << "/coarse_cubes.bin";
        FILE *infile = fopen(filename.str().c_str() , "rb" );
        read_vec(infile, nodes);
        read_vec(infile, leaf_nodes_vector);
        read_vec(infile, nodemap);
        fclose(infile);
        return leaf_nodes_vector.size();
    }
}