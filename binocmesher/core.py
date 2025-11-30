from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import gc
import subprocess
import os
import gin
import trimesh
import shutil

from .utils.interface import AC, POINTER, AsDouble, AsFloat, AsInt, AsBool, c_bool, c_double, c_float, c_int32, c_char_p, load_cdll, register_func
from .utils.timer import Timer

@gin.configurable()
class BinocMesher:
    def __init__(self,
        cameras,
        bounds,
        slicing_time,
        pixels_per_cube=10,
        pixels_per_cube_coarse=30,
        pixels_per_cube_outview=120,
        min_dist=0.1,
        enclosed=False,
        simplify_occluded=True,
        relax_margin=2,
        boundary_margin=2,
        relax_iters=0,
        n_coarse_nodes=10000000,
        bisection_iters=3,
        fading_time=1,
        seed_stride=10,
        medium_group=100000,
        fine_group=10000,
        bisection_group=10000000,
        path=None,
        min_t_offset=0,
        use_alignment=False,
    ):
        dll = load_cdll(str(Path(__file__).parent.resolve()/"lib"/"core.so"))
        self.float_type = c_double
        self.np_float_type = np.float64
        self.AF = AsDouble
        self.sdf_float_type = c_float
        self.sdf_np_float_type = np.float32
        self.sdf_AF = AsFloat
        self.bounds = bounds
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        cam_poses, Ks, Hs, Ws, Ts = cameras
        min_t = min(Ts) - min_t_offset

        Ts = [(t - min_t) for t in Ts]
        if use_alignment:
            self.tsize = 2 ** np.ceil(np.log2((max(Ts) + 1e-5) / fading_time)) * fading_time + 1e-5
        else:
            self.tsize = max(Ts) + 1e-5

        cam_poses = np.array(cam_poses)
        Ks = np.array(Ks)
        Hs = np.array(Hs)
        Ws = np.array(Ws)
        Ts = np.array(Ts)

        self.n_cameras = len(cam_poses)
        self.slicing_time = slicing_time - min_t

        cam_block_size = 27
        self.cameras = np.zeros(cam_block_size * self.n_cameras, dtype=self.np_float_type)
        for i in range(self.n_cameras):
            self.cameras[cam_block_size * i: cam_block_size * (i+1)] = np.concatenate([
                np.linalg.inv(cam_poses[i])[:3, :4].reshape(-1),
                Ks[i].reshape(-1), [Hs[i]], [Ws[i]], [Ts[i]], cam_poses[i][:3, 3]
            ]).astype(self.np_float_type)

        self.pixels_per_cube = self.np_float_type(pixels_per_cube)
        self.pixels_per_cube_coarse = self.np_float_type(pixels_per_cube_coarse)
        self.pixels_per_cube_outview = self.np_float_type(pixels_per_cube_outview)
        self.min_dist = self.np_float_type(min_dist)

        self.center = np.array([(bounds[0]+bounds[1]) / 2, (bounds[2]+bounds[3]) / 2, (bounds[4]+bounds[5]) / 2], self.np_float_type)
        self.size = self.np_float_type(max(max(bounds[1] - bounds[0], bounds[3] - bounds[2]), bounds[5] - bounds[4]) * 1.1)


        self.bisection_iters = bisection_iters
        self.enclosed = enclosed
        self.simplify_occluded = simplify_occluded
        self.relax_margin = relax_margin
        self.boundary_margin = boundary_margin
        self.relax_iters = relax_iters
        self.n_coarse_nodes = n_coarse_nodes
        self.fading_time = fading_time
        self.seed_stride = seed_stride
        self.medium_group = medium_group
        self.fine_group = fine_group
        self.bisection_group = bisection_group
        
        register_func(self, dll, "load_parameters", [
            POINTER(self.float_type), self.float_type, self.float_type,
            c_int32, POINTER(self.float_type),
            self.float_type, self.float_type, self.float_type, self.float_type, self.float_type,
            c_int32, c_char_p,
        ], c_int32)
        register_func(self, dll, "run_coarse", [c_int32], c_int32)
        register_func(self, dll, "coarse_dump")
        register_func(self, dll, "coarse_load", [], c_int32)

        register_func(self, dll, "medium_seeding", [c_int32], c_int32)
        register_func(self, dll, "medium_iteration_init", [c_int32, c_int32, POINTER(c_int32), c_int32], c_int32)
        register_func(self, dll, "medium_iteration_regular", [POINTER(self.sdf_float_type), c_int32], c_int32)
        register_func(self, dll, "medium_iteration_output", [POINTER(self.float_type)])
        register_func(self, dll, "medium_iteration_end", [c_int32], c_bool)
        register_func(self, dll, "medium_dump")
        register_func(self, dll, "medium_load")
        register_func(self, dll, "visibility_filter", [c_bool, c_int32, c_int32, c_int32])
        register_func(self, dll, "visibility_filter_ending", [], c_int32)
        register_func(self, dll, "visibility_filter_dump")
        register_func(self, dll, "visibility_filter_load")
        register_func(self, dll, "medium_clean_up")

        register_func(self, dll, "fine_init", [c_int32])
        register_func(self, dll, "fine_iteration", [c_int32, c_int32], c_int32)
        register_func(self, dll, "fine_iteration_output", [POINTER(self.float_type)])
        register_func(self, dll, "fine_iteration_propagate", [POINTER(self.sdf_float_type)], c_int32)
        register_func(self, dll, "fine_dump")
        register_func(self, dll, "fine_load")

        register_func(self, dll, "rearrange_nodes", [], c_int32)
        register_func(self, dll, "pre_tree_building", [c_int32])
        register_func(self, dll, "rearrange_fine_nodes", [c_int32])
        register_func(self, dll, "write_tree_size")
        register_func(self, dll, "load_tree_size")
        register_func(self, dll, "rearrange_dump")
        register_func(self, dll, "rearrange_load")
        register_func(self, dll, "fine_clean_up")

        register_func(self, dll, "dual_contouring_init")
        register_func(self, dll, "load_group", [c_int32, c_bool, c_bool], c_int32)
        register_func(self, dll, "load_active_group_output", [POINTER(self.float_type), c_int32, c_int32])
        register_func(self, dll, "constructing_meshes", [POINTER(self.sdf_float_type), c_int32])
        register_func(self, dll, "load_unfinalized_edges")
        register_func(self, dll, "save_unfinalized_edges")

        register_func(self, dll, "bisection_init", [c_int32])
        register_func(self, dll, "verts_count", [c_int32], c_int32)
        register_func(self, dll, "bisection_init_t", [c_int32])
        register_func(self, dll, "bisection_hypermesh_verts", [c_int32, POINTER(c_int32), POINTER(c_int32)], c_int32)
        register_func(self, dll, "bisection_hypermesh_verts_output_center", [POINTER(self.float_type)])
        register_func(self, dll, "bisection_hypermesh_verts_output", [POINTER(self.float_type), c_int32])
        register_func(self, dll, "bisection_hypermesh_verts_iter", [POINTER(self.sdf_float_type), POINTER(self.sdf_float_type)])
        register_func(self, dll, "bisection_hypermesh_verts_finishing", [c_int32, POINTER(self.sdf_float_type), POINTER(self.sdf_float_type)])
        register_func(self, dll, "write_final_hypermesh", [c_int32])
        register_func(self, dll, "bisection_clean_up")
        
        register_func(self, dll, "slicing_preprocess")
        register_func(self, dll, "run_slicing", [self.float_type, POINTER(c_int32), POINTER(c_int32), c_bool])
        register_func(self, dll, "slicing_output", [c_int32, POINTER(self.float_type), POINTER(c_int32), POINTER(c_int32)])
        register_func(self, dll, "slicing_clean_up")


    def kernel_caller(self, kernels, XYZ_all, step=2**31-1):
        with Timer("SDF call", write_to_file=self.path/"sdf_calls.txt"):
            n_XYZ = len(XYZ_all)
            with open(self.path/"sdf_calls.txt", "a") as f:
                f.write(f"#queries: {n_XYZ}\n")
            if n_XYZ == 0: return np.zeros((0, len(kernels)), dtype=self.sdf_np_float_type)
            sdfs = []
            for i in range(0, n_XYZ, step):
                XYZ = XYZ_all[i: i+step]
                sdfs_i = []
                if self.enclosed:
                    out_bound = np.zeros(len(XYZ), dtype=bool)
                    for c in range(3):
                        out_bound |= XYZ[:, c] <= self.bounds[c*2]
                        out_bound |= XYZ[:, c] >= self.bounds[c*2+1]
                for kernel in kernels:
                    sdf = kernel(XYZ)
                    if self.enclosed: sdf[out_bound] = 1
                    sdfs_i.append(sdf)
                sdfs.append(np.stack(sdfs_i, -1).astype(self.sdf_np_float_type))
            sdfs = np.concatenate(sdfs, 0)
        return sdfs
    
    def medium_loop(self, n_coarse_nodes, kernels, pbar=False):
        entry = 0
        rg = tqdm(range(0, n_coarse_nodes, self.medium_group)) if pbar else range(0, n_coarse_nodes, self.medium_group)
        for start_node in rg:
            end_node = min(n_coarse_nodes, start_node + self.medium_group)
            update = np.zeros(1, dtype=np.int32)
            n = self.medium_iteration_init(start_node, end_node, AsInt(update), pbar)
            entry += update[0]
            iters = 0
            while n >= 0:
                positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                self.medium_iteration_output(self.AF(positions))
                sdf = self.kernel_caller(kernels, positions)
                sdf = sdf.min(axis=-1)
                sdf = AC(sdf)
                n = self.medium_iteration_regular(self.sdf_AF(sdf), pbar and iters==0)
                iters += 1
        return entry

    def __call__(self, kernels, debug=0):
        skip_save0 = True
        skip_save1 = True
        path = self.path
        n_elements = len(kernels)
        with Timer("load_parameters"):
            time_slices = self.load_parameters(
                self.AF(self.center), self.size, self.tsize,
                self.n_cameras, self.AF(self.cameras),
                self.fading_time,
                self.pixels_per_cube, 
                self.pixels_per_cube_coarse,
                self.pixels_per_cube_outview,
                self.min_dist,
                n_elements,
                str(path).encode('utf-8'),
            )
 
        if not os.path.exists(str(path / "slicing_preprocess.finish")):
            files_to_delete = list(path.glob("*.txt"))
            for file_path in files_to_delete:
                file_path.unlink()
            (path/"fine_nodes").mkdir(exist_ok=True)
            (path/"rearranged_fine_nodes").mkdir(exist_ok=True)
            (path/"rearranged_nodemap").mkdir(exist_ok=True)
            (path/"graphs").mkdir(exist_ok=True)
            (path/"bip_edges").mkdir(exist_ok=True)
            (path/"constructing").mkdir(exist_ok=True)
            (path/"saved_graphs").mkdir(exist_ok=True)
            (path/"saved_bip_edges").mkdir(exist_ok=True)
            (path/"bisection").mkdir(exist_ok=True)
            (path/"computed_vertices").mkdir(exist_ok=True)
            (path/"hypervertices").mkdir(exist_ok=True)
            (path/"hyperpolys").mkdir(exist_ok=True)
            (path/"processed_hyperpolys").mkdir(exist_ok=True)

            # Coarse step that split the node up to n_coarse_nodes and mark leaf nodes using virtual grid
            if not os.path.exists(str(path / "coarse_cubes.bin")):
                with Timer("coarse step"):
                    n_coarse_nodes = self.run_coarse(self.n_coarse_nodes)
                if not skip_save0: self.coarse_dump()
            else:
                n_coarse_nodes = self.coarse_load()
            # A medium_cube is the subcube in the virtual grid technique
            # The medium step is to find all medium_cubes that intersect the surface through propagation
            if not os.path.exists(str(path / "medium_cubes.bin")):
                with Timer("medium step"):
                    n_seed = self.medium_seeding(self.seed_stride)
                    target_entry = int(np.log(1+self.medium_loop(n_coarse_nodes, kernels, pbar=True)))
                    end = self.medium_iteration_end(1)
                    if not end:
                        with tqdm(total=target_entry) as pbar:
                            last_time = 0
                            while True:
                                entry = int(np.log(1+self.medium_loop(n_coarse_nodes, kernels)))
                                pbar.update(target_entry - entry - last_time)
                                last_time = target_entry - entry
                                end = self.medium_iteration_end(0)
                                if end:
                                    pbar.update(target_entry - last_time)
                                    break
                if not skip_save0: self.medium_dump()
            else:
                self.medium_load()
            # Find whether each medium_cube is visible (1) or occluded (0) to any camera within its time window
            # simplify_occluded indicates whether consider cubes behind other cubes as occluded (usually true for opaque materials)
            if not os.path.exists(str(path / "visibility.bin")):
                with Timer("visibility filter"):
                    self.visibility_filter(self.simplify_occluded, self.relax_margin, self.boundary_margin, self.relax_iters)
                if not skip_save0: self.visibility_filter_dump()
                self.medium_clean_up()
            else:
                self.visibility_filter_load()
            n_vis_mc = self.visibility_filter_ending()
            # The fine step is to split medium_cubes and do potential flooding iteratively
            self.fine_init(self.fine_group)
            if not os.path.exists(str(path / "fine.bin")):
                with Timer("fine step"), tqdm(total=n_vis_mc) as pbar:
                    start_mc = 0
                    while True:
                        end_mc = min(n_vis_mc, start_mc + self.fine_group)
                        n = self.fine_iteration(start_mc, end_mc)
                        positions = AC(np.zeros((n, 3), dtype=self.np_float_type))
                        self.fine_iteration_output(self.AF(positions))
                        sdf = AC(self.kernel_caller(kernels, positions).min(axis=-1))
                        inc = self.fine_iteration_propagate(self.sdf_AF(sdf))
                        finish = False
                        if inc == -1:
                            inc = 0
                            finish = True
                        pbar.n = end_mc
                        start_mc = end_mc
                        if finish: break
                        n_vis_mc += inc
                        pbar.total += inc
                if not skip_save0: self.fine_dump()
            else:
                with Timer("load fine"):
                    self.fine_load()
            # after the fine step, there is a rearrange step to instantiate and sort the visible medium cubes in temporal order
            # and them the medium nodes are assigned group ids
            if not os.path.exists(str(path / "rearrange.bin")):
                with Timer("rearrange nodes"):
                    self.rearrange_nodes()
                with Timer("pre tree building"):
                    for i in tqdm(range(time_slices)):
                        self.pre_tree_building(i)
                if not skip_save0: self.rearrange_dump()
            else:
                with Timer("load rearrange"):
                    self.rearrange_load()
            if not os.path.exists(str(path / "rearrange_fine_nodes.finish")):
                with Timer("rearrange fine nodes"):
                    for i in tqdm(range(time_slices)):
                        self.rearrange_fine_nodes(i)
                (path / "rearrange_fine_nodes.finish").touch()
                self.write_tree_size()
                self.fine_clean_up()
            else:
                with Timer("load tree size"):
                    self.load_tree_size()
            # To explain first: the scene is repsented as a union of several scene elements (mountain, ground, etc) for a good reason
            # We treat the overall occupancy function as their "OR" results in all previous steps
            # But in this step, we look into individual elements and do dual contouring each element
            for t in tqdm(range(time_slices), desc='dual contouring'):
                if not os.path.exists(str(path / f"constructing/{t}.finish")) and not os.path.exists(str(path / "constructing/finish")):
                    if t == 0:
                        files_to_delete = list(path.glob("graphs/*")) + list(path.glob("bip_edges/*")) + list(path.glob("saved_graphs/*")) + list(path.glob("saved_bip_edges/*"))
                        files_to_delete += list(path.glob("unfinalized_edges.bin"))
                        for file_path in files_to_delete:
                            file_path.unlink()
                        self.dual_contouring_init()
                    else:
                        if not skip_save1:
                            self.load_unfinalized_edges()
                            files_to_activate = list(path.glob("saved_graphs/*")) + list(path.glob("saved_bip_edges/*"))
                            for file_path in files_to_activate:
                                dir = Path(file_path).parent
                                filename = Path(file_path).name
                                shutil.copyfile(file_path, str(dir / filename[6:]))
                    with Timer("load_group", write_to_file=path/"log.txt"):
                        n = self.load_group(t, 1, 0)
                    with Timer("load_active_group_output", write_to_file=path/"log.txt"):
                        if n > 0:
                            S = int(1e8)
                            sdfs = []
                            for i in range(0, n, S):
                                start = i
                                end = min(i + S, n)
                                positions = AC(np.zeros((end - start, 3), dtype=self.np_float_type))
                                self.load_active_group_output(self.AF(positions), start, end)
                                sdfs.append(self.kernel_caller(kernels, positions))
                            sdf = AC(np.concatenate(sdfs, 0))
                        else:
                            sdf = np.zeros((0, n_elements))
                    with Timer("constructing_meshes", write_to_file=path/"log.txt"):
                        self.constructing_meshes(self.sdf_AF(sdf), debug)
                    if not skip_save1:
                        self.save_unfinalized_edges()
                        files_to_save = list(path.glob("graphs/*")) + list(path.glob("saved_bip_edges/*"))
                        for file_path in files_to_save:
                            dir = Path(file_path).parent
                            filename = Path(file_path).name
                            shutil.copyfile(file_path, str(dir / ("saved_" + filename)))
                        (path / f"constructing/{t}.finish").touch()
            (path / f"constructing/finish").touch()
            # cnt = 0
            # for t in tqdm(range(time_slices)):
            #     cnt += self.verts_count(t)
            # The bisection step is to refine the 4D mesh vertices within the hypercubes
            self.bisection_init(self.bisection_group)
            for t in tqdm(range(time_slices), desc='bisection'):
                if not os.path.exists(str(path / f"bisection/{t}.finish")):
                    with Timer("bisection_init_t", write_to_file=path/"log.txt"):
                        self.bisection_init_t(t)
                    cnts = np.zeros(n_elements, dtype=np.int32)
                    center_cnts = np.zeros(n_elements, dtype=np.int32)
                    while True:
                        with Timer("bisection group init", write_to_file=path/"log.txt"):
                            if not self.bisection_hypermesh_verts(t, AsInt(cnts), AsInt(center_cnts)): break
                            positions = AC(np.zeros((cnts.sum(), 3),  dtype=self.np_float_type))
                            center_positions = AC(np.zeros((center_cnts.sum(), 3),  dtype=self.np_float_type))
                            self.bisection_hypermesh_verts_output_center(self.AF(center_positions))
                            center_sdfs = []
                            for e in range(n_elements):
                                center_sdfs.append(self.kernel_caller(kernels[e:e+1], center_positions[center_cnts[:e].sum(): center_cnts[:e+1].sum()]))
                            center_sdfs = AC(np.concatenate(center_sdfs))
                        for it in range(self.bisection_iters):
                            self.bisection_hypermesh_verts_output(self.AF(positions), 0)
                            sdfs = np.zeros(positions.shape[0], dtype=self.sdf_np_float_type)
                            for e in range(n_elements):
                                e_positions = positions[cnts[:e].sum(): cnts[:e+1].sum()]
                                sdfs[cnts[:e].sum(): cnts[:e+1].sum()] = self.kernel_caller(kernels[e:e+1], e_positions)[:, 0]
                            self.bisection_hypermesh_verts_iter(self.sdf_AF(sdfs), self.sdf_AF(center_sdfs))
                        with Timer("bisection group finishing", write_to_file=path/"log.txt"):
                            self.bisection_hypermesh_verts_output(self.AF(positions), 1)
                            sdfs = np.zeros(positions.shape[0], dtype=self.sdf_np_float_type)
                            for e in range(n_elements):
                                e_positions = positions[cnts[:e].sum(): cnts[:e+1].sum()]
                                sdfs[cnts[:e].sum(): cnts[:e+1].sum()] = self.kernel_caller(kernels[e:e+1], e_positions)[:, 0]
                            self.bisection_hypermesh_verts_finishing(t, self.sdf_AF(sdfs), self.sdf_AF(center_sdfs))
                    self.write_final_hypermesh(t)
                    (path / f"bisection/{t}.finish").touch()
            self.bisection_clean_up()
            # precompute data for slicing the 4D hypermesh by slicing at critical time points
            with Timer("slicing_preprocess"):
                files_to_delete = list(path.glob("processed_hyperpolys/*"))
                for file_path in files_to_delete:
                    file_path.unlink()
                self.slicing_preprocess()
            (path / f"slicing_preprocess.finish").touch()
            # clean up unnecessary files
            files_to_delete = list(path.glob("*/*")) + list(path.glob("*"))
            files_to_keep = list(path.glob("processed_hyperpolys/*")) + list(path.glob("hypervertices/*")) + \
                list(path.glob("slicing_preprocess.finish")) + list(path.glob("*.txt"))
            for file_path in files_to_delete:
                if file_path not in files_to_keep and not os.path.isdir(file_path):
                    file_path.unlink()
            # All the 4D mesh preparation ends here

        # The slicing process, can be called repeatedly once the 4D mesh is ready
        with Timer("slicing"):
            v_cnts = np.zeros(n_elements, dtype=np.int32)
            f_cnts = np.zeros(n_elements, dtype=np.int32)
            self.run_slicing(self.slicing_time, AsInt(v_cnts), AsInt(f_cnts), True)
            meshes = []
            in_view_tags = []
            for ele in range(n_elements):
                vertices = AC(np.zeros((v_cnts[ele], 3), dtype=self.float_type))
                inview_tag = AC(np.zeros(v_cnts[ele], dtype=np.int32))
                faces = AC(np.zeros((f_cnts[ele], 3), dtype=np.int32))
                self.slicing_output(ele, self.AF(vertices), AsInt(faces), AsInt(inview_tag))
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                meshes.append(mesh)
                in_view_tags.append(inview_tag.astype(bool))
            self.slicing_clean_up()

        return meshes, in_view_tags
