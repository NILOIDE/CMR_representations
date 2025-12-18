import time
from pathlib import Path

import imageio
import plotly.graph_objects as go
import tqdm
from skimage.measure import marching_cubes
import nibabel as nib
import trimesh
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
import numpy as np
import pyvista as pv
from PIL import Image


# seg_path = r"D:\logs\20251203-104135-inference_no_seg_lowLR\logs\test_ft5000_volumes\epoch_0\1384486\pred\full_seg.nii.gz"
# seg_path = r"D:\logs\20251125-034718-psf20k_100subj-20ann\pred_volumes\epoch_24000\1021869\pred\full_seg.nii.gz"
# seg_path = r"/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251215-002046-inference_trainSet_visualization/logs/train_volumes/epoch_0/1000456/pred/full_seg.nii.gz"
model_path = r"/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251215-032410-inference_trainSet_visualization/logs/train_volumes/epoch_0"
model_path = Path(model_path)
for subj_path in list(model_path.iterdir())[4:5]:
    seg_path = subj_path / 'pred' / 'full_seg.nii.gz'
    seg = nib.load(str(seg_path))
    seg = seg.dataobj[:].copy()
    pad_start = (15, 15, 15)
    pad_end = (-15, -15, -15)
    down_scale = 1
    seg_crop = seg[pad_start[0]:pad_end[0]:down_scale,
               pad_start[1]:pad_end[1]:down_scale,
               pad_start[2]:pad_end[2]:down_scale]
    # seg_crop = seg_crop[...,::5]
    H, W, D, T = seg_crop.shape

    SHOW_RENDER = False
    CREATE_GIF = True
    REPLACE_EXISTING = False
    renders = []
    render_save_dir = seg_path.parent.parent.parent.parent.parent / 'renders' / model_path.name / seg_path.parent.parent.name
    render_save_dir.mkdir(exist_ok=True, parents=True)

    for t_prop in tqdm.tqdm(np.arange(0, T, 1)/T, desc=f'Generating seg mesh frames for subj {subj_path.name} at {render_save_dir}'):
        t=int(T*t_prop)
        save_path = render_save_dir / 'renders' / f"render{t}.png"
        renders.append(save_path)
        if not REPLACE_EXISTING and save_path.exists():
            continue
        save_path.parent.mkdir(exist_ok=True, parents=True)
        seg_masked = seg_crop[...,t]
        size = np.array(seg_masked.shape)

        # Clean up surroundings by eroding and dilating
        struct = generate_binary_structure(3, 3)
        mask = seg_masked > 0
        seg_masked = np.where(mask, seg_masked, 0)

        class_meshes = []
        labels = [3, 2, 1]
        for label in labels:
            if not np.any(seg_masked==label):
                continue
            mask = seg_masked == label
            # if label > 1:
            #     mask = binary_erosion(mask, structure=struct, iterations=1)
            verts, faces, normals, _ = marching_cubes(mask, level=0.5)
            # ------------------------------------------------------------------------------------
            if label == 3:
                verts += np.array((1, 0, 0))
            verts = verts / size * 2 - 1
            verts *= np.array((1, 1.3, 1))
            if r"1011525" in str(seg_path):
                if label == 1:
                    verts *= np.array((0.95, 0.98, 0.98))
                    verts += np.array((-0.009, -0.01, 0))
                # if int(T*0.3) <= t < int(T*0.5):
                #     if label == 1:
                #         verts *= np.array((0.99, 0.99, 0.99))
            elif "1021869" in str(seg_path):
                if label == 1:
                    verts *= np.array((0.96, 0.98, 0.98))
                    verts += np.array((-0.005, -0.01, 0))
                if int(T*0.47) <= t < int(T*0.54):
                    if label == 1:
                        verts *= np.array((0.99, 0.99, 0.99))
            else:
                if label == 1:
                    verts *= np.array((0.95, 0.98, 0.98))
                    verts += np.array((-0.0055, -0.01, 0))

            class_meshes.append((verts, faces, normals))

        # color palette per class
        colors = {
            1: "rgb(150,0,0)",
            2: "rgb(0,150,0)",
            3: "rgb(200, 200, 0)",
        }
        label_names = {
            1: "LV BloodPool",
            2: "LV Myocardium",
            3: "RV BloodPool",
        }
        light_pos = {'x': 10.,
                     'y': 10.,
                     'z': 0.}
        lighting = dict(
            ambient=0.2,     # less fill light
            diffuse=1.0,     # strong directional shading
            specular=0.5,    # highlight reflections
            roughness=0.5,   # slightly glossy
            fresnel=0.1
        )
        mesh_obj = []

        for label, (v, f, n) in zip(labels, class_meshes):
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            original_vertices = mesh.vertices.copy()
            mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=15)
            displacement = np.linalg.norm(mesh.vertices - original_vertices, axis=1)
            problematic = displacement > 0.03
            mesh.vertices[problematic] = original_vertices[problematic] + (mesh.vertices - original_vertices)[problematic]*0.2
            # mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.5, iterations=10)
            v, f = mesh.vertices, mesh.faces
            f = np.hstack([np.full((f.shape[0], 1), 3), f]).ravel()
            clipped = pv.PolyData(v, f,)# clip the mesh
            if label == 1:
                clipped = clipped.clip(
                    normal=(0, 1, 0),     # diagonal plane normal
                    origin=(0,-0.2,0),      # a point on the plane
                    invert=False           # flip if you want the other half
                )
            else:
                clipped = clipped.clip(
                    normal=(0, 1, 0),     # diagonal plane normal
                    origin=(0,-0.2,0),      # a point on the plane
                    invert=False           # flip if you want the other half
                )
            v = clipped.points
            f = clipped.faces.reshape(-1, 4)[:, 1:]  # remove leading 3
            m = go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=colors[label],
                opacity=1.0,  # <--- fully opaque
                # flatshading=True,
                lighting=lighting,
                name=label_names[label],
                lightposition=light_pos,
            )
            mesh_obj.append(m)

        fig = go.Figure()
        for obj in mesh_obj:
            fig.add_trace(obj)
        zoom = 2.0
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[-1, 1]),
                yaxis=dict(visible=False, range=[-1, 1]),
                zaxis=dict(visible=False, range=[-1, 1]),
                aspectmode='cube',
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=-0.7/zoom, y=-2.5/zoom, z=1.5/zoom),  # Camera position (viewing from back)
                    center=dict(x=0, y=0, z=0),  # Look-at point
                    up=dict(x=0, y=0, z=1)  # Up direction
                ),
            ),
            dragmode='orbit',
            margin=dict(l=0, r=0, t=0, b=0),
            scene_aspectmode='data',
        )
        if SHOW_RENDER:
            fig.show()
        res = (600,600)
        fig.write_image(str(save_path), width=res[0], height=res[1], scale=1)  # requires kaleido
        time.sleep(2)
        img = Image.open(str(save_path))
        # # Define crop box: (left, top, right, bottom)
        crop_box = (0.1, 0.1, 0.1, 0.1)
        crop_box = (int(img.size[0] * crop_box[0]), int(img.size[1] * crop_box[1]),
                    int(img.size[0] - (img.size[0] * crop_box[2])), int(img.size[1] - (img.size[1] * crop_box[3])))
        cropped = img.crop(crop_box)
        cropped.save(str(save_path))

    if CREATE_GIF:
        # Video parameters
        total_duration = 2.0  # seconds
        fps = len(renders) / total_duration
        output_path = render_save_dir / f"{subj_path.name}.mp4"
        with imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=8,  # 0–10, higher = better
                pixelformat="yuv420p",  # required for broad browser compatibility
        ) as writer:
            for frame_path in renders:
                img = Image.open(str(frame_path))
                # Convert PIL Image -> numpy array (RGB)
                writer.append_data(np.asarray(img))
