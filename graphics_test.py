import time

import numpy as np
import plotly.graph_objects as go
# import pymeshlab
from skimage.measure import marching_cubes
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
import nibabel as nib
import trimesh
from trimesh.smoothing import filter_taubin
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
import numpy as np
import pyvista as pv

# -----------------------------------------
# 1. Create voxel segmentation of a sphere
# -----------------------------------------
# im = nib.load(r"D:\logs\20251114-181651\1135402\pred\full.nii.gz")
# pad_start = (40, 90, 40)
# pad_end = (-30, -30, -40)
im = nib.load("/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251120-233109-long_train/logs/train_volumes/epoch_7400/1011525/pred/full.nii.gz")  # Visible papillary
# im = nib.load(r"D:\logs\20251117-162436-foundation-cluster\logs\train_volumes\epoch_50000\1011525\full.nii.gz")  # Somewhat visible papillary
pad_start = (40, 30, 40)
pad_end = (-30, -30, -40)
# im = nib.load(r"D:\logs\20251027-151026no_point_spread_no_conv\logs\train_volumes\1347746\pred\full.nii.gz")
# pad_start = (20, 20, 20)
# pad_end = (-100, -20, -20)

im = im.dataobj[...].astype(np.uint8)
seg = nib.load(r"/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251120-233109-long_train/logs/train_volumes/epoch_7400/1011525/pred/full_seg.nii.gz")
seg = seg.dataobj[...].copy()
a = im[...,105, 0]

down_scale = 1
im_crop = im[pad_start[0]:pad_end[0]:down_scale,
          pad_start[1]:pad_end[1]:down_scale,
          pad_start[2]:pad_end[2]:down_scale]
seg_crop = seg[pad_start[0]:pad_end[0]:down_scale,
           pad_start[1]:pad_end[1]:down_scale,
           pad_start[2]:pad_end[2]:down_scale]
H, W, D, T = im_crop.shape


# Bright sections (blood pools + surrounds)
# im_crop_scale = np.clip((im_crop.astype(float)/255)**0.5 * 255/16, 0, 255).astype(np.uint8)
# img = pv.ImageData(dimensions=im_crop.shape)
# img.point_data["values"] = im_crop_scale.ravel(order="F")
# cut_index_high = 200
# cut_index_low = 100
# opacity = 100
# opacity_map = np.zeros((256,))
# cut_index_mid = 130
# fine1 = np.linspace(0,1, cut_index_mid-cut_index_low) * opacity
# fine2 = np.linspace(1,0, cut_index_high-cut_index_mid) * opacity
# opacity_map[cut_index_low: cut_index_low+fine1.shape[0]] = fine1.clip(0,255)
# opacity_map[cut_index_low+fine1.shape[0]: cut_index_low+fine1.shape[0]+fine2.shape[0]] = fine2.clip(0,255)

# Mask out everything where y > x
# x, y, z = np.mgrid[0:im_crop_scale.shape[0], 0:im_crop_scale.shape[1], 0:im_crop_scale.shape[2]]
# im_crop_scale[x < y] = 0   # or np.nan if you prefer removing it completely


# Dark sections (myocardium with empty blood pools + surrounds)
LOWER_THRESH = 20
HIGHER_THRESH = 90
INTENS_SCALING = 1
def get_opacity_map(low, high, max_opacity = 1.0):
    if low >= high:
        low = high - 1
    if high <= low:
        high = low + 1
    cut_index_low = max(0, min(254, int(low * INTENS_SCALING)))
    cut_index_high = max(1, min(255, int(high * INTENS_SCALING)))
    opacity_map = np.zeros((256,))
    fine_range = cut_index_high-cut_index_low
    fine1 = np.sqrt(np.linspace(0,1, fine_range//2)) * max_opacity
    fine2 = np.sqrt(np.linspace(1,0, fine_range//2)) * max_opacity
    opacity_map[cut_index_low: cut_index_low+fine1.shape[0]] = fine1.clip(0,1.0)
    opacity_map[cut_index_low+fine1.shape[0]: cut_index_low+fine1.shape[0]+fine2.shape[0]] = fine2.clip(0,1.0)
    return opacity_map.astype(float)

def get_scaled_vol(scaling):
    im_crop_scale = np.clip(im_crop[..., CURRENT_FRAME].astype(int) * scaling, 0, 255).astype(np.uint8)
    return im_crop_scale

# pl = pv.Plotter()
# CURRENT_FRAME = 0
# img = pv.ImageData(dimensions=(H,W,D))
# img.point_data["values"] = get_scaled_vol(INTENS_SCALING).ravel(order="F")
# actor = pl.add_volume(img, cmap="gray", opacity=get_opacity_map(LOWER_THRESH, HIGHER_THRESH))
# actor.prop.interpolation_type = 'linear'
#
#
# def update_lower_threshold(value):
#     global LOWER_THRESH
#     LOWER_THRESH = value
#     update_opacity_map()
#
# def update_upper_threshold(value):
#     global HIGHER_THRESH
#     HIGHER_THRESH = value
#     update_opacity_map()
#
# def update_scaling(value):
#     global INTENS_SCALING
#     INTENS_SCALING = value
#     img.point_data["values"][:] = get_scaled_vol(INTENS_SCALING).ravel(order="F")
#     update_opacity_map()
#     img.GetPointData().Modified()  # Mark the point data as modified
#     pl.render_window.Render()
#
# def update_opacity_map():
#     # Rebuild opacity transfer function
#     opacity_func = actor.GetProperty().GetScalarOpacity()
#     opacity_func.RemoveAllPoints()
#     opacity_map = get_opacity_map(LOWER_THRESH, HIGHER_THRESH, INTENS_SCALING)
#     for i, alpha in enumerate(opacity_map):
#         opacity_func.AddPoint(i, alpha)
#     pl.render_window.Render()
# update_opacity_map()
#
# slider1 = pl.add_slider_widget(
#     callback=update_lower_threshold,
#     rng=[0, 255],
#     value=LOWER_THRESH,
#     title="Lower Threshold",
#     pointa=(0.9, 0.0),
#     pointb=(0.9, 0.2),
#     style='modern'
# )
# slider2 = pl.add_slider_widget(
#     callback=update_upper_threshold,
#     rng=[0, 255],
#     value=HIGHER_THRESH,
#     title="Upper Threshold",
#     pointa=(0.9, 0.3),
#     pointb=(0.9, 0.5),
#     style='modern'
# )
#
# slider3 = pl.add_slider_widget(
#     callback=update_scaling,
#     rng=[0.0, 3.0],
#     value=INTENS_SCALING,
#     title="Intensity scaling",
#     pointa=(0.9, 0.6),
#     pointb=(0.9, 0.8),
#     style='modern'
# )
#
#
# widget = pl.add_volume_clip_plane(
#     actor,                # or img instead of actor if you prefer
#     normal=(1, -1, 0),    # plane normal → y = x
#     origin=(-11, 13, 0),     # passes through (0,0,0); adjust if needed
#     invert=False,         # flip to True if it removes the wrong side
# )
# widget.GetOutlineProperty().SetOpacity(0)
#
#
# def update(obj, event):
#     global CURRENT_FRAME
#     # print(CURRENT_FRAME)
#     CURRENT_FRAME = (CURRENT_FRAME + 1) % T
#     img.point_data["values"][:] = get_scaled_vol(INTENS_SCALING).ravel(order="F")
#     img.GetPointData().Modified()  # Mark the point data as modified
#     pl.render_window.Render()
#
# pl.iren.initialize()
# pl.iren.add_observer('TimerEvent', update)
# loop_duration = 2.0  # seconds
# frame_time = loop_duration / T
# pl.iren.create_timer(int(frame_time * 1000))  # ms intervals, repeats indefinitely
#
# # for label, color in [
# #     (1, "darkred"),
# #     (2, "darkgreen"),
# #     (3, "darkblue")
# # ]:
# #     mask = (seg_crop == label)
# #     if not np.any(mask):
# #         continue
# #     struct = generate_binary_structure(3,3)
# #     mask = np.logical_and(mask, ~binary_erosion(mask, structure=struct, iterations=1))
# #     # if label == 2:
# #     #     mask[:-10] = 0
# #     mask[-3:] = 0
# #     mask = mask.astype(float)
# #
# #     img_mask = pv.ImageData(dimensions=mask.shape)
# #     img_mask.point_data["m"] = mask.ravel(order="F")
# #
# #     cmap = np.tile(np.array(color), (256, 1))
# #     actor = pl.add_volume(
# #         img_mask,
# #         scalars="m",
# #         opacity=[0, 0.03],         # 0 = no mask, 1 = label
# #         cmap=[color],
# #     )
# #     actor.prop.interpolation_type = 'linear'
#     # pl.add_volume_clip_plane(actor, normal=(1, -1, 0), origin=(0, 0, 0), invert=False)
#
# pl.iren.start()
# quit()

# -----------------------------------------
# 2. Run marching cubes
# -----------------------------------------
t=T//2
seg_masked = seg_crop[...,t]
seg_masked_ED = seg_crop[...,0]
size = np.array(seg_masked.shape)

# Clean up surroundings by eroding and dilating
struct = generate_binary_structure(3, 3)
mask = seg_masked > 0
mask = binary_erosion(mask, structure=struct, iterations=15)
mask = binary_dilation(mask, structure=struct, iterations=18)
seg_masked = np.where(mask, seg_masked, 0)
mask = seg_masked_ED > 0
mask = binary_erosion(mask, structure=struct, iterations=15)
mask = binary_dilation(mask, structure=struct, iterations=18)
seg_masked_ED = np.where(mask, seg_masked_ED, 0)

class_meshes = []
labels = [3, 2, 1]
for label in labels:
    if not np.any(seg_masked==label):
        continue
    verts, faces, normals, _ = marching_cubes(seg_masked == label, level=0.5)
    verts = verts / size * 2 - 1
    if label == 2:
        mask = binary_dilation(seg_masked == 1, structure=struct, iterations=2)
        verts1, faces1, normals1, _ = marching_cubes(seg_masked == label, level=0.5)
        mesh2 = trimesh.Trimesh(verts, faces)
        mesh1 = trimesh.Trimesh(verts1, faces1)
        inside = mesh2.contains(mesh1.vertices)
        # Keep faces only if *all three* vertices are outside B
        faces_to_keep = np.all(~inside[mesh2.faces], axis=1)
        clean_faces = mesh2.faces[faces_to_keep]
        clean_vertices = mesh2.vertices  # unused vertices will auto-prune later
        mesh2 = trimesh.Trimesh(
            vertices=clean_vertices,
            faces=clean_faces,
            process=True  # this removes orphan vertices
        )
        verts, faces, normals = mesh2.vertices, mesh2.faces, mesh2.face_normals
    class_meshes.append((verts, faces, normals))

# verts are in index units, rescale back to [-1,1]

# -----------------------------------------
# 3. Convert to trimesh + smoothing
# -----------------------------------------
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
             'y': 0.,
             'z': 0.}
lighting = dict(
    ambient=0.2,     # less fill light
    diffuse=1.0,     # strong directional shading
    specular=0.5,    # highlight reflections
    roughness=0.9,   # slightly glossy
    fresnel=0.1
)
mesh_obj = []
for label, (v, f, n) in zip(labels, class_meshes):

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh = filter_taubin(mesh, lamb=0.5, nu=-0.5, iterations=5)
    v, f = mesh.vertices, mesh.faces
    f = np.hstack([np.full((f.shape[0], 1), 3), f]).ravel()
    mesh = pv.PolyData(v, f,)# clip the mesh
    clipped = mesh.clip(
        normal=(0, 1, 0),     # diagonal plane normal
        origin=(0,0,0),      # a point on the plane
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


# -----------------------------------------
# 4. Estimate deformations (smooth)
# -----------------------------------------
ds_factor = 1000
# choose random mesh points
verts, faces, normals, _ = marching_cubes(seg_masked_ED > 0, level=0.5)
verts = verts / size * 2 - 1
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh = filter_taubin(mesh, lamb=0.5, nu=-0.5, iterations=5)
v, f = mesh.vertices, mesh.faces
f = np.hstack([np.full((f.shape[0], 1), 3), f]).ravel()
mesh = pv.PolyData(v, f,)# clip the mesh
clipped = mesh.clip(
        normal=(0, 1, 0),     # diagonal plane normal
        origin=(0,0,0),      # a point on the plane
    invert=False           # flip if you want the other half
)
verts = clipped.points
faces = clipped.faces.reshape(-1, 4)[:, 1:]  # remove leading 3
verts = (verts + 1) / 2 * size

start_points = verts[::ds_factor]
num_lines = len(start_points)

d = nib.load(r"/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251120-233109-long_train/logs/train_volumes/epoch_7400/1011525/pred/full_def.nii.gz")
d_crop = d.dataobj[pad_start[0]:pad_end[0]:down_scale,
            pad_start[1]:pad_end[1]:down_scale,
            pad_start[2]:pad_end[2]:down_scale]

# deformation array (N, T, 3)
deform_lines = np.zeros((num_lines, T, 3))
# deform_lines = deform_lines + start_points[:,None] / size

line_obj = []
cone_plot_rate = 10
cone_plot_offset = 10
cone_size = 0.05
cone_obj = []
for i in range(num_lines):
    p0 = start_points[i]
    p0_norm = p0 / size * 2 - 1

    # Fake deformation: outward along normal + small noise wiggle
    for t in range(T):
        x, y, z = p0.round().astype(int)
        deform_lines[i, t] = p0_norm + d_crop[x, y, z, :, t] *2  # Origin coord system is from [0,1]

    obj = go.Scatter3d(
            x=deform_lines[i, :, 0],
            y=deform_lines[i, :, 1],
            z=deform_lines[i, :, 2],
            mode="lines",
            line=dict(width=4, color="rgb(0,0,150)"),
            showlegend=False
        )
    line_obj.append(obj)
    cone_pos = deform_lines[i, cone_plot_offset::cone_plot_rate]
    cone_direction = cone_pos - deform_lines[i, cone_plot_offset - 1:-1:cone_plot_rate]
    cone_direction_n = cone_direction / np.linalg.norm(cone_direction, axis=-1)[:, None]
    cone = go.Cone(
        x=cone_pos[:, 0],
        y=cone_pos[:, 1],
        z=cone_pos[:, 2],
        u=cone_direction_n[:, 0],
        v=cone_direction_n[:, 1],
        w=cone_direction_n[:, 2],
        colorscale=[[0, "pink"], [1, "pink"]],
        showscale=False,
        sizemode="absolute",
        sizeref=cone_size,  # size of arrowhead
        anchor="tail"  # tail is at (x,y,z)
    )
    cone_obj.append(cone)

# -----------------------------------------
# 6. Plotly mesh + lines
# -----------------------------------------
fig = go.Figure()
for obj in mesh_obj:
    fig.add_trace(obj)

for line in line_obj:
    fig.add_trace(line)

# for c in cone_obj:
#     fig.add_trace(c)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False, range=[-1, 1]),
        yaxis=dict(visible=False, range=[-1, 1]),
        zaxis=dict(visible=False, range=[-1, 1]),
        aspectmode='cube',
        bgcolor='rgba(0,0,0,0)',
    ),
    dragmode='orbit',
    margin=dict(l=0, r=0, t=0, b=0),
    scene_aspectmode='data',
)

fig.show()
