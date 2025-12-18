from pathlib import Path

import torch
import tqdm
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image

INTERACTIVE = False

# im = nib.load(r"D:\logs\20251117-162436-foundation-cluster\logs\train_volumes\epoch_60000\1011525\pred\full.nii.gz")  # Visible papillary
# im_path = r"D:\logs\20251125-034718-psf20k_100subj-20ann\pred_volumes\epoch_24000\1011525\pred\full.nii.gz"  # Visible papillary
im_path = Path("/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251216-163025-inference_trainSet_FT_ImageVisualization/logs/test_opt2500_ft4000_volumes/epoch_0/1012959/pred/full.nii.gz")
im_path = Path("/home/nil/Documents/git/CMR_intensity_alignment/trained_models/20251217-035047-inference_trainSet_FT_ImageVisualization/logs/test_opt5000_ft0250_volumes/epoch_0/1037287/pred/full.nii.gz")
# im_path = Path(r"D:\logs\20251203-104135-inference_no_seg_lowLR\logs\test_ft5000_volumes\epoch_0\1384486\pred\full.nii.gz")
im = nib.load(str(im_path))  # Somewhat visible papillary
im = im.dataobj[...].astype(np.uint8)
a = im[..., 105, 0]
pad_start = (50, 50, 60)
pad_end = (-50, -50, -80)
down_scale = 1
im_crop = im[pad_start[0]:pad_end[0]:down_scale,
          pad_start[1]:pad_end[1]:down_scale,
          pad_start[2]:pad_end[2]:down_scale]

seg_path = im_path.parent / 'full_seg.nii.gz'
seg = nib.load(str(seg_path))  # Somewhat visible papillary
seg = seg.dataobj[...].copy()
seg_crop = seg[pad_start[0]:pad_end[0]:down_scale,
           pad_start[1]:pad_end[1]:down_scale,
           pad_start[2]:pad_end[2]:down_scale]
H, W, D, T = im_crop.shape
# im_crop = np.where(seg_crop>0, im_crop*1.1, im_crop)

# Dark sections (myocardium with empty blood pools + surrounds)
LOWER_THRESH = 20
HIGHER_THRESH = 115
INTENS_SCALING = 1.0
def get_opacity_map(low, high, max_opacity = 1.0):
    if low >= high:
        low = high - 1
    if high <= low:
        high = low + 1
    cut_index_low = max(0, min(254, int(low * INTENS_SCALING)))
    cut_index_high = max(1, min(255, int(high * INTENS_SCALING)))
    opacity_map = np.zeros((256,))
    opacity_map[cut_index_low: cut_index_high] = 1.0
    # Soft thresholding
    fine_range = cut_index_high-cut_index_low
    fine1 = np.sqrt(np.linspace(0,1, fine_range//4)) * max_opacity
    fine2 = np.sqrt(np.linspace(1,0, fine_range//4)) * max_opacity
    opacity_map[cut_index_low: cut_index_low+fine1.shape[0]] = fine1.clip(0,1.0)
    opacity_map[cut_index_high-fine2.shape[0]: cut_index_high] = fine2.clip(0,1.0)
    return opacity_map.astype(float)

def get_scaled_vol(frame_idx, scaling):
    im_crop_scale = np.clip(im_crop[..., frame_idx].astype(int) * scaling, 0, 255).astype(np.uint8)
    return im_crop_scale

def update_opacity_map():
    # Rebuild opacity transfer function
    opacity_func = actor.GetProperty().GetScalarOpacity()
    opacity_func.RemoveAllPoints()
    opacity_map = get_opacity_map(LOWER_THRESH, HIGHER_THRESH, INTENS_SCALING)
    for i, alpha in enumerate(opacity_map):
        opacity_func.AddPoint(i, alpha)
    pl.render_window.Render()

pl = pv.Plotter(off_screen=(not INTERACTIVE))
CURRENT_FRAME = 0
img = pv.ImageData(dimensions=(H,W,D))
img.point_data["values"] = get_scaled_vol(CURRENT_FRAME, INTENS_SCALING).ravel(order="F")
actor = pl.add_volume(img, cmap="gray", opacity=get_opacity_map(LOWER_THRESH, HIGHER_THRESH), show_scalar_bar=False)
actor.prop.interpolation_type = 'linear'
update_opacity_map()

# Add clipping plane
CLIP_PLANE_NORMAL = (0.2, 1.0, -0.15)
WIDTH_OFFSET = -40
widget = pl.add_volume_clip_plane(
    actor,
    normal=CLIP_PLANE_NORMAL,
    origin=(H//2, W//2+WIDTH_OFFSET, 0),
    invert=False,
)
widget.GetOutlineProperty().SetOpacity(0)

if INTERACTIVE:
    def update_lower_threshold(value):
        global LOWER_THRESH
        LOWER_THRESH = value
        update_opacity_map()

    def update_upper_threshold(value):
        global HIGHER_THRESH
        HIGHER_THRESH = value
        update_opacity_map()

    def update_scaling(value):
        global INTENS_SCALING
        INTENS_SCALING = value
        img.point_data["values"][:] = get_scaled_vol(CURRENT_FRAME, INTENS_SCALING).ravel(order="F")
        update_opacity_map()
        img.GetPointData().Modified()  # Mark the point data as modified
        pl.render_window.Render()

    def update_opacity_map():
        # Rebuild opacity transfer function
        opacity_func = actor.GetProperty().GetScalarOpacity()
        opacity_func.RemoveAllPoints()
        opacity_map = get_opacity_map(LOWER_THRESH, HIGHER_THRESH, INTENS_SCALING)
        for i, alpha in enumerate(opacity_map):
            opacity_func.AddPoint(i, alpha)
        pl.render_window.Render()
    update_opacity_map()

    slider1 = pl.add_slider_widget(
        callback=update_lower_threshold,
        rng=[0, 255],
        value=LOWER_THRESH,
        title="Lower Threshold",
        pointa=(0.9, 0.0),
        pointb=(0.9, 0.2),
        style='modern'
    )
    slider2 = pl.add_slider_widget(
        callback=update_upper_threshold,
        rng=[0, 255],
        value=HIGHER_THRESH,
        title="Upper Threshold",
        pointa=(0.9, 0.3),
        pointb=(0.9, 0.5),
        style='modern'
    )

    slider3 = pl.add_slider_widget(
        callback=update_scaling,
        rng=[0.0, 3.0],
        value=INTENS_SCALING,
        title="Intensity scaling",
        pointa=(0.9, 0.6),
        pointb=(0.9, 0.8),
        style='modern'
    )

    def update(obj, event):
        global CURRENT_FRAME
        # print(CURRENT_FRAME)
        CURRENT_FRAME = (CURRENT_FRAME + 1) % T
        img.point_data["values"][:] = get_scaled_vol(CURRENT_FRAME, INTENS_SCALING).ravel(order="F")
        img.GetPointData().Modified()  # Mark the point data as modified
        pl.render_window.Render()

    pl.iren.initialize()
    pl.iren.add_observer('TimerEvent', update)
    loop_duration = 3.0  # seconds
    frame_time = loop_duration / T
    pl.iren.create_timer(int(frame_time * 1000))  # ms intervals, repeats indefinitely

    pl.iren.start()
else:
    # Calculate center of volume for camera orbit
    center = np.array([H / 2, W / 2, D / 2])

    def generate_circular_orbit(n_points, radius=300, plane_normal=(0., 1., 0.)):
        """Generate a circular camera path that orbits around the plane_normal direction"""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Normalize plane normal
        plane_normal = np.array(plane_normal, dtype=float)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Find two orthogonal vectors perpendicular to plane_normal
        # These define the orbit plane
        if abs(plane_normal[2]) < 0.9:
            arbitrary = np.array([0, 0, 1])
        else:
            arbitrary = np.array([1, 0, 0])

        u = np.cross(plane_normal, arbitrary)
        u = u / np.linalg.norm(u)
        v = np.cross(plane_normal, u)
        v = v / np.linalg.norm(v)

        # Create circular orbit in the plane perpendicular to plane_normal
        # Start from the NEGATIVE plane_normal side (flip the direction)
        positions = np.zeros((n_points, 3))
        for i, angle in enumerate(t):
            # Circular motion in the uv-plane, offset along plane_normal
            circle_pos = np.cos(angle) * u + np.sin(angle) * v
            # Position camera at radius distance along NEGATIVE plane_normal, then offset in circle
            positions[i] = center - plane_normal * radius + circle_pos * (radius * 0.3)

        return positions

    # Video settings
    fps = 30
    duration_seconds = 10  # Total video duration
    n_frames = fps * duration_seconds
    save_dir = Path('vol_renders')
    save_dir.mkdir(exist_ok=True, parents=True)
    video_file = save_dir / f'{im_path.parent.parent.parent.parent.parent.parent.name}_{im_path.parent.parent.parent.parent.name}_{im_path.parent.parent.name}_highThresh{HIGHER_THRESH}_planeNorm{CLIP_PLANE_NORMAL}_offset{WIDTH_OFFSET}_cardiac_volume.mp4'

    # Generate camera path
    # Circular orbit around the clipping plane normal
    volume_size = np.sqrt(H ** 2 + W ** 2 + D ** 2)  # Diagonal distance
    camera_positions = generate_circular_orbit(n_frames, radius=volume_size * 1.2, plane_normal=CLIP_PLANE_NORMAL)
    # Setup video writer
    pl.open_movie(video_file, framerate=fps, quality=9)

    for i in tqdm.tqdm(range(n_frames), desc=f"Rendering {n_frames} frames..."):
        # Update time frame (cycle through T frames)
        time_frame = int((i*2 / n_frames) * T) % T

        # Update volume data
        img.point_data["values"][:] = get_scaled_vol(time_frame, INTENS_SCALING).ravel(order="F")
        img.GetPointData().Modified()

        # Update camera position
        camera_pos = camera_positions[i]
        pl.camera.position = camera_pos
        pl.camera.focal_point = center

        # Keep camera upright
        pl.camera.up = (0, 0, 1)

        # Write frame
        pl.write_frame()

    pl.close()
    print(f"Video saved to: {video_file}")

