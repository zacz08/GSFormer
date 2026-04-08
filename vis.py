import os
import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid

from model.utils.safe_ops import safe_sigmoid

# --------------- colormaps ---------------

NUSC_SEM_COLORS = np.array([
    [  0,   0,   0],       # 0  others
    [255, 120,  50],       # 1  barrier              orange
    [255, 192, 203],       # 2  bicycle              pink
    [255, 255,   0],       # 3  bus                  yellow
    [  0, 150, 245],       # 4  car                  blue
    [  0, 255, 255],       # 5  construction_vehicle cyan
    [255, 127,   0],       # 6  motorcycle           dark orange
    [255,   0,   0],       # 7  pedestrian           red
    [255, 240, 150],       # 8  traffic_cone         light yellow
    [135,  60,   0],       # 9  trailer              brown
    [160,  32, 240],       # 10 truck                purple
    [255,   0, 255],       # 11 driveable_surface    dark pink
    [139, 137, 137],       # 12 other_flat           grey
    [ 75,   0,  75],       # 13 sidewalk             dark purple
    [150, 240,  80],       # 14 terrain              light green
    [230, 230, 250],       # 15 manmade              white
    [  0, 175,   0],       # 16 vegetation           green
], dtype=np.uint8)


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2
    return coords_grid


def _prepare_occ_voxels(gaussian, sem, cap, dataset):
    """Extract filtered voxel positions and labels from occupancy grid."""
    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
        vmin, vmax = 0, 16
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 19
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 18
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    voxels = gaussian[0].cpu().to(torch.int)
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1

    grid_coords = get_grid_coords(
        voxels.shape, voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    if not sem:
        fov_voxels = grid_coords[
            (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 100)
        ]
    else:
        if dataset == 'nusc':
            fov_voxels = grid_coords[
                (grid_coords[:, 3] >= 0) & (grid_coords[:, 3] < 17)
            ]
        elif dataset == 'kitti360':
            fov_voxels = grid_coords[
                (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 19)
            ]
        else:
            fov_voxels = grid_coords[
                (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 20)
            ]
    print(f"Number of occupied voxels: {len(fov_voxels)}")
    return fov_voxels, sem, vmin, vmax, dataset


def _get_voxel_colors(fov_voxels, sem, vmin, vmax, dataset):
    """Map voxel labels to RGB colors."""
    labels = fov_voxels[:, 3].astype(int)
    if sem:
        if dataset == 'nusc':
            cmap = NUSC_SEM_COLORS
            label_colors = np.zeros((len(labels), 3), dtype=np.uint8)
            for i, lb in enumerate(labels):
                lb = int(np.clip(lb, 0, len(cmap) - 1))
                label_colors[i] = cmap[lb]
            return label_colors
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            mapped = cm.jet(norm(labels))[:, :3]
            return (mapped * 255).astype(np.uint8)
    else:
        norm = colors.Normalize(vmin=fov_voxels[:, 3].min(), vmax=fov_voxels[:, 3].max())
        mapped = cm.jet(norm(fov_voxels[:, 3]))[:, :3]
        return (mapped * 255).astype(np.uint8)


def _save_ply(filepath, positions, rgb):
    """Write a colored point cloud to PLY format."""
    n = len(positions)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
    ])
    data = np.empty(n, dtype=dtype)
    data['x'] = positions[:, 0]
    data['y'] = positions[:, 1]
    data['z'] = positions[:, 2]
    data['r'] = rgb[:, 0]
    data['g'] = rgb[:, 1]
    data['b'] = rgb[:, 2]
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def save_occ(save_dir, gaussian, name, sem=False, cap=2, dataset='nusc'):
    """Save occupancy visualization as PLY point cloud + matplotlib PNG.

    The PLY file can be loaded interactively via ``view_in_viser()``
    or any 3D viewer (MeshLab, Open3D, etc.).
    A matplotlib scatter PNG is also saved for quick preview.
    """
    fov_voxels, sem, vmin, vmax, dataset = _prepare_occ_voxels(
        gaussian, sem, cap, dataset)
    rgb = _get_voxel_colors(fov_voxels, sem, vmin, vmax, dataset)

    positions = fov_voxels[:, :3].copy()
    positions[:, 1] = -positions[:, 1]  # flip y for consistent view

    # --- Save PLY file (loadable in viser / MeshLab / Open3D) ---
    ply_path = os.path.join(save_dir, f'{name}.ply')
    _save_ply(ply_path, positions, rgb)
    print(f"Saved PLY: {ply_path}")

    # --- Save matplotlib PNG for quick preview ---
    fig = plt.figure(figsize=(12, 12), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=40, azim=-135)

    n = len(positions)
    if n > 200000:
        idx = np.random.choice(n, 200000, replace=False)
        pos_sub = positions[idx]
        rgb_sub = rgb[idx]
    else:
        pos_sub = positions
        rgb_sub = rgb

    ax.scatter(
        pos_sub[:, 0], pos_sub[:, 1], pos_sub[:, 2],
        c=rgb_sub.astype(np.float32) / 255.0,
        s=0.1, marker='s', depthshade=True)

    ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
    ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
    ax.set_zlim(positions[:, 2].min(), positions[:, 2].max())
    ax.grid(False)
    ax.set_axis_off()

    png_path = os.path.join(save_dir, f'{name}.png')
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved PNG: {png_path}")


def get_nuscenes_colormap():
    return NUSC_SEM_COLORS.astype(np.float32) / 255.


def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False):

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

    means = gaussian.means[0].detach().cpu().numpy()
    scales = gaussian.scales[0].detach().cpu().numpy()
    rotations = gaussian.rotations[0].detach().cpu().numpy()
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy()
    sems = gaussian.semantics[0].detach().cpu().numpy()
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.75)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk

        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask

    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    ellipNumber = means.shape[0]

    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=46, azim=-180)

    border = np.array([
        [-50.0, -50.0, 0.0],
        [-50.0, 50.0, 0.0],
        [50.0, -50.0, 0.0],
        [50.0, 50.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:],
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in range(ellipNumber):

        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1)
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2],
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("equal")
    ax.grid(False)
    ax.set_axis_off()

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath)

    plt.cla()
    plt.clf()


def save_gaussian_topdown(save_dir, anchor_init, gaussian, name):
    init_means = safe_sigmoid(anchor_init[:, :2]) * 100 - 50
    means = [init_means] + [g.means[0, :, :2] for g in gaussian]

    plt.clf(); plt.cla()
    fig = plt.figure(figsize=(24., 16.))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, 5),
                    axes_pad=0.,
                    share_all=True
                    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    for ax, im in zip(grid, means):
        im = im.cpu()
        ax.scatter(im[:, 0], im[:, 1], s=0.1, marker='o')
    plt.savefig(os.path.join(save_dir, f"{name}.jpg"))
    plt.clf(); plt.cla()


# --------------- Interactive viser viewer (optional) ---------------

def view_in_viser(ply_path, port=8080):
    """Launch an interactive viser web viewer for a saved PLY point cloud.

    Usage:
        python -c "from vis import view_in_viser; view_in_viser('out/nuscenes_gs128/vis_ep0/val_0_pred.ply')"

    Then open http://localhost:8080 in a browser.
    """
    import viser
    import trimesh

    cloud = trimesh.load(ply_path)
    positions = np.asarray(cloud.vertices).astype(np.float32)
    rgb = np.asarray(cloud.colors)[:, :3].astype(np.uint8)

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.add_point_cloud(
        "/occ",
        points=positions,
        colors=rgb,
        point_size=0.3,
        point_shape="rounded",
    )
    print(f"Viser viewer running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
