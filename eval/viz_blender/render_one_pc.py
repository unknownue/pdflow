import os
import sys
import numpy as np
import bpy
import bmesh
import math
sys.path.append(os.getcwd()) # for some reason the working directory is not in path

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

point_radius = 0.007
result_type = 'orig'

candidate_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

candidate_colors = np.array([[float(c)/255.0 for c in hex2rgb(color)] for color in candidate_colors])

pts_filename = sys.argv[6]
output_filename = sys.argv[7]

rotate_x = math.radians(int(sys.argv[8]))
rotate_y = math.radians(int(sys.argv[9]))
rotate_z = math.radians(int(sys.argv[10]))

ipts = np.loadtxt(pts_filename, dtype=np.float32)
pts = ipts[:, :3]  # xyz

# Change coordinate from [-1, 1] to [-2, 2]
pts = pts * 2.0

if ipts.shape[1] > 3:
    ptc = ipts[:, 3:]
else:
    ptc = None

# rotate for blender
# coord_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
# pts = np.matmul(pts, coord_rot.transpose())

# coord_rot = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
# ])
coord_rot_x = np.array([
    [1, 0, 0],
    [0, math.cos(rotate_x), -math.sin(rotate_x)],
    [0, math.sin(rotate_x),  math.cos(rotate_x)],
])
coord_rot_y = np.array([
    [ math.cos(rotate_y), 0, math.sin(rotate_y)],
    [0, 1, 0],
    [-math.sin(rotate_y), 0, math.cos(rotate_y)],
])
coord_rot_z = np.array([
    [math.cos(rotate_z), -math.sin(rotate_z), 0],
    [math.sin(rotate_z),  math.cos(rotate_z), 0],
    [0, 0, 1],
])
coord_rot_total = np.matmul(coord_rot_z, coord_rot_y, coord_rot_x)
pts = np.matmul(pts, coord_rot_total.transpose())

if 'object' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['object'], do_unlink=True)

if 'sphere' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['sphere'], do_unlink=True)

if 'object' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['object'], do_unlink=True)

sphere_mesh = bpy.data.meshes.new('sphere')
sphere_bmesh = bmesh.new()
bmesh.ops.create_icosphere(sphere_bmesh, subdivisions=2, diameter=point_radius*2)
sphere_bmesh.to_mesh(sphere_mesh)
sphere_bmesh.free()

sphere_verts = np.array([[v.co.x, v.co.y, v.co.z] for v in sphere_mesh.vertices])
sphere_faces = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in sphere_mesh.polygons])

verts = (np.expand_dims(sphere_verts, axis=0) + np.expand_dims(pts, axis=1)).reshape(-1, 3)
faces = (np.expand_dims(sphere_faces, axis=0) + (np.arange(pts.shape[0]) * sphere_verts.shape[0]).reshape(-1, 1, 1)).reshape(-1, 3)
vert_colors = np.repeat(ptc, sphere_verts.shape[0], axis=0).astype(dtype='float64')
vert_colors = vert_colors[faces.reshape(-1), :]

verts[:, 2] -= verts.min(axis=0)[2]

print(verts.shape, faces.shape, vert_colors.shape)

verts = verts.tolist()
faces = faces.tolist()
vert_colors = vert_colors.tolist()

scene = bpy.context.scene
mesh = bpy.data.meshes.new('object')
mesh.from_pydata(verts, [], faces)
mesh.validate()

mesh.vertex_colors.new(name='Col') # named 'Col' by default

if ptc is None:
    for i, c in enumerate(mesh.vertex_colors['Col'].data):
        c.color = candidate_colors[0]
else:
    for i, c in enumerate(mesh.vertex_colors['Col'].data):
        c.color = vert_colors[i]


obj = bpy.data.objects.new('object', mesh)
obj.data.materials.append(bpy.data.materials['sphere_material'])
scene.objects.link(obj)

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = output_filename
bpy.ops.render.render(write_still=True)
