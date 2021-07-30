
# usage bash render_one_pc.sh path/to/input.pts path/to/output.png

blender/blender --background point_clouds.blend --python render_one_pc.py -- example_pc.pts example_pc_same_color.png 0 0 0
