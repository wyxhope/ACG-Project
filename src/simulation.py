import bpy
from . import rigid_body 
import numpy as np
import math
import os

class Renderer:
    def __init__(self, output_dir, resolution=(1280, 720)):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        bpy.ops.wm.read_homefile(use_empty=True)

        self.scene = bpy.context.scene

        bpy.context.scene.render.engine = 'CYCLES' 
        bpy.context.scene.cycles.samples = 64     
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
            print(f"Using device: {device.name}")
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]


        # set camera position
        self.set_camera(location=(0, -10, 8), rotation_euler=(math.radians(60), 0, 0))
        # bpy.ops.object.camera_add(location=(0, -10, 8), rotation=(math.radians(60), 0, 0))
        # bpy.context.scene.camera = bpy.context.object

        # set light
        # bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        # sun = bpy.context.object
        # sun.data.energy = 5.0

        self.setup_world(strength=1.0)

        self.objects = {}
        self.materials = {}
    
    def set_light(self, name, light_type='SUN', location=(5,5,10), energy=5.0, color=(1,1,1), rotation_euler=(0,0,0)):
        if name in self.objects:
            light = self.objects[name]
        else:
            if name in bpy.data.lights:
                light = bpy.data.lights[name]
            else:
                bpy.ops.object.light_add(type=light_type, location=location)
                light = bpy.context.object
                light.name = name
        
            self.objects[name] = light
        
        light.location = location
        light.rotation_euler = rotation_euler

        if light.data:
            light.data.energy = energy
            light.data.color = color
            light.data.type = light_type

    
    def set_camera(self, location, rotation_euler):
        if self.scene.camera is None:
            bpy.ops.object.camera_add(location=location, rotation=rotation_euler)
            self.scene.camera = bpy.context.object
        else:
            cam = self.scene.camera
            cam.location = location
            cam.rotation_euler = rotation_euler
    
    def setup_world(self, strength=1.0, sun_size=math.radians(0.145), sun_elevation=math.radians(5.2), sun_rotation=math.radians(0)
                    , sun_intensity=1.0, air_density=1.0, aerosol_density=0.7, ozone_density=1.0):

        world = bpy.data.worlds.new("World")
        self.scene.world = world
        world.use_nodes = True
        
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        nodes.clear()
        
        node_out = nodes.new(type='ShaderNodeOutputWorld')
        node_out.location = (300, 0)
        
        node_bg = nodes.new(type='ShaderNodeBackground')
        node_bg.location = (0, 0)
        node_bg.inputs['Strength'].default_value = strength
        
        node_sky = nodes.new(type='ShaderNodeTexSky')
        node_sky.location = (-300, 0)
        node_sky.sky_type = 'MULTIPLE_SCATTERING'

        node_sky.sun_disc = True

        node_sky.sun_size = sun_size
        node_sky.sun_intensity = sun_intensity

        node_sky.sun_elevation = sun_elevation
        node_sky.sun_rotation = sun_rotation
        
        node_sky.air_density = air_density
        node_sky.aerosol_density = aerosol_density
        node_sky.ozone_density = ozone_density

        links.new(node_sky.outputs['Color'], node_bg.inputs['Color'])
        links.new(node_bg.outputs['Background'], node_out.inputs['Surface'])
    
    def get_material(self, name: str, color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.5):
        # Set material for an object with name
        if name in self.materials:
            return self.materials[name]
        
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if principled_bsdf:
            principled_bsdf.inputs['Base Color'].default_value = color
            principled_bsdf.inputs['Metallic'].default_value = metallic
            principled_bsdf.inputs['Roughness'].default_value = roughness
            principled_bsdf.inputs['Transmission Weight'].default_value = 0.0

        self.materials[name] = mat
        return self.materials[name]
    
    def update_rigid_body(self, rb: rigid_body.RigidBody, name: str, material_parameters: dict = None):
        pos = rb.pos_of_center.to_numpy()
        quat = rb.quat.to_numpy()

        if name not in self.objects:

            vertices = rb.vertices.to_numpy()
            faces = rb.faces.to_numpy()

            mesh_data = bpy.data.meshes.new(name)
            mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
            mesh_data.update()

            obj = bpy.data.objects.new(name, mesh_data)
            self.scene.collection.objects.link(obj)

            # Default material
            mat_args = {
                'color': (0.8, 0.8, 0.8, 1.0), 
                'metallic': 0.0,
                'roughness': 0.5
            }
            if material_parameters is not None:
                mat_args.update(material_parameters)
            obj.data.materials.append(self.get_material(name + "_mat", mat_args['color'], mat_args['metallic'], mat_args['roughness']))

            self.objects[name] = obj
        
        obj = self.objects[name]
        obj.location = pos
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = quat

    def add_static_mesh(self, trimesh_obj, name, material_names="Glass"):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(trimesh_obj.vertices.tolist(), [], trimesh_obj.faces.tolist())
        mesh.update()

        obj = bpy.data.objects.new(name, mesh)
        self.scene.collection.objects.link(obj)
        self.objects[name] = obj

        mat = self.get_material(material_names, color=(0.9, 0.9, 0.9, 1.0), metallic=0.1, roughness=0.05)
        if mat.node_tree.nodes.get('Principled BSDF'):
            principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if material_names == "Glass":
                principled_bsdf.inputs['Transmission Weight'].default_value = 1.0
                principled_bsdf.inputs['Roughness'].default_value = 0.0
                principled_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
                principled_bsdf.inputs['IOR'].default_value = 1.45
            elif material_names == "Metal":
                principled_bsdf.inputs['Metallic'].default_value = 1.0
                principled_bsdf.inputs['Roughness'].default_value = 0.2
                principled_bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    
    
    def _ensure_fluid_object(self, name, particle_radius):
        if name in self.objects:
            return self.objects[name]

        # 1. 创建物体
        mesh = bpy.data.meshes.new(name + "_Mesh")
        obj = bpy.data.objects.new(name, mesh)
        self.scene.collection.objects.link(obj)
        self.objects[name] = obj

        # 2. 设置材质
        mat = self.get_material("Water_Mat", color=(0.8, 0.9, 1.0, 1.0), roughness=0.0)
        if mat.node_tree.nodes.get('Principled BSDF'):
            bsdf = mat.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Transmission Weight'].default_value = 1.0 # 全透射
            bsdf.inputs['IOR'].default_value = 1.333
            bsdf.inputs['Base Color'].default_value = (0.9, 0.95, 1.0, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.0
        obj.data.materials.append(mat)

        # 3. 设置几何节点 (完全复用 load_fluid_from_ply 的逻辑)
        modifier = obj.modifiers.new(name="FluidGN", type='NODES')
        node_group = bpy.data.node_groups.new('FluidGeoNodes', 'GeometryNodeTree')
        modifier.node_group = node_group

        if hasattr(node_group, "interface"):
            node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
            node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

        nodes = node_group.nodes
        links = node_group.links
        nodes.clear()

        # 节点创建
        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-600, 0)
        
        mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
        mesh_to_points.location = (-400, 0)
        mesh_to_points.inputs['Radius'].default_value = 0.0 # 半径由下一步控制

        points_to_vol = nodes.new('GeometryNodePointsToVolume')
        points_to_vol.location = (-200, 0)
        points_to_vol.inputs['Radius'].default_value = particle_radius * 1.8 # 增大半径促进融合
        points_to_vol.inputs['Voxel Amount'].default_value = 128 # 提高精度
        points_to_vol.inputs['Density'].default_value = 10.0

        vol_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
        vol_to_mesh.location = (0, 0)
        vol_to_mesh.inputs['Threshold'].default_value = 0.5
        vol_to_mesh.inputs['Adaptivity'].default_value = 0.1

        set_smooth = nodes.new('GeometryNodeSetShadeSmooth') # 关键：平滑着色
        set_smooth.location = (200, 0)
        
        set_mat = nodes.new('GeometryNodeSetMaterial')
        set_mat.location = (400, 0)
        set_mat.inputs['Material'].default_value = mat

        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (600, 0)

        # 连接节点
        links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
        links.new(mesh_to_points.outputs['Points'], points_to_vol.inputs['Points'])
        links.new(points_to_vol.outputs['Volume'], vol_to_mesh.inputs['Volume'])
        links.new(vol_to_mesh.outputs['Mesh'], set_smooth.inputs['Geometry'])
        links.new(set_smooth.outputs['Geometry'], set_mat.inputs['Geometry'])
        links.new(set_mat.outputs['Geometry'], output_node.inputs['Geometry'])

        return obj
    
    def update_fluid(self, particle_positions, name="Fluid", particle_radius=0.1):
        self._ensure_fluid_object(name, particle_radius)
        obj = self.objects[name]
        mesh = obj.data

        num_particles = particle_positions.shape[0]

        if len(mesh.vertices) != num_particles:
            mesh.clear_geometry()
            mesh.from_pydata(particle_positions.reshape(-1, 3), [], [])
        else:
            # 快速更新
            mesh.vertices.foreach_set("co", particle_positions.flatten())
        mesh.update()
            
        

        
    
    def render_frame(self, frame_idx: int):
        filepath = os.path.join(self.output_dir, f"rigid_frame_{frame_idx:04d}.png")
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

    def load_fluid_from_ply(self, ply_path, name="Fluid", particle_radius=0.1):
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file not found: {ply_path}")
            return

        positions = None
        try:
            with open(ply_path, 'rb') as f:
                num_particles = 0
                while True:
                    line = f.readline()
                    if b"element vertex" in line:
                        num_particles = int(line.split()[-1])
                    if b"end_header" in line:
                        break
                
                data = f.read(num_particles * 3 * 4)
                positions = np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            print(f"Error reading PLY {ply_path}: {e}")
            return

        if name not in self.objects:
            mesh = bpy.data.meshes.new(name + "_Mesh")
            obj = bpy.data.objects.new(name, mesh)
            self.scene.collection.objects.link(obj)
            self.objects[name] = obj

            mat = self.get_material("Water_Mat", color=(0.8, 0.9, 1.0, 1.0), roughness=0.0)
            if mat.node_tree.nodes.get('Principled BSDF'):
                bsdf = mat.node_tree.nodes['Principled BSDF']
                bsdf.inputs['Transmission Weight'].default_value = 0.8
                bsdf.inputs['IOR'].default_value = 1.333
                bsdf.inputs['Base Color'].default_value = (0.9, 0.95, 1.0, 1.0)
            obj.data.materials.append(mat)


            modifier = obj.modifiers.new(name="FluidGN", type='NODES')
            node_group = bpy.data.node_groups.new('FluidGeoNodes', 'GeometryNodeTree')
            modifier.node_group = node_group

            if hasattr(node_group, "interface"):
                node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
                node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

            nodes = node_group.nodes
            links = node_group.links
            nodes.clear()


            input_node = nodes.new('NodeGroupInput')
            input_node.location = (-400, 0)
            
            mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
            mesh_to_points.location = (-200, 0)
            mesh_to_points.inputs['Radius'].default_value = particle_radius * 0.6

            points_to_vol = nodes.new('GeometryNodePointsToVolume')
            points_to_vol.location = (0, 0)
            points_to_vol.inputs['Radius'].default_value = particle_radius * 1.2

            points_to_vol.inputs['Voxel Amount'].default_value = 128

            vol_to_mesh = nodes.new('GeometryNodeVolumeToMesh')
            vol_to_mesh.location = (200, 0)

            set_smooth = nodes.new('GeometryNodeSetShadeSmooth')
            set_smooth.location = (200, 0)
            
            set_mat = nodes.new('GeometryNodeSetMaterial')
            set_mat.location = (400, 0)
            set_mat.inputs['Material'].default_value = mat

            output_node = nodes.new('NodeGroupOutput')
            output_node.location = (600, 0)

            # 连接节点
            links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
            links.new(mesh_to_points.outputs['Points'], points_to_vol.inputs['Points'])
            links.new(points_to_vol.outputs['Volume'], vol_to_mesh.inputs['Volume'])
            links.new(vol_to_mesh.outputs['Mesh'], set_smooth.inputs['Geometry'])
            links.new(set_smooth.outputs['Geometry'], set_mat.inputs['Geometry'])
            links.new(set_mat.outputs['Geometry'], output_node.inputs['Geometry'])


        obj = self.objects[name]
        mesh = obj.data
        

        if len(mesh.vertices) != num_particles:
            mesh.clear_geometry()
            mesh.from_pydata(positions.reshape(-1, 3), [], [])
        else:
            # 快速更新
            mesh.vertices.foreach_set("co", positions)
        
        mesh.update()
    
    def save_blend(self, filepath):
        folder = os.path.dirname(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        bpy.ops.wm.save_as_mainfile(filepath=filepath)