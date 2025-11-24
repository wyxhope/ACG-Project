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
        
    
    def render_frame(self, frame_idx: int):
        filepath = os.path.join(self.output_dir, f"rigid_frame_{frame_idx:04d}.png")
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)


