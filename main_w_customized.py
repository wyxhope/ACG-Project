import sys
import os
import site
import math
import json
import time
import numpy as np
import concurrent.futures
import itertools


user_site_packages = site.getusersitepackages()
if user_site_packages not in sys.path:
    sys.path.append(user_site_packages)

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import taichi as ti
from src.rigid_body import *
from src.fluid import *
from src.rigid_fluid import FluidSimulator
from src.simulation import Renderer
from src.make_video import make_video
from src.cloth import Cloth

def load_config(config_path):
    """Loads simulation configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

@ti.kernel
def fix_net_corners(c: ti.template()):
    for i, j in c.is_fixed:
        if (i == 0 or i == c.N-1) and (j == 0 or j == c.N-1):
            c.is_fixed[i, j] = 1

@ti.kernel
def fix_curtain_hooks(c: ti.template()):
    for i in range(c.N):
        if i % 8 == 0:
            c.is_fixed[i, c.N - 1] = 1

def custom_rigid_body_simulation(config):
    """Runs a rigid body simulation based on the provided configuration."""
    
    # --- 1. Setup Directories and Parameters ---
    output_settings = config['output_settings']
    sim_params = config['simulation_params']
    
    output_dir = os.path.join(project_root, output_settings['output_dir'])
    video_path = os.path.join(project_root, output_settings['video_path'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure the directory for the video exists
    video_dir = os.path.dirname(video_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    num_frames = sim_params['num_frames']
    dt = sim_params['dt']
    gravity = ti.Vector(sim_params['gravity'])
    threshold = sim_params.get('threshold', 1e-3)

    # --- 2. Initialize Taichi ---
    ti.init(arch=ti.gpu)
    # --- 3. Setup Renderer ---
    camera_settings = config['camera_settings']
    
    rotation_in_radians = [math.radians(angle) for angle in camera_settings['rotation_euler']]


    renderer = Renderer(output_dir=output_dir)
    renderer.set_camera(
        location=camera_settings['location'],
        rotation_euler=rotation_in_radians
    )


    # --- 4. Create Rigid Bodies from Config ---
    bodies = []
    bodie = []
    for body_conf in config['rigid_bodies']:
        
        body = RigidBody(
            pos=body_conf['pos'],
            type=body_conf['type'],
            mass=body_conf['mass'],
            radius=body_conf.get('radius'),
            is_fixed=body_conf.get('is_fixed'),
            shape=body_conf.get('shape'),
            restitution=body_conf.get('restitution', 0.9),
            velocity=np.array(body_conf.get('velocity', [0,0,0]))
        )
        bodie.append(body)
        bodies.append({'obj': body, 'conf': body_conf})

    nets = []
    curtains = []
    for cloth_conf in config.get('cloths', []):
        cloth = Cloth(
            N=cloth_conf['N'],
            pos_center=cloth_conf['pos_center'],
            size=cloth_conf['size'],
            is_curtain=cloth_conf.get('is_curtain', False),
            stiffness=cloth_conf.get('stiffness', 100.0),
            damping=cloth_conf.get('damping', 10.0),
            mass=cloth_conf.get('mass', 1.0),
            compress_ratio=cloth_conf.get('compress_ratio', 0.5)
        )
        if cloth_conf.get('is_curtain', True) == False:
            fix_net_corners(cloth)
            nets.append({'obj': cloth, 'conf': cloth_conf})
        if cloth_conf.get('is_curtain', False):
            fix_curtain_hooks(cloth)
            curtains.append({'obj': cloth, 'conf': cloth_conf})

    fluid_conf = config.get('water', [])
    container = Container(fluid_conf['container_pos'], fluid_conf['container_shape'])
    fluid = Fluid(max_particles=fluid_conf['max_particles'],
                      position=fluid_conf['position'],
                      init_box=fluid_conf['init_box'],
                        particle_radius=fluid_conf['particle_radius'],
                        gravity=gravity)
    fluid.init_cube(spacing = fluid_conf.get('initial_distance', 2.0) * fluid_conf['particle_radius'])
    renderer.add_static_mesh(container.mesh, name=fluid_conf['name'] + "_container", material_names="Glass")
    renderer.objects[fluid_conf['name'] + "_container"].hide_render = True 
    renderer.objects[fluid_conf['name'] + "_container"].hide_viewport = True
    simulator = FluidSimulator(fluid, container, rigid_bodies=bodie, has_rigid=len(bodie)>0)
    water = {'water': fluid, 'simulator': simulator, 'conf': fluid_conf}

    # --- 5. Simulation Loop ---
    
    # Profiling stats
    total_time_cloth = 0
    total_time_fluid = 0
    total_time_collision = 0
    total_time_render = 0

    print("Warming up...")
    for frame in range(10):
        for curtain in curtains:
            curtain['obj'].step(dt, substeps=8000, wind_t=frame*dt)

    print("Starting Main Loop...")
    
    candidates = list(itertools.combinations(range(len(bodies)), 2))
    for frame in range(num_frames):
        t0 = time.time()
        for curtain in curtains:
            curtain['obj'].step(dt, substeps=8000, wind_t=frame*dt, gravity=gravity)
        for net in nets:
            net['obj'].step(dt, substeps=5000, rigid_bodies=bodie, gravity=gravity, thickness=0.2)
        t1 = time.time()
        total_time_cloth += (t1 - t0)

        t0 = time.time()
        water['simulator'].step(dt)
        t1 = time.time()
        total_time_fluid += (t1 - t0)
        

        t0 = time.time()
        
        for i, j in candidates:
            body1_dict = bodies[i]
            body2_dict = bodies[j]
            
            b1 = body1_dict['obj']
            b2 = body2_dict['obj']

            if b1.type == 'sphere' and b2.type == 'sphere':
                sphere_collision_simulation(b1, b2, threshold)
        
            if b1.type == 'sphere' and b2.type == 'box':
                sphere_box_collision_simulation(b1, b2, threshold)
            if b1.type == 'box' and b2.type == 'sphere':
                sphere_box_collision_simulation(b2, b1, threshold)
            if b1.type == 'box' and b2.type == 'box':
                box_collision_simulation(b1, b2, threshold)
        t1 = time.time()
        total_time_collision += (t1 - t0)

        # --- 6. Render Frame ---
        t0 = time.time()
        p_np = water['water'].pos.to_numpy()[:water['water'].num_particles[None]]
        renderer.update_fluid(p_np, name=water['conf']['name'], particle_radius=water['conf']['particle_radius'])



        for body_dict in bodies:
            renderer.update_rigid_body(
                body_dict['obj'],
                name=body_dict['conf']['name'],
                material_parameters=body_dict['conf']['material']
            )
        for net in nets:
            renderer.update_cloth(net['obj'], name=net['conf']['name'], material_params=net['conf'].get('material', {'color': (0.2, 0.8, 0.2, 1.0)}))
        for curtain in curtains:
            renderer.update_cloth(curtain['obj'], name=curtain['conf']['name'], material_params=curtain['conf'].get('material', {'color': (0.8, 0.2, 0.2, 1.0)}))
        renderer.render_frame(frame)
        print(f"Rendered Frame {frame}/{num_frames}")
        t1 = time.time()
        total_time_render += (t1 - t0)
    
    print("\n--- Timing Analysis (Average per frame) ---")
    if num_frames > 0:
        print(f"  Cloth/Net Step: {total_time_cloth / num_frames * 1000:.2f} ms")
        
        # Detailed Cloth Analysis
        total_cloth_forces = 0
        total_cloth_collision = 0
        total_cloth_update = 0
        
        # Aggregate stats from all cloths (nets and curtains)
        all_cloths = [c['obj'] for c in curtains] + [n['obj'] for n in nets]
        for cloth in all_cloths:
            if hasattr(cloth, 'breakdown_timers'):
                total_cloth_forces += cloth.breakdown_timers['forces']
                total_cloth_collision += cloth.breakdown_timers['collision']
                total_cloth_update += cloth.breakdown_timers['update']
        
        if all_cloths: # divided by num_frames to match previous scale
             print(f"    - Compute Forces: {total_cloth_forces / num_frames * 1000:.2f} ms")
             print(f"    - Rigid Collision: {total_cloth_collision / num_frames * 1000:.2f} ms")
             print(f"    - Position Update: {total_cloth_update / num_frames * 1000:.2f} ms")

        print(f"  Fluid Step    : {total_time_fluid / num_frames * 1000:.2f} ms")
        print(f"  Rigid Collision     : {total_time_collision / num_frames * 1000:.2f} ms")
        print(f"  Render        : {total_time_render / num_frames * 1000:.2f} ms")
    print("-------------------------------------------\n")

    # --- 7. Create Video ---
    print("Rendering completed. Creating video...")
    make_video(output_dir, video_path, fps=output_settings['fps'])
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    config_file = os.path.join(project_root, "scene_config.json")
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at {config_file}")
    else:
        config_data = load_config(config_file)
        custom_rigid_body_simulation(config_data)