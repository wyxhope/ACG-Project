import sys 
import os
import site
import math
user_site_packages = site.getusersitepackages()
if user_site_packages not in sys.path:
    sys.path.append(user_site_packages)

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import taichi as ti
import os
from src.rigid_body import *
from src.fluid import *
from src.rigid_fluid import FluidSimulator
from src.simulation import Renderer
from src.make_video import make_video

ti.init(arch=ti.gpu)

project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_dir, "output")
video_dir = os.path.join(output_dir, "video")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
ply_dir = os.path.join(output_dir, "ply")
if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)
fluid_output_dir = os.path.join(output_dir, "fluid_simulation")
if not os.path.exists(fluid_output_dir):
    os.makedirs(fluid_output_dir)
fluid_online_output_dir = os.path.join(output_dir, "fluid_online_simulation")
if not os.path.exists(fluid_online_output_dir):
    os.makedirs(fluid_online_output_dir)
fluid_rigid_output_dir = os.path.join(output_dir, "fluid_rigid_simulation")
if not os.path.exists(fluid_rigid_output_dir):
    os.makedirs(fluid_rigid_output_dir)
duck_output_dir = os.path.join(output_dir, "duck_simulation")
if not os.path.exists(duck_output_dir):
    os.makedirs(duck_output_dir)
cloth_output_dir = os.path.join(output_dir, "cloth_simulation")
if not os.path.exists(cloth_output_dir):
    os.makedirs(cloth_output_dir)

def rigid_body_simulation():
    rigid = RigidBody(pos=[0,0,2], type='sphere', mass=1.0, mesh=None, radius=1.0)
    renderer = Renderer(output_dir=output_dir)
    renderer.set_camera(location=[-5, -15, 1], rotation_euler=[1.5, 0.0, -0.3])
    gravity = ti.Vector([0.0, 0.0, -9.81])
    num_frames = 50
    for frame in range(num_frames):
        # Simple physics update
        rigid.apply_force(gravity * rigid.mass, dt=1/30)
        rigid.update(dt=1/30)

        renderer.update_rigid_body(rigid, name="sphere_1", material_parameters={'color': (0.1, 0.1, 0.8, 1.0), 'metallic': 0.5, 'roughness': 0.3})
        renderer.render_frame(frame)

    print("Rendering completed. Creating video...")
    video_path = os.path.join(video_dir, "rigid_body_simulation_1.mp4")
    make_video(output_dir, video_path, fps=30)
    print(f"Video saved to {video_path}")

def rigid_sphere_collision_simulation():
    output_dir_collision = os.path.join(output_dir, "rigid_sphere_collision_simulation")
    if not os.path.exists(output_dir_collision):
        os.makedirs(output_dir_collision)
    rigid1 = RigidBody(pos=[-2,0,0], type='sphere', mass=1.0, mesh=None, radius=1.0)
    rigid2 = RigidBody(pos=[2,0,0], type='sphere', mass=1.0, mesh=None, radius=1.0)
    rigid1.vel[None] = ti.Vector([5.0, 0.0, 0.0])
    rigid2.vel[None] = ti.Vector([-5.0, 0.0, 0.0])

    renderer = Renderer(output_dir=output_dir_collision)
    renderer.set_camera(location=[0, -15, 5], rotation_euler=[1.2, 0.0, 0.0])
    renderer.setup_world(
        strength=1.0, 
        sun_size=math.radians(0.145), 
        sun_elevation=math.radians(50), 
        sun_rotation=math.radians(0),
        sun_intensity=1.0,
        air_density=1.0,
        aerosol_density=0.7,
        ozone_density=1.0
    )

    num_frames = 100
    for frame in range(num_frames):

        rigid1.update(dt=1/30)
        rigid2.update(dt=1/30)
        sphere_collision_simulation(rigid1, rigid2, 1e-3, 0.8)

        renderer.update_rigid_body(rigid1, name="sphere_1", material_parameters={'color': (0.8, 0.1, 0.1, 1.0), 'metallic': 0.5, 'roughness': 0.3})
        renderer.update_rigid_body(rigid2, name="sphere_2", material_parameters={'color': (0.1, 0.8, 0.1, 1.0), 'metallic': 0.5, 'roughness': 0.3})
        renderer.render_frame(frame)

    print("Rendering completed. Creating video...")
    video_path = os.path.join(video_dir, "rigid_sphere_collision_simulation.mp4")
    make_video(output_dir_collision, video_path, fps=50)
    print(f"Video saved to {video_path}")

def fluid_simulation():
    particle_radius = 0.03
    def run_simulation_export(num_frames=100):
        print("=== Phase 1: Starting Simulation & Export ===")
        

        container_size = [4.0, 4.0, 4.0]
        container_pos = [-2.0, -2.0, 0.0] 
        container = Container(position=container_pos, boundary_box=container_size)
        

        fluid_pos = [-1.0, -1.0, 0.5]
        fluid = Fluid(max_particles=5000000, init_box=(2.0, 2.0, 2.0), position=fluid_pos, particle_radius=particle_radius)
        # 初始化在容器左下角上方一点
        fluid.init_cube(spacing=particle_radius*2.0) 
        
        simulator = FluidSimulator(fluid, container)
        
        dt = 1.0 / 30.0

        for frame in range(num_frames):
            # 物理步进
            simulator.step(dt)
            
            # 保存 PLY
            ply_path = os.path.join(ply_dir, f"fluid_{frame:04d}.ply")
            fluid.save_ply(ply_path)
            
            print(f"Simulated & Exported Frame {frame}/{num_frames}")
        

        
        print("=== Simulation Finished ===")

    # ==========================================
    # 阶段 2: 渲染与合成 (Rendering)
    # ==========================================
    def render_simulation_import(num_frames=100):
        print("=== Phase 2: Starting Rendering ===")
        
        # 1. 初始化渲染器
        renderer = Renderer(output_dir=fluid_output_dir)
        renderer.set_camera(location=[-2, -11, 5], rotation_euler=[math.radians(72), 0.0, math.radians(-10)])
        renderer.setup_world(strength=1.2) # 稍微调亮环境光

        # 2. 加载静态容器 (只需要加载一次)
        # 我们需要重新创建 Container 对象来获取它的 mesh 数据
        container_size = [4.0, 4.0, 4.0]
        container_pos = [-2.0, -2.0, 0.0]
        container = Container(position=container_pos, boundary_box=container_size)
        
        # 将容器添加到 Blender
        renderer.add_static_mesh(container.mesh, name="GlassContainer", material_names="Glass")
        renderer.objects["GlassContainer"].hide_render = True 
        renderer.objects["GlassContainer"].hide_viewport = True

        # 3. 逐帧渲染
        for frame in range(num_frames):
            ply_path = os.path.join(ply_dir, f"fluid_{frame:04d}.ply")
            
            if os.path.exists(ply_path):
                renderer.load_fluid_from_ply(ply_path, name="Water", particle_radius=particle_radius)
                
                renderer.render_frame(frame)
                print(f"Rendered Frame {frame}/{num_frames}")
            else:
                print(f"Skipping Frame {frame} (PLY not found)")
        
        debug_blend_path = os.path.join(output_dir, "debug_scene.blend")
        renderer.save_blend(debug_blend_path)
        print(f"Debug scene saved to: {debug_blend_path}")

        # 4. 生成视频
        print("Creating video...")
        video_path = os.path.join(video_dir, "fluid_offline_render.mp4")
        make_video(fluid_output_dir, video_path, fps=30)
        print(f"Done! Video saved to {video_path}")
    
    num_frames =100
    run_simulation_export(num_frames=num_frames)
    render_simulation_import(num_frames=num_frames)

def fluid_online_simulation():
    particle_radius = 0.02
    num_frames = 150
    dt = 1.0 / 30.0

    print("=== Starting Online Simulation & Rendering ===")

    # 1. 初始化物理引擎
    container_size = [4.0, 4.0, 4.0]
    container_pos = [-2.0, -2.0, 0.0] 
    container = Container(position=container_pos, boundary_box=container_size)
    
    fluid_pos = [-1.0, -1.0, 0.5]
    fluid = Fluid(max_particles=500000, init_box=(2.0, 2.0, 2.0), position=fluid_pos, particle_radius=particle_radius, viscosity=1.0)
    fluid.init_cube(spacing=particle_radius*2.0) 
    simulator = FluidSimulator(fluid, container, rigid_bodies=[], has_rigid=False)

    # 2. 初始化渲染器
    # 注意：这里直接使用 fluid_output_dir，或者你可以新建一个 fluid_online_output_dir
    renderer = Renderer(output_dir=fluid_online_output_dir)
    renderer.set_camera(location=[-2, -11, 5], rotation_euler=[math.radians(72), 0.0, math.radians(-10)])
    renderer.setup_world(strength=1.2)
    
    # 加载容器
    renderer.add_static_mesh(container.mesh, name="GlassContainer", material_names="Glass")
    renderer.objects["GlassContainer"].hide_render = True 
    renderer.objects["GlassContainer"].hide_viewport = True

    # 3. 主循环：模拟 -> 传输 -> 渲染
    for frame in range(num_frames):
        # --- A. 物理计算 (GPU) ---
        simulator.step(dt)
        
        # --- B. 数据传输 (内存复制, 极快) ---
        # 获取有效粒子数
        num_p = fluid.num_particles[None]
        # 获取 numpy 数组 (切片只取有效部分)
        # 注意：to_numpy() 会将数据从 GPU 显存拉回 CPU 内存
        positions = fluid.pos.to_numpy()[:num_p]
        
        # --- C. 更新 Blender (内存操作) ---
        # 调用我们在 Renderer 中新加的 update_fluid 方法
        renderer.update_fluid(positions, name="Water", particle_radius=particle_radius)
        
        # --- D. 渲染 (GPU/CPU) ---
        renderer.render_frame(frame)
        
        print(f"Finished Frame {frame}/{num_frames}")

    # 4. 保存调试场景
    debug_blend_path = os.path.join(output_dir, "debug_online_scene.blend")
    renderer.save_blend(debug_blend_path)
    print(f"Debug scene saved to: {debug_blend_path}")

    # 5. 生成视频
    print("Creating video...")
    video_path = os.path.join(video_dir, "fluid_online_render.mp4")
    make_video(fluid_online_output_dir, video_path, fps=30)
    print(f"Done! Video saved to {video_path}")

def rigid_fluid_interaction_simulation():
    particle_radius = 0.02
    num_frames = 100
    dt = 1.0 / 30.0

    print("=== Starting Online Simulation & Rendering ===")

    # 1. 初始化物理引擎
    container_size = [4.0, 4.0, 4.0]
    container_pos = [-2.0, -2.0, 0.0] 
    container = Container(position=container_pos, boundary_box=container_size)
    
    fluid_pos = [-1.0, -1.0, 0.5]
    fluid = Fluid(max_particles=500000, init_box=(2.0, 2.0, 2.0), position=fluid_pos, particle_radius=particle_radius, viscosity=1.0)
    fluid.init_cube(spacing=particle_radius*2.0) 

    rigid_body = RigidBody(pos=[0,0,4], type='sphere', mass=1.0, mesh=None, radius=0.5)

    simulator = FluidSimulator(fluid, container, rigid_bodies=[rigid_body])



    # 2. 初始化渲染器
    # 注意：这里直接使用 fluid_output_dir，或者你可以新建一个 fluid_online_output_dir
    renderer = Renderer(output_dir=fluid_rigid_output_dir)
    renderer.set_camera(location=[-2, -11, 5], rotation_euler=[math.radians(72), 0.0, math.radians(-10)])
    renderer.setup_world(strength=1.2)
    
    # 加载容器
    renderer.add_static_mesh(container.mesh, name="GlassContainer", material_names="Glass")
    renderer.objects["GlassContainer"].hide_render = True 
    renderer.objects["GlassContainer"].hide_viewport = True

    # 3. 主循环：模拟 -> 传输 -> 渲染
    for frame in range(num_frames):
        # --- A. 物理计算 (GPU) ---
        simulator.step(dt)
        
        # --- B. 数据传输 (内存复制, 极快) ---
        # 获取有效粒子数
        num_p = fluid.num_particles[None]
        # 获取 numpy 数组 (切片只取有效部分)
        # 注意：to_numpy() 会将数据从 GPU 显存拉回 CPU 内存
        positions = fluid.pos.to_numpy()[:num_p]
        
        # --- C. 更新 Blender (内存操作) ---
        # 调用我们在 Renderer 中新加的 update_fluid 方法
        renderer.update_fluid(positions, name="Water", particle_radius=particle_radius)

        renderer.update_rigid_body(rigid_body, name="RigidSphere", material_parameters={'color': (0.1, 0.1, 0.8, 1.0), 'metallic': 0.5, 'roughness': 0.3})
        
        # --- D. 渲染 (GPU/CPU) ---
        renderer.render_frame(frame)
        
        print(f"Finished Frame {frame}/{num_frames}")

    # 4. 保存调试场景
    debug_blend_path = os.path.join(output_dir, "debug_online_scene.blend")
    renderer.save_blend(debug_blend_path)
    print(f"Debug scene saved to: {debug_blend_path}")

    # 5. 生成视频
    print("Creating video...")
    video_path = os.path.join(video_dir, "fluid_rigid_render.mp4")
    make_video(fluid_rigid_output_dir, video_path, fps=30)
    print(f"Done! Video saved to {video_path}")

def duck_simulation():
    particle_radius = 0.02
    num_frames = 150
    dt = 1.0 / 30.0

    print("=== Starting Online Simulation & Rendering ===")

    # 1. 初始化物理引擎
    container_size = [4.0, 4.0, 4.0]
    container_pos = [-2.0, -2.0, 0.0] 
    container = Container(position=container_pos, boundary_box=container_size)
    
    fluid_pos = [-1.0, -1.0, 0.5]
    fluid = Fluid(max_particles=500000, init_box=(2.0, 2.0, 2.0), position=fluid_pos, particle_radius=particle_radius, viscosity=1.0)
    fluid.init_cube(spacing=particle_radius*2.0) 

    obj_path = os.path.join(project_dir, "data", "Duck_1204072310_texture_obj", "Duck_1204072310_texture.obj")
    mesh = trimesh.load(obj_path)
    init_quat = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90 degrees around x-axis
    rigid_body = RigidBody(pos=[0,0,4], type='mesh', mass=2.0, mesh=mesh, radius=0.5, rotation_quat=init_quat, scale=(0.8, 0.8, 0.8))

    simulator = FluidSimulator(fluid, container, rigid_bodies=[rigid_body], has_rigid=True)



    # 2. 初始化渲染器
    # 注意：这里直接使用 fluid_output_dir，或者你可以新建一个 fluid_online_output_dir
    renderer = Renderer(output_dir=duck_output_dir)
    renderer.set_camera(location=[-2, -11, 5], rotation_euler=[math.radians(72), 0.0, math.radians(-10)])
    renderer.setup_world(strength=1.2)
    
    # 加载容器
    renderer.add_static_mesh(container.mesh, name="GlassContainer", material_names="Glass")
    renderer.load_obj(obj_path, name="Duck", scale=(1,1,1), set_origin_to_geometry=True)
    renderer.objects["GlassContainer"].hide_render = True 
    renderer.objects["GlassContainer"].hide_viewport = True

    # 3. 主循环：模拟 -> 传输 -> 渲染
    for frame in range(num_frames):
        # --- A. 物理计算 (GPU) ---
        simulator.step(dt)
        
        # --- B. 数据传输 (内存复制, 极快) ---
        # 获取有效粒子数
        num_p = fluid.num_particles[None]
        # 获取 numpy 数组 (切片只取有效部分)
        # 注意：to_numpy() 会将数据从 GPU 显存拉回 CPU 内存
        positions = fluid.pos.to_numpy()[:num_p]
        
        # --- C. 更新 Blender (内存操作) ---
        # 调用我们在 Renderer 中新加的 update_fluid 方法
        renderer.update_fluid(positions, name="Water", particle_radius=particle_radius)

        renderer.update_rigid_body(rigid_body, name="Duck", material_parameters={'color': (0.1, 0.1, 0.8, 1.0), 'metallic': 0.5, 'roughness': 0.3})
        
        # --- D. 渲染 (GPU/CPU) ---
        renderer.render_frame(frame)
        
        print(f"Finished Frame {frame}/{num_frames}")

    # 4. 保存调试场景
    debug_blend_path = os.path.join(output_dir, "debug_online_scene.blend")
    renderer.save_blend(debug_blend_path)
    print(f"Debug scene saved to: {debug_blend_path}")

    # 5. 生成视频
    print("Creating video...")
    video_path = os.path.join(video_dir, "duck_fluid_render.mp4")
    make_video(duck_output_dir, video_path, fps=30)
    print(f"Done! Video saved to {video_path}")

def main():
    # rigid_body_simulation()
    # rigid_sphere_collision_simulation()
    # fluid_simulation()
    # fluid_online_simulation()
    # rigid_fluid_interaction_simulation()
    duck_simulation()



if __name__ == "__main__":
    main()