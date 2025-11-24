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
from src.simulation import Renderer
from src.make_video import make_video

ti.init(arch=ti.gpu)

project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_dir, "output")
video_dir = os.path.join(output_dir, "video")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

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


def main():
    # rigid_body_simulation()
    rigid_sphere_collision_simulation()


if __name__ == "__main__":
    main()