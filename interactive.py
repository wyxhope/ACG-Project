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
import numpy as np
from src.rigid_body import RigidBody
from src.cloth import Cloth


# Note: Raycasting functions removed for simplicity.
# Using direct keyboard control for better reliability.

def run_simple_render():
    # 1. 创建窗口
    ti.init(arch=ti.gpu)

    res = (1024, 1024)
    window = ti.ui.Window("Rigid Body Render Test", res)
    canvas = window.get_canvas()
    scene = window.get_scene()

    print("Successfully created Taichi window.")
    
    # 2. 配置相机
    camera = ti.ui.Camera()
    camera.position(0, -10, 5) # 站在 Y=-10 的位置往里看
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)

    print("Camera configured.")

    # 3. 创建刚体 (球体)
    # 根据你的类定义，mesh=None 时会自动生成球体网格
    rigid = RigidBody(pos=[0, 0, 0], type='sphere', mass=1.0, mesh=None, radius=1.0, is_fixed=True)

    # --- 修复：将 Vector 索引转为 Scalar 索引以供渲染 ---
    # 你的 rigid.faces 是 Vector.field(3)，渲染器需要一维 Scalar field
    rigid_indices = ti.field(int, shape=rigid.num_faces * 3)
    
    # 新增：用于渲染的世界坐标顶点 (World Space Vertices)
    # rigid.vertices 存储的是局部坐标，如果不转换，渲染出来的球永远会在原点
    rigid_vertices_render = ti.Vector.field(3, dtype=float, shape=rigid.num_vertices)

    @ti.kernel
    def flatten_indices(r_faces: ti.template(), r_ind: ti.template()):
        for i in range(r_faces.shape[0]):
            r_ind[i * 3 + 0] = r_faces[i][0]
            r_ind[i * 3 + 1] = r_faces[i][1]
            r_ind[i * 3 + 2] = r_faces[i][2]
    
    @ti.kernel
    def update_rigid_render():
        # 将局部顶点变换为世界空间顶点
        for i in range(rigid.num_vertices):
            rigid_vertices_render[i] = rigid.local_to_world(i)

    flatten_indices(rigid.faces, rigid_indices)

    # 4. 创建布料
    n_cloth = 32
    cloth = Cloth(N=n_cloth, pos_center=[0, 0, 2.5], size=4.0, stiffness=100)
    
    # 布料渲染准备
    cloth_indices_np = cloth.get_indices()
    cloth_indices = ti.field(int, shape=cloth_indices_np.shape[0])
    cloth_indices.from_numpy(cloth_indices_np)

    cloth_vertices = ti.Vector.field(3, dtype=float, shape=n_cloth * n_cloth)

    @ti.kernel
    def update_cloth_vertices():
        for i, j in cloth.pos:
            idx = i * n_cloth + j
            cloth_vertices[idx] = cloth.pos[i, j]

    @ti.kernel
    def fix_corners():
        # 固定两个角 (例如：i=0,j=0 和 i=N-1,j=0)
        cloth.is_fixed[0, 0] = 1
        cloth.is_fixed[n_cloth - 1, 0] = 1
        # 重置这两个点的速度，防止初始速度导致爆发
        cloth.vel[0, 0] = ti.Vector([0.0, 0.0, 0.0])
        cloth.vel[n_cloth - 1, 0] = ti.Vector([0.0, 0.0, 0.0])

    fix_corners()

    print("开始渲染循环... 请尝试使用鼠标右键旋转相机。")
    print("控制球体移动：[I/K] 前后, [J/L] 左右, [U/O] 上下")

    while window.running:
        # 更新相机：右键旋转，WASDQE 移动
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # --- Keyboard Interaction Logic (Simpler & More Robust) ---
        move_speed = 0.05
        moved = False
        if window.is_pressed('i'): 
            rigid.pos_of_center[None].y += move_speed
            moved = True
        if window.is_pressed('k'): 
            rigid.pos_of_center[None].y -= move_speed
            moved = True
        if window.is_pressed('j'): 
            rigid.pos_of_center[None].x -= move_speed
            moved = True
        if window.is_pressed('l'): 
            rigid.pos_of_center[None].x += move_speed
            moved = True
        if window.is_pressed('u'): 
            rigid.pos_of_center[None].z += move_speed
            moved = True
        if window.is_pressed('o'): 
            rigid.pos_of_center[None].z -= move_speed
            moved = True
            
        if moved:
            # 拖动时重置速度，防止物理引擎抵抗移动
            rigid.vel[None] = ti.Vector([0.0, 0.0, 0.0])
            rigid.ang_vel[None] = ti.Vector([0.0, 0.0, 0.0])
        # -----------------------------

        rigid.update_aabb()

        # 灯光
        scene.point_light(pos=(0, -10, 10), color=(1, 1, 1))
        scene.ambient_light((0.3, 0.3, 0.3))

        rigid.update_aabb()
        # 物理模拟
        # 优化后：融合内核减少了开销，substeps 降低到 30 (足以保持稳定性)
        cloth.step(dt=0.005, rigid_bodies=[rigid], substeps=100)
        update_cloth_vertices()
        
        # 更新刚体渲染网格位置
        update_rigid_render()

        # 渲染刚体网格
        # 使用变换后的 rigid_vertices_render，而不是原始的 rigid.vertices
        scene.mesh(rigid_vertices_render, indices=rigid_indices, color=(0.8, 0.2, 0.2))

        # 渲染布料
        scene.mesh(cloth_vertices, indices=cloth_indices, color=(0.2, 0.8, 0.8), two_sided=True)

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    run_simple_render()