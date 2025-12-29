import taichi as ti
import numpy as np
from src.rigid_body import RigidBody

ti.init(arch=ti.gpu)

def run_simple_render():
    # 1. 创建窗口
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
    rigid = RigidBody(pos=[0, 0, 0], type='sphere', mass=1.0, mesh=None, radius=1.0)

    # --- 修复：将 Vector 索引转为 Scalar 索引以供渲染 ---
    # 你的 rigid.faces 是 Vector.field(3)，渲染器需要一维 Scalar field
    rigid_indices = ti.field(int, shape=rigid.num_faces * 3)

    @ti.kernel
    def flatten_indices(r_faces: ti.template(), r_ind: ti.template()):
        for i in range(r_faces.shape[0]):
            r_ind[i * 3 + 0] = r_faces[i][0]
            r_ind[i * 3 + 1] = r_faces[i][1]
            r_ind[i * 3 + 2] = r_faces[i][2]

    flatten_indices(rigid.faces, rigid_indices)

    print("开始渲染循环... 请尝试使用鼠标右键旋转相机。")

    while window.running:
        # 更新相机：右键旋转，WASDQE 移动
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # 灯光
        scene.point_light(pos=(0, -10, 10), color=(1, 1, 1))
        scene.ambient_light((0.3, 0.3, 0.3))

        # 渲染刚体网格
        # vertices 是 Vector.field(3)，indices 必须是刚才拉平的 Scalar field
        scene.mesh(rigid.vertices, indices=rigid_indices, color=(0.8, 0.2, 0.2))

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    run_simple_render()