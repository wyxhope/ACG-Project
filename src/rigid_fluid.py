import taichi as ti
import trimesh
from .rigid_body import *
from .fluid import Fluid, Container

@ti.data_oriented
class FluidSimulator:
    def __init__(self, fluid: Fluid, container: Container, rigid_bodies: list[RigidBody]):
        self.fluid = fluid
        self.container = container
        self.rigid_bodies = rigid_bodies[0] # Now only use 1 rigid body, maybe not extend
    
    @ti.kernel
    def solve_rigid_interaction(self, dt: float):
        # 获取刚体对象
        rb = self.rigid_bodies
        
        for i in range(self.fluid.num_particles[None]):
            pos = self.fluid.pos[i]
            
            # 1. 查询 SDF：获取距离和法线
            dist, normal = rb.get_sdf(pos)
            
            # 2. 碰撞检测：如果距离小于粒子半径，视为接触/穿透
            if dist < self.fluid.particle_radius:
                # 计算穿透深度
                penetration = self.fluid.particle_radius - dist
                
                # 3. 计算相对速度
                # 刚体在接触点的速度 v_rb = v_cm + omega x r
                r = pos - rb.pos_of_center[None]
                v_rb = rb.vel[None] + rb.ang_vel[None].cross(r)
                
                v_rel = self.fluid.vel[i] - v_rb
                v_n = v_rel.dot(normal)
                
                # 4. 计算惩罚力 (弹簧-阻尼模型)
                k = 10000.0   # 刚度系数 (Stiffness)
                damping = 10.0 # 阻尼系数 (Damping)
                
                force_mag = k * penetration
                # 仅当粒子向刚体内部运动时施加阻尼
                if v_n < 0:
                    force_mag -= damping * v_n
                
                # 力的方向沿法线向外推粒子
                force = force_mag * normal
                
                # 6. 应用反作用力到刚体 (F_rb = -F_fluid)
                f_rb = -force
                t_rb = r.cross(f_rb) # 力矩 = r x F
                
                # 更新刚体线速度 (使用 atomic_add 防止并行冲突)
                # dv = F * dt / m
                dv = f_rb * dt / rb.mass
                ti.atomic_add(rb.vel[None][0], dv[0])
                ti.atomic_add(rb.vel[None][1], dv[1])
                ti.atomic_add(rb.vel[None][2], dv[2])
                
                # 更新刚体角速度
                # d_omega = I_inv_world * torque * dt
                # 需要计算世界坐标系下的惯性张量逆: I_inv_world = R @ I_inv_local @ R.T
                R = rb.quat_to_matrix(rb.quat[None])
                I_inv_world = R @ rb.I_inv[None] @ R.transpose()
                
                d_ang_vel = (I_inv_world @ t_rb) * dt
                
                ti.atomic_add(rb.ang_vel[None][0], d_ang_vel[0])
                ti.atomic_add(rb.ang_vel[None][1], d_ang_vel[1])
                ti.atomic_add(rb.ang_vel[None][2], d_ang_vel[2])

                if v_n < 0:
                    restitution = 0.0
                    friction = 0.5

                    v_normal = v_n * normal
                    v_tangent = v_rel - v_normal

                    v_rel_new = -restitution * v_normal + (1 - friction) * v_tangent
                    self.fluid.vel[i] = v_rel_new + v_rb
                
                self.fluid.pos[i] += normal * penetration 

    def step(self, dt):
        substeps = 50
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.fluid.update_grid()
            self.fluid.compute_density()
            self.fluid.compute_forces()

            if self.rigid_bodies:
                gravity_force = ti.Vector([0.0, 0.0, -9.8]) * self.rigid_bodies.mass
                self.rigid_bodies.apply_force(gravity_force, sub_dt)
                self.solve_rigid_interaction(sub_dt)
            self.fluid.integrate(sub_dt)
            self.container.enforce_boundary(self.fluid)
            if self.rigid_bodies:
                self.rigid_bodies.update(sub_dt)
