import taichi as ti
import numpy as np

@ti.data_oriented
class Cloth:
    def __init__(self, N, pos_center, size, stiffness=1000.0, damping=2.0, mass=1.0):
        self.N = N
        self.num_particles = N * N
        self.pos = ti.Vector.field(3, dtype=float, shape=(N, N))
        self.vel = ti.Vector.field(3, dtype=float, shape=(N, N))
        self.force = ti.Vector.field(3, dtype=float, shape=(N, N))
        self.mass = mass / (N * N)
        self.stiffness = stiffness
        self.damping = damping
        
        # 弹簧连接关系 (x偏移, y偏移, 弹簧原长系数, 刚度系数)
        self.spring_offsets = []
        
        # 1. Structural (结构弹簧)
        self.spring_offsets.append(ti.Vector([1, 0]))
        self.spring_offsets.append(ti.Vector([0, 1]))
        
        # 2. Shear (剪切弹簧)
        self.spring_offsets.append(ti.Vector([1, 1]))
        self.spring_offsets.append(ti.Vector([1, -1]))
        
        # 3. Bending (弯曲弹簧) - 跨越两个格点
        self.spring_offsets.append(ti.Vector([2, 0]))
        self.spring_offsets.append(ti.Vector([0, 2]))

        self.rest_len = size / (N - 1)
        self.init_pos(pos_center, size)

    # @ti.kernel
    # def init_pos(self, center: ti.types.vector(3, float), size: float):
    #     offset = ti.Vector([center[0] - size/2, center[1] - size/2, center[2]])
    #     for i, j in self.pos:
    #         self.pos[i, j] = ti.Vector([i * self.rest_len, j * self.rest_len, 0.0]) + offset
    #         self.vel[i, j] = ti.Vector([0.0, 0.0, 0.0])
    @ti.kernel
    def init_pos(self, pos_center: ti.types.vector(3, float), size: float):
        for i, j in self.pos:
            # 使用相对于中心的偏移，确保所有计算都在显存内完成
            offset_x = (i / (self.N - 1) - 0.5) * size
            offset_y = (j / (self.N - 1) - 0.5) * size
            self.pos[i, j] = ti.Vector([pos_center[0] + offset_x, 
                                    pos_center[1] + offset_y, 
                                    pos_center[2]])
            self.vel[i, j] = ti.Vector([0, 0, 0])
            self.force[i, j] = ti.Vector([0, 0, 0])

    @ti.kernel
    def compute_forces(self, gravity: ti.types.vector(3, float)):
        # 1. 重力初始化
        for i, j in self.pos:
            self.force[i, j] = gravity * self.mass

        # 2. 弹簧力计算
        for i, j in self.pos:
            for k in ti.static(range(6)): # 遍历 6 种弹簧方向
                offset = self.spring_offsets[k]
                ni, nj = i + offset[0], j + offset[1]
                
                if 0 <= ni < self.N and 0 <= nj < self.N:
                    x_ij = self.pos[ni, nj] - self.pos[i, j]
                    v_ij = self.vel[ni, nj] - self.vel[i, j]
                    
                    dist = x_ij.norm()
                    
                    # 关键修复：增加距离阈值保护，并明确定义 dir 为向量
                    if dist > 1e-6:
                        direction = x_ij / dist  # 手动归一化，确保它是 Vector
                        
                        rest_length_k = offset.norm() * self.rest_len

                        # 计算 Hooke's Law (弹簧力)
                        current_stiffness = self.stiffness
                        if k >= 4: # 弯曲弹簧稍软一点
                            current_stiffness *= 0.5

                        force_mag = current_stiffness * (dist - rest_length_k)
                        
                        # 计算阻尼力 (Damping)
                        # 这里使用 direction 明确变量名，防止与内置关键词冲突
                        damping_mag = self.damping * v_ij.dot(direction)
                        
                        total_force_mag = force_mag + damping_mag
                        total_force_vec = total_force_mag * direction
                        
                        self.force[i, j] += total_force_vec
                        self.force[ni, nj] -= total_force_vec

    @ti.kernel
    def update(self, dt: float):
        for i, j in self.pos:
            # Simple Semi-implicit Euler
            acc = self.force[i, j] / self.mass
            self.vel[i, j] += acc * dt
            self.pos[i, j] += self.vel[i, j] * dt
    
    @ti.kernel
    def solve_rigid_collision(self, rb: ti.template()):
        """
        利用 RigidBody 的 SDF 进行碰撞解算
        """
        friction = 0.1
        restitution = 0 # 布料通常不想弹跳太大

        for i, j in self.pos:
            pos = self.pos[i, j]
            
            # 1. 查询 SDF
            dist, normal = rb.get_sdf(pos)
            
            # 2. 简单的碰撞阈值 (稍微留一点厚度)
            thickness = 0.08
            
            if dist < thickness:
                # 穿透深度
                penetration = thickness - dist
                
                # 3. 位置修正 (直接投影到表面)
                self.pos[i, j] += normal * penetration
                
                # 4. 速度修正
                # 刚体在该点的速度
                r = self.pos[i, j] - rb.pos_of_center[None]
                v_rb = rb.vel[None] + rb.ang_vel[None].cross(r)
                
                v_rel = self.vel[i, j] - v_rb
                v_n = v_rel.dot(normal)
                
                if v_n < 0:
                    v_normal = v_n * normal
                    v_tangent = v_rel - v_normal
                    
                    # 应用摩擦力和反弹
                    v_rel_new = -restitution * v_normal + (1 - friction) * v_tangent
                    
                    self.vel[i, j] = v_rel_new + v_rb
                    
                    # (可选) 对刚体施加反作用力 - 这里可以简化为单向耦合
                    # 如果需要双向，需要在这里 atomic_add 到 rb.vel 和 rb.ang_vel
                    if rb.is_fixed == True:
                        continue
                    impulse = -(1 + restitution) * v_n * self.mass
                    ti.atomic_add(rb.vel[None], -impulse * normal / rb.mass)


    def step(self, dt, rigid_bodies=None):
        substeps = 300
        dt /= substeps
        gravity = ti.Vector([0.0, 0.0, -9.8])
        
        for _ in range(substeps):
            self.compute_forces(gravity)
            self.update(dt)
            if rigid_bodies:
                # 假设只处理第一个刚体
                self.solve_rigid_collision(rigid_bodies[0])

    def get_indices(self):
        """生成三角形拓扑结构 (用于Blender渲染)"""
        indices = []
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                # 两个三角形组成一个网格四边形
                # Tri 1: (i, j), (i+1, j), (i, j+1)
                idx1 = i * self.N + j
                idx2 = (i + 1) * self.N + j
                idx3 = i * self.N + (j + 1)
                
                # Tri 2: (i+1, j), (i+1, j+1), (i, j+1)
                idx4 = (i + 1) * self.N + j
                idx5 = (i + 1) * self.N + (j + 1)
                idx6 = i * self.N + (j + 1)
                
                indices.extend([idx1, idx2, idx3, idx4, idx5, idx6])
        return np.array(indices, dtype=np.int32)