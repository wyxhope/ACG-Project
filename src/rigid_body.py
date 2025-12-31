import taichi as ti
import trimesh
import numpy as np
import math
from mesh_to_sdf import mesh_to_voxels

@ti.data_oriented
class RigidBody:
    def __init__(self, pos, type: str, mass, mesh, color=(0.8, 0.8, 0.8, 1.0), radius=1.0,
                 velocity=np.zeros(3),
                 angular_velocity=np.zeros(3),
                 rotation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                 scale=(1.0, 1.0, 1.0),
                 is_fixed=False,
                 mass_distribution='uniform',
                 shape=[]):
        self.pos_of_center = ti.Vector.field(3, dtype=float, shape=()) 
        self.pos_of_center[None] = ti.Vector(pos)

        self.mass = mass

        self.vel = ti.Vector.field(3, dtype=float, shape=())
        self.vel[None] = ti.Vector(velocity)
        self.ang_vel = ti.Vector.field(3, dtype=float, shape=())
        self.ang_vel[None] = ti.Vector(angular_velocity)

        self.quat = ti.Vector.field(4, dtype=float, shape=())
        self.quat[None] = ti.Vector(rotation_quat)

        self.is_fixed = is_fixed

        self.type = type

        if type == 'sphere':
            self.mesh = trimesh.creation.icosphere(subdivisions=5, radius=radius)
        elif type == 'box':
            self.mesh = trimesh.creation.box(extents=(shape[0], shape[1], shape[2]))
        else:
            if mesh is not None:
                self.mesh = mesh
        
        if self.mesh is not None:
            self.mesh.apply_scale(scale)
        
        # Get the center of mass and inertia tensor relative to pos
        self.mesh.density = self.mass / self.mesh.volume
        inertia_tensor = self.mesh.moment_inertia
        center_of_mass_offset = self.mesh.center_mass

        self.mesh.vertices -= center_of_mass_offset  # Center the mesh at COM

        self.I_inv = ti.Matrix.field(3, 3, dtype=float, shape=())
        self.I_inv[None] = ti.Matrix(np.linalg.inv(inertia_tensor))

        
        # Move mesh from cpu to ti to get acceleration
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.astype(np.int32)
        self.num_vertices = vertices.shape[0]
        self.num_faces = faces.shape[0]
        self.vertices = ti.Vector.field(3, dtype=float, shape=self.num_vertices)
        self.faces = ti.Vector.field(3, dtype=int, shape=self.num_faces)
        self.vertices.from_numpy(vertices)
        self.faces.from_numpy(faces)
        self.radius = radius

        self.sdf_res = 64


        import copy
        mesh_copy = self.mesh.copy()

        bounds = mesh_copy.bounds
        bbox_min, bbox_max = bounds[0], bounds[1]
        bbox_center = (bbox_min + bbox_max) / 2

        original_extents = bbox_max - bbox_min
        max_extent = np.max(original_extents)

        voxels = mesh_to_voxels(mesh_copy, voxel_resolution=self.sdf_res, pad=False)
        self.sdf = ti.field(dtype=float, shape=(self.sdf_res, self.sdf_res, self.sdf_res))
        self.sdf.from_numpy(voxels)

        self.half_size = max_extent / 2 

        self.sdf_offset = ti.Vector(bbox_center)


    @ti.func
    def get_sdf(self, world_pos):
        dist = 0.0
        normal = ti.Vector([0.0, 0.0, 0.0])

        # 使用 ti.static 进行编译时分支检查，优化性能
        if ti.static(self.type == 'sphere'):
            # --- 球体解析解析解 ---
            # 直接使用几何公式：dist = ||p - center|| - radius
            p_rel = world_pos - self.pos_of_center[None]
            d_norm = p_rel.norm()
            dist = d_norm - self.radius
            
            # 法线从球心指向外部
            if d_norm > 1e-6:
                normal = p_rel / d_norm
            else:
                normal = ti.Vector([0.0, 0.0, 1.0])
        else:
            # --- 原有的通用网格 SDF 查询逻辑 ---
            center = self.pos_of_center[None]
            R = self.quat_to_matrix(self.quat[None])
            local_pos = R.transpose() @ (world_pos - center)
            local_pos_sdf = local_pos - self.sdf_offset

            normalized_pos = local_pos_sdf / (self.half_size * 2) + 0.5
            uvw = normalized_pos * self.sdf_res

            dist = 1000.0
            if (uvw.x >= 0 and uvw.x < self.sdf_res - 1 and \
                uvw.y >= 0 and uvw.y < self.sdf_res - 1 and \
                uvw.z >= 0 and uvw.z < self.sdf_res - 1):
                base = ti.cast(ti.floor(uvw), ti.i32)
                frac = uvw - base

                # 三线性插值计算距离
                c000 = self.sdf[base]
                c100 = self.sdf[base + ti.Vector([1, 0, 0])]
                c010 = self.sdf[base + ti.Vector([0, 1, 0])]
                c110 = self.sdf[base + ti.Vector([1, 1, 0])]
                c001 = self.sdf[base + ti.Vector([0, 0, 1])]
                c101 = self.sdf[base + ti.Vector([1, 0, 1])]
                c011 = self.sdf[base + ti.Vector([0, 1, 1])]
                c111 = self.sdf[base + ti.Vector([1, 1, 1])]

                lerp_x_00 = c000 * (1 - frac.x) + c100 * frac.x
                lerp_x_10 = c010 * (1 - frac.x) + c110 * frac.x
                lerp_x_01 = c001 * (1 - frac.x) + c101 * frac.x
                lerp_x_11 = c011 * (1 - frac.x) + c111 * frac.x
                lerp_y_0 = lerp_x_00 * (1 - frac.y) + lerp_x_10 * frac.y
                lerp_y_1 = lerp_x_01 * (1 - frac.y) + lerp_x_11 * frac.y
                dist = lerp_y_0 * (1 - frac.z) + lerp_y_1 * frac.z
                dist = dist * self.half_size * 2

                # 中心差分计算法线
                dx = (self.sdf[base + ti.Vector([1, 0, 0])] - self.sdf[base + ti.Vector([-1, 0, 0])]) 
                dy = (self.sdf[base + ti.Vector([0, 1, 0])] - self.sdf[base + ti.Vector([0, -1, 0])]) 
                dz = (self.sdf[base + ti.Vector([0, 0, 1])] - self.sdf[base + ti.Vector([0, -1, 1])]) 

                local_normal = ti.Vector([dx, dy, dz])
                if local_normal.norm() > 1e-8:
                    local_normal = local_normal.normalized()
                normal = R @ local_normal
        
        return dist, normal


    @ti.func
    def quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return ti.Vector([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    @ti.func
    def quat_to_matrix(self, q):
        w, x, y, z = q
        return ti.Matrix([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
    
    @ti.func 
    def local_to_world(self, i: int):
        R = self.quat_to_matrix(self.quat[None])
        return R @ self.vertices[i] + self.pos_of_center[None]
    @ti.func
    def is_in_triangle(self, p, a, b, c, normal):
        # p is a point in the plane of triangle abc, then we can use this
        ab = b - a
        bc = c - b
        ca = a - c
        ap = p - a
        bp = p - b
        cp = p - c

        return (ab.cross(ap).dot(normal) >= 0 and
                bc.cross(bp).dot(normal) >= 0 and
                ca.cross(cp).dot(normal) >= 0)


    @ti.func
    def check_mesh_collision(self, point, threshold: float):
        min_dist = 1e8
        closest_normal = ti.Vector([0.0, 0.0, 0.0])
        has_collision = False

        for f in range(self.num_faces):
            idx0, idx1, idx2 = self.faces[f][0], self.faces[f][1], self.faces[f][2]
            v0 = self.local_to_world(idx0)
            v1 = self.local_to_world(idx1)
            v2 = self.local_to_world(idx2)

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = edge1.cross(edge2).normalized()

            to_point = point - v0
            distance = to_point.dot(normal)

            if ti.abs(distance) < threshold:
                proj_point = point - distance * normal
                if self.is_in_triangle(proj_point, v0, v1, v2, normal):
                    if ti.abs(distance) < min_dist:
                        min_dist = ti.abs(distance)
                        closest_normal = normal
                        has_collision = True
        return has_collision, closest_normal
    
    @ti.kernel
    def apply_force(self, force: ti.types.vector(3, float), dt: float):
        # F = ma => a = F / m
        # v_new = v_old + a * dt
        acceleration = force / self.mass
        self.vel[None] += acceleration * dt
    @ti.kernel
    def apply_torque(self, torque: ti.types.vector(3, float), dt: float):
        # tau = I * alpha => alpha = I_inv * tau
        # ang_v_new = ang_v_old + alpha * dt
        R = self.quat_to_matrix(self.quat[None])
        R_inv = R.transpose()
    
        local_torque = R_inv @ torque
        
        local_alpha = self.I_inv[None] @ local_torque

        world_alpha = R @ local_alpha
        
        self.ang_vel[None] += world_alpha * dt

    @ti.kernel
    def update(self, dt: float):
        # Update position
        if not self.is_fixed:
            self.pos_of_center[None] += self.vel[None] * dt

            # Update rotation
            omega = self.ang_vel[None]
            omega_mag = omega.norm()
            if omega_mag > 1e-8:
                theta = omega_mag * dt
                axis = omega / omega_mag
                half_theta = theta * 0.5
                sin_half_theta = ti.sin(half_theta)
                delta_quat = ti.Vector([
                    ti.cos(half_theta),
                    axis[0] * sin_half_theta,
                    axis[1] * sin_half_theta,
                    axis[2] * sin_half_theta
                ])
                self.quat[None] = self.quat_mul(delta_quat, self.quat[None])
                # Normalize quaternion
                q = self.quat[None]
                norm_q = ti.sqrt(q.dot(q))
                self.quat[None] = q / norm_q

@ti.kernel
def sphere_collision_simulation(rb1: ti.template(), rb2: ti.template(), threshold: float, restitution: float):
    p1, p2 = rb1.pos_of_center[None], rb2.pos_of_center[None]
    v1, v2 = rb1.vel[None], rb2.vel[None]

    m1, m2 = rb1.mass, rb2.mass
    r1, r2 = rb1.radius, rb2.radius

    diff = p2 - p1
    dist = diff.norm()
    if dist < r1 + r2 + threshold:
        normal = diff.normalized()
        relative_velocity = v2 - v1
        vel_along_normal = relative_velocity.dot(normal)
        if vel_along_normal < 0:
            impulse_magnitude = -(1 + restitution) * vel_along_normal
            impulse_magnitude /= (1 / m1 + 1 / m2)

            impulse = impulse_magnitude * normal

            rb1.vel[None] -= impulse / m1
            rb2.vel[None] += impulse / m2

@ti.kernel
def table_constrain_function(ball: ti.template(), table_pos: ti.types.vector(3, float), table_half_extents: ti.types.vector(3, float), ball_radius: float, elasticity: float):
    pos = ball.pos_of_center[None]
    vel = ball.vel[None]

    # 1. 计算桌面的边界范围
    x_min, x_max = table_pos.x - table_half_extents.x, table_pos.x + table_half_extents.x
    y_min, y_max = table_pos.y - table_half_extents.y, table_pos.y + table_half_extents.y
    top_surface_z = table_pos.z + table_half_extents.z

    # 2. 检查球是否在桌子的水平投影范围内
    is_over_table = (pos.x >= x_min and pos.x <= x_max and 
                     pos.y >= y_min and pos.y <= y_max)

    # 3. 如果在桌子上，且球的底部触碰到或穿透了桌面
    if is_over_table and (pos.z - ball_radius < top_surface_z) and (pos.z - ball_radius > top_surface_z - 0.5):
        # 位置修正：强制让球停在表面
        ball.pos_of_center[None].z = top_surface_z + ball_radius
        
        # 速度修正：如果球正在向下运动，消除垂直速度
        if vel.z < 0:
            ball.vel[None].z = - vel.z * elasticity  # 简单反弹，并损失部分能量
            # 可选：添加一点表面摩擦
            # ball.vel[None].x *= 0.95
            # ball.vel[None].y *= 0.95
@ti.kernel
def wall_constrain_function(ball: ti.template(), table_pos: ti.types.vector(3, float), table_half_extents: ti.types.vector(3, float), ball_radius: float, elasticity: float):
    pos = ball.pos_of_center[None]
    vel = ball.vel[None]

    # 1. 计算桌面的边界范围
    x_min, x_max = table_pos.x - table_half_extents.x, table_pos.x + table_half_extents.x
    y_min, y_max = table_pos.y - table_half_extents.y, table_pos.y + table_half_extents.y
    z_min, z_max = table_pos.z - table_half_extents.z, table_pos.z + table_half_extents.z

    # 2. 检查球是否在桌子的水平投影范围内
    is_over_table = (pos.y >= y_min and pos.y <= y_max and
                     pos.z >= z_min and pos.z <= z_max)

    # 3. 如果在桌子上，且球的底部触碰到或穿透了桌面
    if is_over_table and (pos.x + ball_radius > x_min):
        # 位置修正：强制让球停在表面
        ball.pos_of_center[None].x = x_min - ball_radius
        
        # 速度修正：如果球正在向下运动，消除垂直速度
        if vel.x > 0:
            ball.vel[None].x = - vel.x * elasticity  # 简单反弹，并损失部分能量
            # 可选：添加一点表面摩擦
            # ball.vel[None].x *= 0.95
            # ball.vel[None].y *= 0.95