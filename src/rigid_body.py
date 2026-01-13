import taichi as ti
import trimesh
import numpy as np
import math
from mesh_to_sdf import mesh_to_voxels

@ti.data_oriented
class RigidBody:
    def __init__(self, pos, type: str, mass, mesh = None, color=(0.8, 0.8, 0.8, 1.0), radius=1.0,
                 velocity=np.zeros(3),
                 angular_velocity=np.zeros(3),
                 rotation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                 scale=(1.0, 1.0, 1.0),
                 is_fixed=False,
                 mass_distribution='uniform',
                 shape=[],
                 restitution=0.5):
        self.pos_of_center = ti.Vector.field(3, dtype=float, shape=()) 
        self.pos_of_center[None] = ti.Vector(pos)

        self.mass = mass
        if is_fixed:
            self.mass = 1e8

        self.vel = ti.Vector.field(3, dtype=float, shape=())
        self.vel[None] = ti.Vector(velocity)
        self.ang_vel = ti.Vector.field(3, dtype=float, shape=())
        self.ang_vel[None] = ti.Vector(angular_velocity)

        self.quat = ti.Vector.field(4, dtype=float, shape=())
        self.quat[None] = ti.Vector(rotation_quat)

        self.is_fixed = is_fixed

        self.type = type
        self.shape = shape
        self.restitution = restitution

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
        elif ti.static(self.type == 'box'):
            # --- 旋转盒子解析解 ---
            center = self.pos_of_center[None]
            R = self.quat_to_matrix(self.quat[None])
            # World to Local
            p_local = R.transpose() @ (world_pos - center)
            
            extents = ti.Vector([self.shape[0]*0.5, self.shape[1]*0.5, self.shape[2]*0.5])
            d_vec = ti.abs(p_local) - extents

            # SDF Distance
            inside_dist = ti.min(ti.max(d_vec.x, ti.max(d_vec.y, d_vec.z)), 0.0)
            outside_vec = ti.max(d_vec, 0.0)
            outside_dist = outside_vec.norm()
            dist = outside_dist + inside_dist
            
            # Normal calculation
            normal_local = ti.Vector([0.0, 0.0, 0.0])
            
            if dist > 0:
                # Outside the box
                if outside_dist > 1e-8:
                     normal_local = (outside_vec / outside_dist) * \
                        ti.Vector([1.0 if p_local.x >= 0 else -1.0, 
                                   1.0 if p_local.y >= 0 else -1.0, 
                                   1.0 if p_local.z >= 0 else -1.0])
                else:
                     normal_local = ti.Vector([0.0, 0.0, 1.0]) # Fallback
            else:
                # Inside the box
                if d_vec.x > d_vec.y and d_vec.x > d_vec.z:
                    normal_local = ti.Vector([1.0 if p_local.x > 0 else -1.0, 0.0, 0.0])
                elif d_vec.y > d_vec.z:
                    normal_local = ti.Vector([0.0, 1.0 if p_local.y > 0 else -1.0, 0.0])
                else:
                    normal_local = ti.Vector([0.0, 0.0, 1.0 if p_local.z > 0 else -1.0])
            
            normal = R @ normal_local
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
    def apply_force(self, force: ti.math.vec3, dt: float):
        # F = ma => a = F / m
        # v_new = v_old + a * dt
        if not self.is_fixed:
            acceleration = force / self.mass
            self.vel[None] += acceleration * dt
    @ti.kernel
    def apply_torque(self, torque: ti.math.vec3, dt: float):
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
def sphere_collision_simulation(rb1: ti.template(), rb2: ti.template(), threshold: float):
    p1, p2 = rb1.pos_of_center[None], rb2.pos_of_center[None]
    v1, v2 = rb1.vel[None], rb2.vel[None]

    restitution = min(rb1.restitution, rb2.restitution)

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
def sphere_box_collision_simulation(ball: ti.template(), box: ti.template(), threshold: float):
    ball_pos = ball.pos_of_center[None]
    box_pos = box.pos_of_center[None]
    
    R_box = box.quat_to_matrix(box.quat[None])
    R_inv_box = R_box.transpose()
    ball_pos_local = R_inv_box @ (ball_pos - box_pos)

    box_half_extents = ti.Vector([box.shape[0] * 0.5, box.shape[1] * 0.5, box.shape[2] * 0.5])
    closest_point_local = ti.Vector([
        ti.max(-box_half_extents.x, ti.min(ball_pos_local.x, box_half_extents.x)),
        ti.max(-box_half_extents.y, ti.min(ball_pos_local.y, box_half_extents.y)),
        ti.max(-box_half_extents.z, ti.min(ball_pos_local.z, box_half_extents.z))
    ])

    closest_point_world = R_box @ closest_point_local + box_pos
    
    restitution = min(ball.restitution, box.restitution)

    diff = ball_pos - closest_point_world
    dist = diff.norm()

    mb = ball.mass
    mbx = box.mass
    
    if dist < ball.radius + threshold:
        normal = diff.normalized()
        if dist < 1e-6:
            normal = R_box @ ti.Vector([0.0, 0.0, 1.0])

        r_box = closest_point_world - box.pos_of_center[None]
        v_rel = ball.vel[None] - (box.vel[None] + box.ang_vel[None].cross(r_box))
        
        vel_along_normal = v_rel.dot(normal)

        if vel_along_normal < 0:

            I_inv_box = box.I_inv[None]
            rot_inertia_box = (I_inv_box @ r_box.cross(normal)).cross(r_box).dot(normal)
            
            inv_mass_sum = 1 / mb + 1 / mbx + rot_inertia_box
            
            impulse_magnitude = -(1 + restitution) * vel_along_normal / inv_mass_sum
            impulse = impulse_magnitude * normal
            ball.vel[None] += impulse / mb
            box.vel[None] -= impulse / mbx

            box.ang_vel[None] -= I_inv_box @ r_box.cross(impulse) 

            
@ti.func
def check_vertex_penetration(p, box, threshold):
    # Transform point p to box's local space
    R_inv = box.quat_to_matrix(box.quat[None]).transpose()
    
    p_local = R_inv @ (p - box.pos_of_center[None])

        # Check for penetration along each axis of the box
    half_extents = ti.Vector([box.shape[0] * 0.5, box.shape[1] * 0.5, box.shape[2] * 0.5])
    penetration_depth = 0.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    is_penetrating = (ti.abs(p_local.x) < half_extents.x + threshold and
                          ti.abs(p_local.y) < half_extents.y + threshold and
                          ti.abs(p_local.z) < half_extents.z + threshold)

    if is_penetrating:
            # Find axis of minimum penetration
        min_penetration = 1e9
            
        for i in ti.static(range(3)):
            dist_to_face = half_extents[i] - ti.abs(p_local[i])
            if dist_to_face < min_penetration:
                min_penetration = dist_to_face
                # Normal points outwards from the box face
                axis = ti.Vector.zero(float, 3)
                axis[i] = 1.0 if p_local[i] > 0 else -1.0
                    # Transform normal to world space
                normal = box.quat_to_matrix(box.quat[None]) @ axis
            
        penetration_depth = min_penetration
            
    return penetration_depth, normal, is_penetrating

@ti.kernel
def box_collision_simulation(box1: ti.template(), box2: ti.template(), threshold: float):
    # Vertex-based penetration check
    pos1, pos2 = box1.pos_of_center[None], box2.pos_of_center[None]
    R1, R2 = box1.quat_to_matrix(box1.quat[None]), box2.quat_to_matrix(box2.quat[None])
    half1 = ti.Vector([box1.shape[0] * 0.5, box1.shape[1] * 0.5, box1.shape[2] * 0.5])
    half2 = ti.Vector([box2.shape[0] * 0.5, box2.shape[1] * 0.5, box2.shape[2] * 0.5])

    avg_normal = ti.Vector([0.0, 0.0, 0.0])
    avg_contact_point = ti.Vector([0.0, 0.0, 0.0])
    num_contacts = 0

    # Check vertices of box2 against box1
    for i in ti.static(range(8)):
        offset = ti.Vector([
            half2.x * (1 if (i & 1) else -1),
            half2.y * (1 if (i & 2) else -1),
            half2.z * (1 if (i & 4) else -1)
        ])
        vertex = pos2 + R2 @ offset
        depth, normal, is_penetrating = check_vertex_penetration(vertex, box1, threshold)
        if is_penetrating:
            avg_normal += normal
            avg_contact_point += vertex - normal * (depth * 0.5)
            num_contacts += 1

    # Check vertices of box1 against box2
    for i in ti.static(range(8)):
        offset = ti.Vector([
            half1.x * (1 if (i & 1) else -1),
            half1.y * (1 if (i & 2) else -1),
            half1.z * (1 if (i & 4) else -1)
        ])
        vertex = pos1 + R1 @ offset
        depth, normal, is_penetrating = check_vertex_penetration(vertex, box2, threshold)
        if is_penetrating:
            avg_normal -= normal # Normal should point from box1 to box2
            avg_contact_point += vertex - normal * (depth * 0.5)
            num_contacts += 1

    if num_contacts > 0:
        # Average the normals and contact points
        collision_normal = (avg_normal / num_contacts).normalized()
        contact_point = avg_contact_point / num_contacts

        # --- Penetration Resolution (simplified) ---
        # A more robust method would be needed for complex multi-contact scenarios
        
        # --- Collision Response ---
        r1 = contact_point - pos1
        r2 = contact_point - pos2
        
        v_rel = (box2.vel[None] + box2.ang_vel[None].cross(r2)) - \
                (box1.vel[None] + box1.ang_vel[None].cross(r1))
        
        vel_along_normal = v_rel.dot(collision_normal)

        if vel_along_normal < 0:
            restitution = min(box1.restitution, box2.restitution)
            m1, m2 = box1.mass, box2.mass
            I_inv1, I_inv2 = box1.I_inv[None], box2.I_inv[None]

            rot_inertia1 = (I_inv1 @ r1.cross(collision_normal)).cross(r1).dot(collision_normal)
            rot_inertia2 = (I_inv2 @ r2.cross(collision_normal)).cross(r2).dot(collision_normal)
            inv_mass_sum = 1/m1 + 1/m2 + rot_inertia1 + rot_inertia2

            if inv_mass_sum > 0:
                impulse_magnitude = -(1 + restitution) * vel_along_normal / inv_mass_sum
                impulse = impulse_magnitude * collision_normal

                if not box1.is_fixed:
                    box1.vel[None] -= impulse / m1
                    box1.ang_vel[None] -= I_inv1 @ r1.cross(impulse)
                if not box2.is_fixed:
                    box2.vel[None] += impulse / m2
                    box2.ang_vel[None] += I_inv2 @ r2.cross(impulse)

    
