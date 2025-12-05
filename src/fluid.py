# import taichi as ti
# import numpy as np
# import trimesh
# import math

# @ti.data_oriented
# class Fluid:
#     def __init__(self, max_particles, position=[0.0, 0.0, 0.0], init_box=(10.0, 10.0, 5.0),
#                  viscosity=0.05, rest_density=1000.0, gamma=7.0, sound_speed=25.0, particle_radius=0.1,
#                  stiffness=50000, surface_tension=0.01):
#         self.max_particles = max_particles
#         self.num_particles = ti.field(dtype=int, shape=())

#         self.pos = ti.Vector.field(3, dtype=float, shape=max_particles)
#         self.vel = ti.Vector.field(3, dtype=float, shape=max_particles)
#         self.acc = ti.Vector.field(3, dtype=float, shape=max_particles)
#         self.density = ti.field(dtype=float, shape=max_particles)
#         self.pressure = ti.field(dtype=float, shape=max_particles)
#         self.mass = ti.field(dtype=float, shape=())

#         self.particle_radius = particle_radius
#         self.h = 4 * particle_radius
                 
#         self.rest_density = rest_density

#         self.particle_volume = (self.particle_radius*2)**3
#         self.mass = self.rest_density * self.particle_volume


#         self.viscosity = viscosity
#         self.gravity = ti.Vector([0.0, 0.0, -9.8])

#         self.init_box = init_box
#         self.position = position
#         self.stiffness = stiffness
#         self.surface_tension = surface_tension

#         self.gamma = gamma            
#         self.sound_speed = sound_speed     

#         self.B = self.rest_density * (self.sound_speed**2) / self.gamma

    
#     def add_particles(self, positions):
#         n = len(positions)
#         assert self.num_particles[None] + n <= self.max_particles, "Exceeding maximum number of particles"
#         for i in range(n):
#             idx = self.num_particles[None]
#             self.pos[idx] = ti.Vector(positions[i])
#             self.vel[idx] = ti.Vector([0.0, 0.0, 0.0])
#             self.density[idx] = self.rest_density
#             self.pressure[idx] = 0.0
#             self.num_particles[None] += 1
    
#     def init_cube(self, spacing=None):
#         if spacing is None:
#             spacing = self.particle_radius * 2.0
#         new_positions = []
#         start_pos = np.array(self.position) + spacing * 0.5
#         nx = int(self.init_box[0] / spacing)
#         ny = int(self.init_box[1] / spacing)
#         nz = int(self.init_box[2] / spacing)
#         for i in range(nx):
#             for j in range(ny):
#                 for k in range(nz):
#                     pos = start_pos + np.array([i * spacing, j * spacing, k * spacing])
#                     new_positions.append(pos)
#         self.add_particles(new_positions)

    
#     @ti.func
#     def cubic_kernel(self, r_norm):
#         res = 0.0
#         h = self.h
#         k = 8 / (math.pi * h**3)
#         q = r_norm / h
#         if q <= 1.0:
#             if q <= 0.5: res = k * (6 * (q**3) - 6 * (q**2) + 1)
#             else: res = k * 2 * ((1 - q)**3)
#         return res
    
#     @ti.func
#     def cubic_kernel_gradient(self, r_vec):
#         h = self.h
#         k = 48 / (math.pi * h**3)
#         r = r_vec.norm()
#         q = r / h
#         res = ti.Vector([0.0, 0.0, 0.0])
#         if r > 1e-5:
#             if q <= 1.0:
#                 grad_q = r_vec / (r * h)
#                 if q <= 0.5:
#                     res = k * (3 * (q**2) - 2 * q) * grad_q
#                 else:
#                     res = -k * ((1 - q)**2) * grad_q
#         return res
    
#     @ti.kernel
#     def compute_density(self):
#         for i in range(self.num_particles[None]):
#             self.density[i] = 0.0
#             for j in range(self.num_particles[None]):
#                 r_ij = self.pos[i] - self.pos[j]
#                 r_norm = r_ij.norm()
#                 if r_norm < self.h:
#                     self.density[i] += self.mass * self.cubic_kernel(r_norm)
#             # self.density[i] = ti.max(self.density[i], self.rest_density)

#     @ti.kernel
#     def compute_forces(self):
#         for i in range(self.num_particles[None]):
#             ratio = self.density[i] / self.rest_density
#             self.pressure[i] = ti.max(self.B * (ratio**self.gamma - 1), 0.0)
        
#         for i in range(self.num_particles[None]):
#             pressure_force = ti.Vector([0.0, 0.0, 0.0])
#             viscosity_force = ti.Vector([0.0, 0.0, 0.0])
#             for j in range(self.num_particles[None]):
#                 if i != j:
#                     r_ij = self.pos[i] - self.pos[j]
#                     r_norm = r_ij.norm()
#                     if r_norm < self.h:
#                         # Pressure force
#                         pressure_term = (self.pressure[i] / (self.density[i]**2) +
#                                          self.pressure[j] / (self.density[j]**2))
#                         pressure_force += -self.mass * pressure_term * self.cubic_kernel_gradient(r_ij)
                        
#                         # Viscosity force
#                         vel_diff = self.vel[j] - self.vel[i]
#                         W = self.cubic_kernel(r_norm)
#                         viscosity_force += self.viscosity * self.mass * (vel_diff / self.density[j]) * W / self.h
            
#             gravity_force = self.gravity
#             self.acc[i] = pressure_force + viscosity_force + gravity_force
    
#     @ti.kernel
#     def integrate(self, dt: float):
#         for i in range(self.num_particles[None]):
#             self.vel[i] += self.acc[i] * dt
#             self.pos[i] += self.vel[i] * dt
    
    
#     def update(self, dt, container: ti.template()):
#         substeps = 20
#         dt /= substeps
#         for _ in range(substeps):
#             self.compute_density()
#             self.compute_forces()
#             self.integrate(dt)
#             if container is not None:
#                 container.enforce_boundary(self)
    
#     def save_ply(self, filename):
#         num_p = self.num_particles[None]
#         pos_np = self.pos.to_numpy()[:num_p]
#         pos_np = pos_np.astype(np.float32)
#         header = f'''ply
# format binary_little_endian 1.0
# element vertex {num_p}
# property float x
# property float y
# property float z
# end_header
# '''
#         with open(filename, 'wb') as f:
#             f.write(header.encode('ascii'))
#             f.write(pos_np.tobytes())

# @ti.data_oriented
# class Container:
#     def __init__(self, position, boundary_box):
#         self.position = ti.Vector(position)
#         self.boundary_box = ti.Vector(boundary_box)
#         self.thickness = 0.01
#         self.mesh = self.mesh_generate()
    
#     def mesh_generate(self):
#         sx, sy, sz = self.boundary_box
#         tx, ty, tz = self.position
#         t = self.thickness

#         cx = tx + sx / 2
#         cy = ty + sy / 2
#         cz = tz + sz / 2
#         meshes = []

#         bottom = trimesh.creation.box(extents=[sx, sy, t], transform=trimesh.transformations.translation_matrix([cx, cy, tz - t / 2]))
#         left = trimesh.creation.box(extents=[t, sy, sz], transform=trimesh.transformations.translation_matrix([tx - t / 2, cy, tz + sz / 2]))
#         right = trimesh.creation.box(extents=[t, sy, sz], transform=trimesh.transformations.translation_matrix([tx + sx + t / 2, cy, tz + sz / 2]))
#         front = trimesh.creation.box(extents=[sx + 2 * t, t, sz], transform=trimesh.transformations.translation_matrix([cx, ty - t / 2, tz + sz / 2]))
#         back = trimesh.creation.box(extents=[sx + 2 * t, t, sz], transform=trimesh.transformations.translation_matrix([cx, ty + sy + t / 2, tz + sz / 2]))
#         meshes.extend([bottom, left, right, front, back])

#         container_mesh = trimesh.util.concatenate(meshes)
#         return container_mesh

#     @ti.kernel
#     def enforce_boundary(self, fluid: ti.template()):
#         particle_radius = fluid.particle_radius
#         restitution = 0.5
#         for i in range(fluid.num_particles[None]):
#             pos = fluid.pos[i]
#             vel = fluid.vel[i]
#             for d in ti.static(range(3)):
#                 if fluid.pos[i][d] < self.position[d] + self.thickness:
#                     fluid.pos[i][d] = self.position[d] + self.thickness
#                     fluid.vel[i][d] *= -0.1  # simple bounce with damping
#                 if fluid.pos[i][d] > self.position[d] + self.boundary_box[d] - self.thickness:
#                     fluid.pos[i][d] = self.position[d] + self.boundary_box[d] - self.thickness
#                     fluid.vel[i][d] *= -0.1  # simple bounce with damping

import taichi as ti
import numpy as np
import trimesh
import math

@ti.data_oriented
class Fluid:
    def __init__(self, max_particles, position=[0.0, 0.0, 0.0], init_box=(10.0, 10.0, 5.0),
                 viscosity=10.0, rest_density=1000.0, gamma=7.0, sound_speed=20.0, particle_radius=0.01):
        
        self.max_particles = max_particles
        self.num_particles = ti.field(dtype=int, shape=())

        # 粒子属性
        self.pos = ti.Vector.field(3, dtype=float, shape=max_particles)
        self.vel = ti.Vector.field(3, dtype=float, shape=max_particles)

        self.forces = ti.Vector.field(3, dtype=float, shape=max_particles) 
        self.density = ti.field(dtype=float, shape=max_particles)
        self.pressure = ti.field(dtype=float, shape=max_particles)
        self.mass = ti.field(dtype=float, shape=max_particles)


        self.particle_radius = particle_radius
        self.particle_diameter = 2 * self.particle_radius
        self.h = 4.0 * self.particle_radius 
        
        self.rest_density = rest_density
        self.viscosity = viscosity
        self.gravity = ti.Vector([0.0, 0.0, -9.8])
        

        self.gamma = gamma
        self.stiffness = 50000.0 
        self.surface_tension = 0.01 


        self.grid_origin = ti.Vector([-2.5, -2.5, -0.5])
        self.grid_cell_size = self.h
        self.grid_dim = (64, 64, 64)

        self.max_particles_per_cell = 100
        self.grid_num = ti.field(dtype=int, shape=self.grid_dim)
        self.grid = ti.field(dtype=int, shape=(*self.grid_dim, self.max_particles_per_cell))
        
        # Not use this
        self.sound_speed = sound_speed

        self.init_box = init_box
        self.position = position

    def add_particles(self, positions):
        n = len(positions)
        assert self.num_particles[None] + n <= self.max_particles, "Exceeding maximum number of particles"
        
        particle_vol = (self.particle_diameter) ** 3
        mass_per_particle = self.rest_density * particle_vol

        for i in range(n):
            idx = self.num_particles[None]
            self.pos[idx] = ti.Vector(positions[i])
            self.vel[idx] = ti.Vector([0.0, 0.0, 0.0])
            self.density[idx] = self.rest_density
            self.pressure[idx] = 0.0
            self.mass[idx] = mass_per_particle
            self.num_particles[None] += 1
    
    def init_cube(self, spacing=None):
        if spacing is None:
            spacing = self.particle_radius * 2.0
        new_positions = []
        start_pos = np.array(self.position) + spacing * 0.5
        nx = int(self.init_box[0] / spacing)
        ny = int(self.init_box[1] / spacing)
        nz = int(self.init_box[2] / spacing)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = start_pos + np.array([i * spacing, j * spacing, k * spacing])
                    new_positions.append(pos)
        self.add_particles(new_positions)

    @ti.func
    def kernel_func(self, R_mod):
        # cubic kernel
        res = ti.cast(0.0, ti.f32)
        h = self.h
        k = 8 / np.pi
        k /= h ** 3
        q = R_mod / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def kernel_grad(self, R):
        # cubic kernel gradient
        res = ti.Vector([0.0, 0.0, 0.0])
        h = self.h
        k = 8 / np.pi
        k = 6. * k / h ** 3
        R_mod = R.norm()
        q = R_mod / h
        if R_mod > 1e-5 and q <= 1.0:
            grad_q = R / (R_mod * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res
    
    @ti.kernel
    def update_grid(self):
        for I in ti.grouped(self.grid_num):
            self.grid_num[I] = 0
        
        for i in range(self.num_particles[None]):
            cell_idx = ((self.pos[i] - self.grid_origin) / self.grid_cell_size).cast(int)
            if 0 <= cell_idx[0] < self.grid_dim[0] and 0 <= cell_idx[1] < self.grid_dim[1] and 0 <= cell_idx[2] < self.grid_dim[2]:
                idx = ti.atomic_add(self.grid_num[cell_idx], 1)
                if idx < self.max_particles_per_cell:
                    self.grid[cell_idx, idx] = i
    
    @ti.kernel
    def compute_density(self):
        for i in range(self.num_particles[None]):
            self.density[i] = 0.0
            pos_i = self.pos[i]
            cell_idx = ti.cast((pos_i - self.grid_origin) / self.grid_cell_size, int)
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                neighbor_cell = cell_idx + offset
                if 0 <= neighbor_cell[0] < self.grid_dim[0] and 0 <= neighbor_cell[1] < self.grid_dim[1] and 0 <= neighbor_cell[2] < self.grid_dim[2]:
                    num_in_cell = self.grid_num[neighbor_cell]
                    for j_idx in range(num_in_cell):
                        j = self.grid[neighbor_cell, j_idx]
                        r = pos_i - self.pos[j]
                        r_len = r.norm()
                        if r_len < self.h:
                            self.density[i] += self.kernel_func(r_len) * self.mass[j]
            self.density[i] = ti.max(self.density[i], self.rest_density)
            # for j in range(self.num_particles[None]):
            #     r = self.pos[i] - self.pos[j]
            #     r_len = r.norm()
            #     if r_len < self.h:
            #         self.density[i] += self.kernel_func(r_len) * self.mass[j]
            # self.density[i] = ti.max(self.density[i], self.rest_density)

    @ti.kernel
    def compute_forces(self):
        for i in range(self.num_particles[None]):
            ratio = self.density[i] / self.rest_density
            self.pressure[i] = ti.max(0.0, self.stiffness * (ratio**self.gamma - 1))

        for i in range(self.num_particles[None]):
            self.forces[i] = self.gravity * self.mass[i]

            pos_i = self.pos[i]
            cell_idx = ti.cast((pos_i - self.grid_origin) / self.grid_cell_size, int)
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                neighbor_cell = cell_idx + offset
                if 0 <= neighbor_cell[0] < self.grid_dim[0] and 0 <= neighbor_cell[1] < self.grid_dim[1] and 0 <= neighbor_cell[2] < self.grid_dim[2]:
                    num_in_cell = self.grid_num[neighbor_cell]
                    for j_idx in range(num_in_cell):
                        j = self.grid[neighbor_cell, j_idx]
                        if i != j:
                            r = self.pos[i] - self.pos[j]
                            r_len = r.norm()
                            if r_len < self.h:
                                nabla_ij = self.kernel_grad(r)

                                # Pressure Force
                                pressure_term = (self.pressure[i] / (self.density[i]**2) + 
                                                 self.pressure[j] / (self.density[j]**2))
                                
                                pressure_force = -self.mass[j] * pressure_term * nabla_ij
                                
                                # Viscosity Force
                                v_xy = (self.vel[i] - self.vel[j]).dot(r)
                                m_ij = (self.mass[i] + self.mass[j]) / 2.0
                                
                                viscosity_force = (2 * 5 * self.viscosity * m_ij / self.density[j] / 
                                                   (r_len**2 + 0.01 * self.h**2) * v_xy * nabla_ij / self.rest_density)

                                # Surface Tension
                                surface_tension_force = ti.Vector([0.0, 0.0, 0.0])
                                if r_len > self.particle_diameter:
                                    surface_tension_force = -self.surface_tension / self.density[i] * self.density[j] * r * self.kernel_func(r_len)
                                else:
                                    surface_tension_force = -self.surface_tension / self.density[i] * self.density[j] * r * self.kernel_func(self.particle_diameter)
                                
                                self.forces[i] += pressure_force + viscosity_force + surface_tension_force
            
            # for j in range(self.num_particles[None]):
            #     if i != j:
            #         r = self.pos[i] - self.pos[j]
            #         r_len = r.norm()
                    
            #         if r_len < self.h:
            #             nabla_ij = self.kernel_grad(r)

            #             # Pressure Force
            #             pressure_term = (self.pressure[i] / (self.density[i]**2) + 
            #                              self.pressure[j] / (self.density[j]**2))
                        
            #             pressure_force = -self.mass[j] * pressure_term * nabla_ij
                        
            #             # Viscosity Force
            #             v_xy = (self.vel[i] - self.vel[j]).dot(r)
            #             m_ij = (self.mass[i] + self.mass[j]) / 2.0
                        
            #             viscosity_force = (2 * 5 * self.viscosity * m_ij / self.density[j] / 
            #                                (r_len**2 + 0.01 * self.h**2) * v_xy * nabla_ij / self.rest_density)

            #             # Surface Tension
            #             surface_tension_force = ti.Vector([0.0, 0.0, 0.0])
            #             if r_len > self.particle_diameter:
            #                 surface_tension_force = -self.surface_tension / self.density[i] * self.density[j] * r * self.kernel_func(r_len)
            #             else:
            #                 surface_tension_force = -self.surface_tension / self.density[i] * self.density[j] * r * self.kernel_func(self.particle_diameter)
                        
            #             self.forces[i] += pressure_force + viscosity_force + surface_tension_force
    
    @ti.kernel
    def integrate(self, dt: float):
        for i in range(self.num_particles[None]):

            self.vel[i] += (self.forces[i] * dt) / self.mass[i]
            self.pos[i] += self.vel[i] * dt
    
    def update(self, dt, container: ti.template()):
        substeps = 100
        dt /= substeps
        for _ in range(substeps):
            self.update_grid()
            self.compute_density()
            self.compute_forces()
            self.integrate(dt)
            if container is not None:
                container.enforce_boundary(self)
    
    def save_ply(self, filename):
        num_p = self.num_particles[None]
        pos_np = self.pos.to_numpy()[:num_p]
        pos_np = pos_np.astype(np.float32)
        header = f'''ply
format binary_little_endian 1.0
element vertex {num_p}
property float x
property float y
property float z
end_header
'''
        with open(filename, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(pos_np.tobytes())

@ti.data_oriented
class Container:
    def __init__(self, position, boundary_box):
        self.position = ti.Vector(position)
        self.boundary_box = ti.Vector(boundary_box)
        self.thickness = 0.01
        self.mesh = self.mesh_generate()
    
    def mesh_generate(self):
        sx, sy, sz = self.boundary_box
        tx, ty, tz = self.position
        t = self.thickness

        cx = tx + sx / 2
        cy = ty + sy / 2
        cz = tz + sz / 2
        meshes = []

        bottom = trimesh.creation.box(extents=[sx, sy, t], transform=trimesh.transformations.translation_matrix([cx, cy, tz - t / 2]))
        left = trimesh.creation.box(extents=[t, sy, sz], transform=trimesh.transformations.translation_matrix([tx - t / 2, cy, tz + sz / 2]))
        right = trimesh.creation.box(extents=[t, sy, sz], transform=trimesh.transformations.translation_matrix([tx + sx + t / 2, cy, tz + sz / 2]))
        front = trimesh.creation.box(extents=[sx + 2 * t, t, sz], transform=trimesh.transformations.translation_matrix([cx, ty - t / 2, tz + sz / 2]))
        back = trimesh.creation.box(extents=[sx + 2 * t, t, sz], transform=trimesh.transformations.translation_matrix([cx, ty + sy + t / 2, tz + sz / 2]))
        meshes.extend([bottom, left, right, front, back])

        container_mesh = trimesh.util.concatenate(meshes)
        return container_mesh

    @ti.kernel
    def enforce_boundary(self, fluid: ti.template()):
        # 简单的边界反弹
        damping = 0.23
        for i in range(fluid.num_particles[None]):
            for d in ti.static(range(3)):
                # 下边界/左边界/后边界
                if fluid.pos[i][d] < self.position[d] + self.thickness:
                    fluid.pos[i][d] = self.position[d] + self.thickness + 1e-4
                    fluid.vel[i][d] *= -damping
                
                # 上边界/右边界/前边界
                if fluid.pos[i][d] > self.position[d] + self.boundary_box[d] - self.thickness:
                    fluid.pos[i][d] = self.position[d] + self.boundary_box[d] - self.thickness - 1e-4
                    fluid.vel[i][d] *= -damping
