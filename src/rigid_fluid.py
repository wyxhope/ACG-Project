import taichi as ti
import trimesh
from .rigid_body import *
from .fluid import Fluid, Container

@ti.data_oriented
class FluidSimulator:
    def __init__(self, fluid: Fluid, container: Container):
        self.fluid = fluid
        self.container = container
    
    def step(self, dt):
        self.fluid.update(dt, self.container)