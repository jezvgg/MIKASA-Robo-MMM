"""Deterministic salad visuals in the bowl's local frame; no extra physics state."""
from pathlib import Path
import math
import numpy as np
import sapien
from transforms3d.euler import euler2quat

ASSETS = Path(__file__).with_name('assets') / 'season_salad'


def decorate_bowl(builder):
    rng = np.random.default_rng(4816)  # Independent of task answers and episode RNG.
    leaf = str(ASSETS / 'leaf.obj')
    tomato = str(ASSETS / 'tomato_wedge.obj')
    for i in range(22):
        angle = i * math.pi * (3-math.sqrt(5))
        radius = .082 * math.sqrt((i+.5)/22)
        pose = sapien.Pose([radius*math.cos(angle),radius*math.sin(angle),.007+rng.uniform(-.004,.006)],
                            euler2quat(rng.uniform(-.3,.3),rng.uniform(-.3,.3),angle))
        builder.add_visual_from_file(leaf,pose=pose,scale=[.029,.029,.029],
            material=[.12+rng.uniform(0,.08),.35+rng.uniform(0,.16),.035,1.],name=f'salad_leaf_{i}')
    for i,(x,y,angle) in enumerate([(-.05,-.035,.4),(.04,-.045,2.2),(.015,.048,4.1),(-.025,.015,1.5)]):
        pose=sapien.Pose([x,y,.022],euler2quat(.2,-.2,angle))
        builder.add_visual_from_file(tomato,pose=pose,scale=[.021,.021,.018],
            material=[.78,.055,.025,1.],name=f'salad_tomato_{i}')
        for j in range(3):
            seed=pose*sapien.Pose([.001,.004+j*.004,.008-j*.003])
            builder.add_sphere_visual(pose=seed,radius=.0013,material=[.92,.72,.30,1.])
    for i,(x,y,angle) in enumerate([(-.055,.04,.2),(.062,.012,-.15),(.005,-.01,.1)]):
        pose=sapien.Pose([x,y,.020],euler2quat(0,math.pi/2+angle,i))
        builder.add_cylinder_visual(pose=pose,radius=.019,half_length=.003,
            material=[.045,.25,.035,1.],name=f'salad_cucumber_rind_{i}')
        builder.add_cylinder_visual(pose=pose,radius=.0164,half_length=.0032,
            material=[.64,.81,.35,1.],name=f'salad_cucumber_flesh_{i}')
