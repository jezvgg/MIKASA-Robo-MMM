"""Does a SAPIEN drive between two bodies switch off their collision? (assisted grasp makes one drive
per graspable object, hand -> object, when the env is built: utils/mikasa/assisted_grasp.py:226.)

A 10 cm box is dropped onto a fixed 1 m slab, once without a drive and once with a drive between box
and slab whose position limits are all free and whose drives are all zero, i.e. a drive that exerts
no force. If the drive switches collision off, the box falls through the slab.
usage: python drive_collision_test.py"""
import numpy as np
import sapien


def drop(with_drive: bool) -> dict:
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    slab = scene.create_actor_builder()
    slab.add_box_collision(half_size=[0.5, 0.5, 0.05])
    slab = slab.build_kinematic(name="slab")
    box = scene.create_actor_builder()
    box.add_box_collision(half_size=[0.05, 0.05, 0.05])
    box = box.build(name="box")
    box.set_pose(sapien.Pose([0, 0, 0.5]))
    if with_drive:
        drive = scene.create_drive(slab, sapien.Pose(), box, sapien.Pose())
        for name in ("x", "y", "z"):
            getattr(drive, f"set_limit_{name}")(-10, 10)   # free translation; no stiffness set
    contacts = 0
    for _ in range(200):
        scene.step()
        contacts += sum(1 for c in scene.get_contacts()
                        if {c.bodies[0].entity.name, c.bodies[1].entity.name} == {"slab", "box"})
    out = {"final_z_m": round(float(box.get_pose().p[2]), 3), "contact_steps": contacts}
    if with_drive:
        out["drive_methods_with_collision"] = [m for m in dir(drive) if "collision" in m.lower()]
    return out


if __name__ == "__main__":
    print("no drive  :", drop(False))
    print("with drive:", drop(True))
