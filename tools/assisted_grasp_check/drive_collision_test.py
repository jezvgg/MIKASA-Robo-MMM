"""Does a SAPIEN drive between two bodies switch off their collision? (assisted grasp makes one drive
per graspable object, hand -> object, when the env is built and only toggles it with enable/disable:
utils/mikasa/assisted_grasp.py.)

A 10 cm box is dropped onto a fixed 1 m slab. If the pair cannot collide the box falls through.
  none      no drive;
  enabled   a force-free drive box <-> slab (free translation, no drive targets);
  disabled  the same drive, then .disable() (what assisted grasp leaves on every object it is not holding);
  removed   the same drive, then its entity removed from the scene;
  proxy     box <-> proxy <-> slab: two force-free drives through a shapeless gravity-free body.
usage: python drive_collision_test.py"""
import sapien


def drop(mode: str) -> dict:
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[0.5, 0.5, 0.05])
    slab = b.build_kinematic(name="slab")
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[0.05, 0.05, 0.05])
    box = b.build(name="box")
    box.set_pose(sapien.Pose([0, 0, 0.5]))

    def free_drive(a, c):
        d = scene.create_drive(a, sapien.Pose(), c, sapien.Pose())
        for name in ("x", "y", "z"):
            getattr(d, f"set_limit_{name}")(-10, 10)
        return d

    if mode in ("enabled", "disabled", "removed"):
        d = free_drive(slab, box)
        if mode == "disabled":
            d.disable()
        if mode == "removed":
            scene.remove_entity(d.entity)
    if mode == "proxy":
        b = scene.create_actor_builder()
        b.set_mass_and_inertia(0.01, sapien.Pose(), [1e-5, 1e-5, 1e-5])
        proxy = b.build(name="proxy")
        proxy.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).set_disable_gravity(True)
        free_drive(slab, proxy)
        free_drive(proxy, box)
    contacts = 0
    for _ in range(200):
        scene.step()
        contacts += sum(1 for c in scene.get_contacts()
                        if {c.bodies[0].entity.name, c.bodies[1].entity.name} == {"slab", "box"})
    return {"final_z_m": round(float(box.get_pose().p[2]), 3), "contact_steps": contacts}


if __name__ == "__main__":
    for mode in ("none", "enabled", "disabled", "removed", "proxy"):
        print(f"{mode:9s}", drop(mode))
