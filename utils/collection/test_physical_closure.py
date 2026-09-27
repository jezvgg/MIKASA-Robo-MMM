"""The search checker observes hinges; it cannot close them or erase drift."""
import torch
from my_scenes.cabinet_search import CabinetSearchConfig, SearchLatches, compartment_layout, step_search_latches

def test_closure_requires_ten_released_steps_and_reopening_resets_dwell():
    cfg=CabinetSearchConfig(); layout=compartment_layout(cfg.compartments)
    z=SearchLatches.zeros(1,4,0);z.cube_cab[:]=3
    step=0
    def tick(angle, far=True, home=False):
        nonlocal step
        step+=1
        theta=torch.tensor([[angle, .004, 0., 0.]])
        original=theta.clone()
        info,snap=step_search_latches(z,step=torch.tensor([step]),theta=theta,
            hinge_still=torch.ones((1,4),dtype=torch.bool),
            tcp_far=torch.full((1,4),far,dtype=torch.bool),at_home=torch.tensor([home]),
            foreign_hit=torch.tensor([False]),layout=layout,cfg=cfg,moved_m=torch.tensor([0.]))
        assert torch.equal(theta,original)
        assert not snap.any() and not info['snapped_now'].any()
        assert info['door_theta_max'][0,1]==theta[0,1]
        return info
    tick(0,home=True);tick(1.3)
    for _ in range(12):tick(.004,far=False)
    assert z.is_open[0,0]
    for _ in range(9):tick(.004)
    assert z.is_open[0,0]
    tick(.02)
    assert z.closed_count[0,0]==0
    for _ in range(9):tick(.004)
    assert z.is_open[0,0]
    tick(.004)
    assert not z.is_open[0,0]
    assert tick(.10)['reopened'].item()


def test_evaluate_does_not_modify_real_hinges_or_mask_observations():
    """Counterexample to the former small-drift eraser, using actual articulations."""
    import gymnasium as gym
    import my_scenes
    from utils.collection.profile import load_profile
    profile=load_profile('cabinet_search')
    env=gym.make(profile['env_id'],obs_mode='state',**profile['env_kwargs'])
    try:
        env.reset(seed=7800000);task=env.unwrapped
        for h,(art,j) in enumerate(task._hinges[0]):
            q=art.get_qpos().clone();v=art.get_qvel().clone()
            q.reshape(-1)[j]=.007*task.layout.open_dir[h]
            v.reshape(-1)[j]=.003
            art.set_qpos(q);art.set_qvel(v)
        task._elapsed_steps[:]=2
        before_q,before_v=task._read_hinges()
        for _ in range(3):
            info=task.evaluate();after_q,after_v=task._read_hinges()
            torch.testing.assert_close(after_q,before_q,rtol=0,atol=0)
            torch.testing.assert_close(after_v,before_v,rtol=0,atol=0)
            torch.testing.assert_close(info['door_qpos'],before_q,rtol=0,atol=0)
            torch.testing.assert_close(task._get_obs_extra(info)['door_qpos'],before_q,rtol=0,atol=0)
            assert not info['snapped_now'].any()
    finally:env.close()
