import pytest
from utils.collection.drawer_balance import next_seeds, interleave


def manifest():
    strata={i:list(range(100+i,116,4)) for i in range(4)}
    return dict(seeds=interleave(strata),seed_to_drawer={str(s):i for i,ss in strata.items() for s in ss})


def test_failures_count_as_attempted_but_do_not_fill_quota():
    m=manifest()
    assert next_seeds(m,{100,101,102,103},{101,102,103},count=4,quota=1)==[104]


def test_resume_keeps_bank_order_and_never_overshoots_quota():
    m=manifest()
    assert next_seeds(m,set(),set(),count=8,quota=1)==[100,101,102,103]
    assert next_seeds(m,{100,101},{100},count=4,quota=1)==[102,103,105]


def test_acceptance_requires_a_recorded_attempt():
    with pytest.raises(ValueError):next_seeds(manifest(),set(),{100})
