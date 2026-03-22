"""Tests for the RustQuant.stochastics module."""

from RustQuant.stochastics import (
    GeometricBrownianMotion,
    ArithmeticBrownianMotion,
    BrownianMotion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HullWhite,
    MertonJumpDiffusion,
)


def test_gbm_simulate():
    gbm = GeometricBrownianMotion(mu=0.05, sigma=0.20)
    traj = gbm.simulate(x0=100.0, t_end=1.0, n_steps=10, n_paths=5)
    assert len(traj.paths) == 5
    assert len(traj.times) == 11  # n_steps + 1
    assert all(p[0] == 100.0 for p in traj.paths)


def test_abm_simulate():
    abm = ArithmeticBrownianMotion(mu=0.0, sigma=1.0)
    traj = abm.simulate(x0=0.0, t_end=1.0, n_steps=100, n_paths=3)
    assert len(traj.paths) == 3


def test_bm_simulate():
    bm = BrownianMotion()
    traj = bm.simulate(x0=0.0, t_end=1.0, n_steps=50, n_paths=2)
    assert len(traj.paths) == 2


def test_ou_mean_reverts():
    ou = OrnsteinUhlenbeck(mu=1.0, sigma=0.1, theta=5.0)
    traj = ou.simulate(x0=1.0, t_end=10.0, n_steps=1000, n_paths=100)
    terminals = [p[-1] for p in traj.paths]
    mean_t = sum(terminals) / len(terminals)
    assert abs(mean_t - 1.0) < 0.2


def test_cir_non_negative():
    cir = CoxIngersollRoss(mu=0.05, sigma=0.05, theta=1.0)
    traj = cir.simulate(x0=0.05, t_end=5.0, n_steps=500, n_paths=10)
    for path in traj.paths:
        assert all(v >= -0.01 for v in path)  # small tolerance for discretisation


def test_hw_simulate():
    hw = HullWhite(alpha=0.1, sigma=0.01, theta=0.05)
    traj = hw.simulate(x0=0.03, t_end=1.0, n_steps=100, n_paths=3)
    assert len(traj.paths) == 3


def test_mjd_simulate():
    mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, **{"lambda": 5.0}, jump_mean=0.0, jump_volatility=0.05)
    traj = mjd.simulate(x0=100.0, t_end=1.0, n_steps=252, n_paths=3)
    assert len(traj.paths) == 3


def test_gbm_repr():
    gbm = GeometricBrownianMotion(mu=0.05, sigma=0.20)
    assert "GeometricBrownianMotion" in repr(gbm)
