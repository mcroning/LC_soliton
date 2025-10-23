def test_imports():
    import lc_soliton.lc_core as lc
    assert hasattr(lc, "advance_theta_timestep")
