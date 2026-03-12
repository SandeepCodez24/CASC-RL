"""
Unit Tests — world_model package (Layer 2a).

Tests cover:
    - DynamicsNetwork forward pass shapes and residual property
    - EnsembleDynamicsNetwork mean/std output
    - WorldModel.predict_k_steps chain
    - TransitionDataset and DatasetBuilder integration
    - WorldModelTrainer one-epoch smoke test
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader


OBS_DIM   = 8
N_ACTIONS = 5
BATCH     = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_state():
    return torch.rand(BATCH, OBS_DIM)


@pytest.fixture
def dummy_action():
    return torch.randint(0, N_ACTIONS, (BATCH,))


@pytest.fixture
def dummy_state_np():
    return np.random.rand(BATCH, OBS_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# DynamicsNetwork
# ---------------------------------------------------------------------------

class TestDynamicsNetwork:
    def test_output_shape(self, dummy_state, dummy_action):
        from world_model.dynamics_network import DynamicsNetwork
        net = DynamicsNetwork(state_dim=OBS_DIM, action_dim=N_ACTIONS, hidden_dim=64, n_layers=2)
        s_next = net(dummy_state, dummy_action)
        assert s_next.shape == (BATCH, OBS_DIM), \
            f"Expected ({BATCH}, {OBS_DIM}), got {s_next.shape}"

    def test_soc_soh_clamped(self, dummy_state, dummy_action):
        """SoC (idx 0) and SoH (idx 1) should always be in [0, 1]."""
        from world_model.dynamics_network import DynamicsNetwork
        net = DynamicsNetwork(state_dim=OBS_DIM, action_dim=N_ACTIONS, hidden_dim=64, n_layers=2)
        with torch.no_grad():
            s_next = net(dummy_state, dummy_action)
        assert (s_next[:, 0] >= 0.0).all() and (s_next[:, 0] <= 1.0).all()
        assert (s_next[:, 1] >= 0.0).all() and (s_next[:, 1] <= 1.0).all()

    def test_single_sample(self):
        """Accept 1D (unbatched) state and scalar action."""
        from world_model.dynamics_network import DynamicsNetwork
        net = DynamicsNetwork(hidden_dim=64, n_layers=2)
        s = torch.rand(OBS_DIM)
        a = torch.tensor(2)
        with torch.no_grad():
            s_next = net(s, a)
        assert s_next.shape == (1, OBS_DIM)

    def test_residual_not_equal_input(self, dummy_state, dummy_action):
        """Output should differ from input (residual is non-zero)."""
        from world_model.dynamics_network import DynamicsNetwork
        net = DynamicsNetwork(hidden_dim=64, n_layers=2)
        with torch.no_grad():
            s_next = net(dummy_state, dummy_action)
        assert not torch.allclose(s_next, dummy_state)


# ---------------------------------------------------------------------------
# EnsembleDynamicsNetwork
# ---------------------------------------------------------------------------

class TestEnsembleDynamicsNetwork:
    def test_output_shapes(self, dummy_state, dummy_action):
        from world_model.dynamics_network import EnsembleDynamicsNetwork
        net = EnsembleDynamicsNetwork(n_ensemble=3, hidden_dim=64, n_layers=2)
        mean, std = net(dummy_state, dummy_action)
        assert mean.shape == (BATCH, OBS_DIM)
        assert std.shape  == (BATCH, OBS_DIM)

    def test_std_is_nonnegative(self, dummy_state, dummy_action):
        from world_model.dynamics_network import EnsembleDynamicsNetwork
        net = EnsembleDynamicsNetwork(n_ensemble=3, hidden_dim=64, n_layers=2)
        _, std = net(dummy_state, dummy_action)
        assert (std >= 0.0).all()

    def test_forward_single(self, dummy_state, dummy_action):
        from world_model.dynamics_network import EnsembleDynamicsNetwork
        net = EnsembleDynamicsNetwork(n_ensemble=3, hidden_dim=64, n_layers=2)
        out = net.forward_single(dummy_state, dummy_action, member_idx=0)
        assert out.shape == (BATCH, OBS_DIM)


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class TestWorldModel:
    def test_predict_one_step_shape(self):
        from world_model.world_model import WorldModel
        wm = WorldModel(hidden_dim=64, n_layers=2, n_ensemble=2, device="cpu")
        s = np.random.rand(OBS_DIM).astype(np.float32)
        s_next, _ = wm.predict_one_step(s, a=1)
        assert s_next.shape == (OBS_DIM,)

    def test_predict_k_steps_length(self):
        from world_model.world_model import WorldModel
        k = 5
        wm = WorldModel(hidden_dim=64, n_layers=2, n_ensemble=2, device="cpu")
        s = np.random.rand(OBS_DIM).astype(np.float32)
        futures = wm.predict_k_steps(s, actions=[1] * k, k=k)
        assert len(futures) == k
        for f in futures:
            assert f.shape == (OBS_DIM,)

    def test_predict_with_normalizer(self):
        from world_model.world_model import WorldModel
        wm = WorldModel(hidden_dim=64, n_layers=2, n_ensemble=2, device="cpu")
        data = np.random.rand(1000, OBS_DIM).astype(np.float32)
        wm.fit_normalizer(data)
        assert wm.is_fitted
        s = np.random.rand(OBS_DIM).astype(np.float32)
        s_next, std = wm.predict_one_step(s, a=0, return_uncertainty=True)
        assert s_next.shape == (OBS_DIM,)
        assert std.shape == (OBS_DIM,)

    def test_predict_all_actions(self):
        from world_model.world_model import WorldModel
        wm = WorldModel(hidden_dim=64, n_layers=2, n_ensemble=2, device="cpu")
        s = np.random.rand(OBS_DIM).astype(np.float32)
        results = wm.predict_all_actions(s, k=3)
        assert len(results) == N_ACTIONS
        for a, futures in results.items():
            assert len(futures) == 3

    def test_save_load(self, tmp_path):
        from world_model.world_model import WorldModel
        wm = WorldModel(hidden_dim=64, n_layers=2, n_ensemble=2, device="cpu")
        data = np.random.rand(100, OBS_DIM).astype(np.float32)
        wm.fit_normalizer(data)
        path = str(tmp_path / "wm_test.pt")
        wm.save(path)

        wm2 = WorldModel(hidden_dim=64, n_layers=2, n_ensemble=2, device="cpu")
        wm2.load(path)
        assert wm2.is_fitted


# ---------------------------------------------------------------------------
# TransitionDataset and DatasetBuilder
# ---------------------------------------------------------------------------

class TestTransitionDataset:
    def _make_dataset(self):
        from world_model.dataset_builder import TransitionDataset
        N = 200
        s      = np.random.rand(N, OBS_DIM).astype(np.float32)
        a      = np.random.randint(0, N_ACTIONS, size=(N,)).astype(np.int32)
        s_next = np.random.rand(N, OBS_DIM).astype(np.float32)
        return TransitionDataset(s, a, s_next)

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == 200

    def test_item_shapes(self):
        ds = self._make_dataset()
        s, a, s_next = ds[0]
        assert s.shape == (OBS_DIM,)
        assert a.shape == ()
        assert s_next.shape == (OBS_DIM,)

    def test_split(self):
        ds = self._make_dataset()
        train_ds, val_ds = ds.split(val_fraction=0.2)
        assert len(train_ds) == 160
        assert len(val_ds)   == 40

    def test_save_load(self, tmp_path):
        from world_model.dataset_builder import TransitionDataset
        ds = self._make_dataset()
        path = str(tmp_path / "transitions.npz")
        ds.save(path)
        ds2 = TransitionDataset.from_npz(path)
        assert len(ds2) == len(ds)

    def test_dataloader(self):
        ds = self._make_dataset()
        loader = DataLoader(ds, batch_size=32)
        batch = next(iter(loader))
        s, a, s_next = batch
        assert s.shape == (32, OBS_DIM)


class TestDatasetBuilder:
    def test_collect_shapes(self):
        from environment.constellation_env import ConstellationEnv
        from world_model.dataset_builder import DatasetBuilder
        env = ConstellationEnv(n_satellites=1, episode_length=50)
        builder = DatasetBuilder(env, seed=0)
        dataset = builder.collect(n_transitions=100, save_path=None)
        assert len(dataset) == 100
        s, a, s_next = dataset[0]
        assert s.shape == (OBS_DIM,)
        assert s_next.shape == (OBS_DIM,)


# ---------------------------------------------------------------------------
# WorldModelTrainer
# ---------------------------------------------------------------------------

class TestWorldModelTrainer:
    def test_one_epoch_smoke(self):
        from world_model.world_model import WorldModel
        from world_model.dataset_builder import TransitionDataset
        from world_model.training import WorldModelTrainer

        N = 512
        s      = np.random.rand(N, OBS_DIM).astype(np.float32)
        a      = np.random.randint(0, N_ACTIONS, size=(N,)).astype(np.int32)
        s_next = np.random.rand(N, OBS_DIM).astype(np.float32)
        dataset = TransitionDataset(s, a, s_next)

        wm = WorldModel(hidden_dim=32, n_layers=2, n_ensemble=2, device="cpu")
        trainer = WorldModelTrainer(wm, device="cpu", lr=1e-3, checkpoint_dir="/tmp/casc_test")
        history = trainer.train(dataset, n_epochs=2, batch_size=64, save_best=False, log_every=1)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
