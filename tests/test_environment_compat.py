"""Verify optional legacy MuJoCo registration without installing mujoco-py."""

import unittest
from types import SimpleNamespace
from unittest import mock

import dqa_continuous_action as trainer


def moved_mujoco_placeholder():
    """Stand in for Gymnasium's relocated legacy-environment entry point."""
    raise AssertionError("Registration checks must not create an environment")


moved_mujoco_placeholder.__name__ = "_raise_mujoco_py_error"
moved_mujoco_placeholder.__module__ = "gymnasium.envs"


class EnvironmentCompatibilityTests(unittest.TestCase):
    def test_available_environments_keep_existing_registration(self):
        entries = {
            "Hopper-v3": "gymnasium.envs.mujoco.hopper_v3:HopperEnv",
            "Hopper-v5": "gymnasium.envs.mujoco.hopper_v5:HopperEnv",
            "CustomTask-v3": "custom_envs:CustomTask",
        }
        for env_id, entry_point in entries.items():
            with self.subTest(env_id=env_id):
                spec = SimpleNamespace(entry_point=entry_point)
                with mock.patch.object(trainer.gym, "registry", {env_id: spec}), \
                     mock.patch.object(trainer.importlib, "import_module") as import_module, \
                     mock.patch.object(trainer.gym, "register_envs") as register_envs:
                    trainer.ensure_environment_registered(env_id)
                import_module.assert_not_called()
                register_envs.assert_not_called()
                self.assertEqual(spec.entry_point, entry_point)

    def test_relocated_mujoco_environment_is_registered_under_requested_id(self):
        env_id = "Hopper-v3"
        registry = {env_id: SimpleNamespace(entry_point=moved_mujoco_placeholder)}
        robotics = SimpleNamespace(__name__="gymnasium_robotics")

        def import_robotics(module_name):
            self.assertEqual(module_name, "gymnasium_robotics")
            registry[env_id] = SimpleNamespace(
                entry_point="gymnasium_robotics.envs.mujoco.hopper_v3:HopperEnv"
            )
            return robotics

        with mock.patch.object(trainer.gym, "registry", registry), \
             mock.patch.object(trainer.importlib, "import_module", side_effect=import_robotics) as import_module, \
             mock.patch.object(trainer.gym, "register_envs") as register_envs, \
             mock.patch.object(trainer.gym, "make") as make:
            trainer.ensure_environment_registered(env_id)
        import_module.assert_called_once_with("gymnasium_robotics")
        register_envs.assert_called_once_with(robotics)
        make.assert_not_called()
        self.assertEqual(list(registry), [env_id])
        self.assertIn("hopper_v3", registry[env_id].entry_point)

    def test_missing_or_broken_optional_dependency_has_clear_error(self):
        env_id = "Hopper-v3"
        for import_error in (ModuleNotFoundError("gymnasium_robotics"), ImportError("broken dependency")):
            with self.subTest(error=type(import_error).__name__):
                registry = {env_id: SimpleNamespace(entry_point=moved_mujoco_placeholder)}
                with mock.patch.object(trainer.gym, "registry", registry), \
                     mock.patch.object(trainer.importlib, "import_module", side_effect=import_error), \
                     mock.patch.object(trainer.gym, "register_envs") as register_envs, \
                     mock.patch.object(trainer.gym, "make") as make:
                    with self.assertRaises(RuntimeError) as raised:
                        trainer.ensure_environment_registered(env_id)
                self.assertIn(env_id, str(raised.exception))
                self.assertIn("gymnasium-robotics", str(raised.exception))
                register_envs.assert_not_called()
                make.assert_not_called()
                self.assertIs(registry[env_id].entry_point, moved_mujoco_placeholder)

    def test_import_that_leaves_placeholder_is_rejected(self):
        env_id = "Hopper-v3"
        registry = {env_id: SimpleNamespace(entry_point=moved_mujoco_placeholder)}
        robotics = SimpleNamespace(__name__="gymnasium_robotics")
        with mock.patch.object(trainer.gym, "registry", registry), \
             mock.patch.object(trainer.importlib, "import_module", return_value=robotics), \
             mock.patch.object(trainer.gym, "register_envs") as register_envs, \
             mock.patch.object(trainer.gym, "make") as make:
            with self.assertRaises(RuntimeError) as raised:
                trainer.ensure_environment_registered(env_id)
        self.assertIn(env_id, str(raised.exception))
        register_envs.assert_called_once_with(robotics)
        make.assert_not_called()

    def test_same_named_callable_outside_gymnasium_is_not_replaced(self):
        def unrelated_entry_point():
            raise AssertionError("Entry points must not be executed")

        unrelated_entry_point.__name__ = "_raise_mujoco_py_error"
        unrelated_entry_point.__module__ = "custom_envs"
        env_id = "CustomTask-v3"
        spec = SimpleNamespace(entry_point=unrelated_entry_point)
        with mock.patch.object(trainer.gym, "registry", {env_id: spec}), \
             mock.patch.object(trainer.importlib, "import_module") as import_module, \
             mock.patch.object(trainer.gym, "register_envs") as register_envs:
            trainer.ensure_environment_registered(env_id)
        import_module.assert_not_called()
        register_envs.assert_not_called()
        self.assertIs(spec.entry_point, unrelated_entry_point)


if __name__ == "__main__":
    unittest.main()
