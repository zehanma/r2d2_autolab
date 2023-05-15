import json
import os

import numpy as np
import torch

from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.evaluation.policy_wrapper import PolicyWrapperRobomimic
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils


def eval_launcher(variant, run_id, exp_id):
    # Get Directory #
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Prepare Log Directory #
    variant["exp_name"] = os.path.join(variant["exp_name"], "run{0}/id{1}/".format(run_id, exp_id))
    log_dir = os.path.join(dir_path, "../../evaluation_logs", variant["exp_name"])

    # Set Random Seeds #
    torch.manual_seed(variant["seed"])
    np.random.seed(variant["seed"])

    # Set Compute Mode #
    use_gpu = variant.get("use_gpu", False)
    torch.device("cuda:0" if use_gpu else "cpu")

    # Load Model + Variant #
    # policy_logdir = os.path.join(dir_path, "../../training_logs", variant["policy_logdir"])
    # policy_filepath = os.path.join(policy_logdir, "models", "{0}.pt".format(variant["model_id"]))
    # policy = torch.load(policy_filepath)

    # load the policy, use hardcoded values for now
    ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-13-pnp-redbull-try1/seed_1_ds_pnp-redbull/2023-05-13-18-13-24/models/model_epoch_100.pth"
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # variant_filepath = os.path.join(policy_logdir, "variant.json")
    # with open(variant_filepath, "r") as jsonFile:
    #     policy_variant = json.load(jsonFile)

    # Prepare Policy Wrapper #
    # data_processing_kwargs = variant.get("data_processing_kwargs", {})
    data_processing_kwargs = dict(
        timestep_filtering_kwargs=dict(
            action_space="cartesian_velocity",
            robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
            camera_extrinsics=[],
        ),
        image_transform_kwargs=dict(
            remove_alpha=True,
            bgr_to_rgb=True,
            to_tensor=True,
            augment=False,
        ),
    )
    timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
    image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

    policy_data_processing_kwargs = {} #policy_variant.get("data_processing_kwargs", {})
    policy_timestep_filtering_kwargs = policy_data_processing_kwargs.get("timestep_filtering_kwargs", {})
    policy_image_transform_kwargs = policy_data_processing_kwargs.get("image_transform_kwargs", {})

    policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
    policy_image_transform_kwargs.update(image_transform_kwargs)

    wrapped_policy = PolicyWrapperRobomimic(
        policy=policy,
        timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
        image_transform_kwargs=policy_image_transform_kwargs,
        eval_mode=True,
    )

    # Prepare Environment #
    policy_action_space = policy_timestep_filtering_kwargs["action_space"]

    # camera_kwargs = variant.get("camera_kwargs", {})
    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(128, 128), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(128, 128), resize_func="cv2"),
    )
    
    policy_camera_kwargs = {} #policy_variant.get("camera_kwargs", {})
    policy_camera_kwargs.update(camera_kwargs)

    env = RobotEnv(action_space=policy_action_space, camera_kwargs=policy_camera_kwargs)
    controller = VRPolicy()

    # Launch GUI #
    data_collector = DataCollecter(
        env=env,
        controller=controller,
        policy=wrapped_policy,
        save_traj_dir=log_dir,
        save_data=variant.get("save_data", True),
    )
    RobotGUI(robot=data_collector)