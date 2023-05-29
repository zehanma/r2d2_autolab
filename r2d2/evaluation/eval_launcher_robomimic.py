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
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-13-pnp-redbull-try1/seed_1_ds_pnp-redbull/2023-05-13-18-13-24/models/model_epoch_200.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-15-pred-velocity-acs/ds_pnp-redbull/2023-05-15-21-07-35/models/model_epoch_300.pth"
    
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee_cam_wrist-3rd_predfutureacs_False/2023-05-16-17-55-44/models/model_epoch_300.pth" # 5/5, slightly jerky, including the gripper
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee_cam_wrist-3rd_predfutureacs_True/2023-05-16-17-55-38/models/model_epoch_300.pth"  # 1/5, missed grasped, also froze once
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee_cam_wrist_predfutureacs_False/2023-05-16-17-56-00/models/model_epoch_300.pth"  # 4/5, froze once, slow to grasp after reaching position
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee_cam_wrist_predfutureacs_True/2023-05-16-17-55-53/models/model_epoch_300.pth"  # 3/5, froze once, missed target
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee-joint_cam_wrist-3rd_predfutureacs_False/2023-05-16-17-55-22/models/model_epoch_300.pth"  # 2/5, didn't grasp once, paused in beginning, missed target once
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee-joint_cam_wrist-3rd_predfutureacs_True/2023-05-16-17-55-17/models/model_epoch_300.pth"  # 2/5, froze once, missed twice
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee-joint_cam_wrist_predfutureacs_False/2023-05-16-17-55-33/models/model_epoch_300.pth"  # 3/5, seems to freeze when can is out of view of wrist camera, missed grasp once, slow release in general
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-16-sweep/ds_pnp-redbull_proprio_ee-joint_cam_wrist_predfutureacs_True/2023-05-16-17-55-27/models/model_epoch_300.pth"
    

    # ckpt_path = "~/expdata/r2d2/im/bc/05-16-sweep/ds_pnp-redbull_proprio_ee_cam_wrist/2023-05-16-17-56-35/models/model_epoch_300.pth"  # 1/5, moves extremely slowly at first, missed grasps two times, unable to locate can sometimes
    # ckpt_path = "~/expdata/r2d2/im/bc/05-16-sweep/ds_pnp-redbull_proprio_ee_cam_wrist-3rd/2023-05-16-17-56-30/models/model_epoch_300.pth"  # 2/5, froze twice, missed grasp once, when it worked it was pretty smooth
    # ckpt_path = "~/expdata/r2d2/im/bc/05-16-sweep/ds_pnp-redbull_proprio_ee-joint_cam_wrist/2023-05-16-17-56-26/models/model_epoch_300.pth"  # 2/5, same issue with wrist camera, missed once
    # ckpt_path = "~/expdata/r2d2/im/bc/05-16-sweep/ds_pnp-redbull_proprio_ee-joint_cam_wrist-3rd/2023-05-16-17-56-21/models/model_epoch_300.pth"  # 2/5, froze 2 times even with 3rd-person camera, missed one

    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-25-c2t-and-t2c-cans/ds_pnp-t2c-cans_predfutureacs_False/2023-05-25-22-34-36/models/model_epoch_400.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-25-c2t-and-t2c-cans/ds_pnp-c2t-cans_predfutureacs_False/2023-05-25-22-34-31/models/model_epoch_100.pth"

    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr_pt/05-28-c2t-cans/ds_pnp-t2c-cans-84_predfutureacs_True/2023-05-28-05-44-43/models/model_epoch_200.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr_pt/05-28-c2t-cans/ds_pnp-t2c-cans-128_predfutureacs_True/2023-05-28-05-44-52/models/model_epoch_200.pth"
    
    ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/05-28-c2t-cans/ds_prior_pnp-t2c-cans-84_rettype_oracle_demoinput_False_fusion_xattn_resacs_False_tp_0_exclsamelyts_False/2023-05-28-05-46-18/models/model_epoch_200.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/05-28-c2t-cans/ds_prior_pnp-t2c-cans-128_rettype_oracle_demoinput_False_fusion_xattn_resacs_False_tp_0_exclsamelyts_False/2023-05-28-05-46-26/models/model_epoch_200.pth"

    IMSIZE = 84

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)

    ### custom code for loading bc_ret ckpt ###
    ISL_CKPT_PATH = "~/expdata/r2d2/im/isl/05-28-t2c-cans/ds_pnp-t2c-cans-84/2023-05-28-19-51-46/models/model_epoch_10.pth"
    config = json.loads(ckpt_dict["config"])
    config["algo"]["retrieval"]["model_ckpt_path"] = ISL_CKPT_PATH
    config["train"]["data_test"] = config["train"]["data_prior"][:3]
    ckpt_dict["config"] = json.dumps(config)

    
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)


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
            # bgr_to_rgb=False, # very important!
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
        hand_camera=dict(image=True, concatenate_images=False, resolution=(IMSIZE, IMSIZE), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(IMSIZE, IMSIZE), resize_func="cv2"),
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
