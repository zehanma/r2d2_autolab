from r2d2.evaluation.eval_launcher_robomimic import eval_launcher
import os

CKPT_BASE_PATHS = [os.path.join("~/expdata/r2d2/im/bc_ret_pt", p) for p in [
    # ### old ckpts ###
    # "06-07-im84-2cams-try2/ds_prior_im84-all_tp_0_predretacs_False/2023-06-07-08-04-46/models/model_epoch_200.pth", # bc
    # "06-07-im84-2cams-try2/ds_prior_im84-all_rettype_z_tp_0_predretacs_False/2023-06-07-08-04-47/models/model_epoch_200.pth", # bam
    # "06-07-im84-2cams-try2/ds_prior_im84-all_rettype_z_tp_0_predretacs_True/2023-06-07-08-04-59/models/model_epoch_200.pth", # bam-acs

    
    ### bc ###
    "06-07-bc-vs-bam/seed_1_ds_prior_im84-all_tp_0_predretacs_False/2023-06-07-19-38-44/models/model_epoch_200.pth"
    # "06-08-bc-vs-bam/seed_2_ds_prior_im84-all_tp_0_predretacs_False/2023-06-08-00-37-44/models/model_epoch_200.pth",
    # "06-08-bc-vs-bam/seed_3_ds_prior_im84-all_tp_0_predretacs_False/2023-06-08-00-37-57/models/model_epoch_200.pth",

    ### bam ###
    # "06-08-bc-vs-bam/seed_1_ds_prior_im84-all_rettype_z_tp_0_predretacs_False/2023-06-08-00-37-44/models/model_epoch_200.pth",
    # "06-08-bc-vs-bam/seed_2_ds_prior_im84-all_rettype_z_tp_0_predretacs_False/2023-06-08-00-37-47/models/model_epoch_200.pth",
    # "06-08-bc-vs-bam/seed_3_ds_prior_im84-all_rettype_z_tp_0_predretacs_False/2023-06-08-00-38-01/models/model_epoch_200.pth",
    
    ### bam-acs ###
    # "06-08-bc-vs-bam/seed_1_ds_prior_im84-all_rettype_z_tp_0_predretacs_True/2023-06-08-00-37-44/models/model_epoch_200.pth",
    # "06-08-bc-vs-bam/seed_2_ds_prior_im84-all_rettype_z_tp_0_predretacs_True/2023-06-08-00-37-52/models/model_epoch_200.pth",
    # "06-08-bc-vs-bam/seed_3_ds_prior_im84-all_rettype_z_tp_0_predretacs_True/2023-06-08-00-38-07/models/model_epoch_200.pth",
    
]]
# print("Evaluating:", CKPT_BASE_PATHS[0])
# input("Press enter to confirm >>")

TASKS=[
    # "t2s-cans",
    # "s2t-chips"
    "t2c-chips",
]

LAYOUT_ID=0

assert len(CKPT_BASE_PATHS) == 1
assert len(TASKS) == 1

variant = dict(
    exp_name="policy_test",
    save_data=False,
    use_gpu=True,
    seed=0,
    policy_logdir="pnp_redbull/run3/id0/",
    model_id=50,
    camera_kwargs=dict(),
    data_processing_kwargs=dict(
        timestep_filtering_kwargs=dict(),
        image_transform_kwargs=dict(),
    ),
    ckpt_path=CKPT_BASE_PATHS[0],
    task=TASKS[0],
    layout_id=LAYOUT_ID,
)

if __name__ == "__main__":
    eval_launcher(variant, run_id=1, exp_id=0)
