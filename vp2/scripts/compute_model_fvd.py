import os
import torch
import json
import numpy as np
import tqdm
import random
import piq
import lpips
import hydra
import matplotlib.pyplot as plt
import pprint
from hydra.utils import instantiate
import wandb

from fitvid.utils.fvd.fvd import get_fvd_logits, frechet_distance
from fitvid.utils.utils import dict_to_cuda
from fitvid.data.robomimic_data import load_dataset_robomimic_torch
from vp2.models.torch_fitvid_interface import FitVidTorchModel
from fitvid.utils.pytorch_metrics import flatten_image
from vp2.mpc.utils import write_moviepy_video, hori_concatenate_image
# from torchmetrics.functional import structural_similarity_index_measure as ssim

N_BATCHES = 32

@hydra.main(config_path="configs", config_name="config")
def compute_fvd(cfg):
    log_wandb = True
    ag_mode = 'ag' # ag (autoregressive) or non_ag
    np.set_printoptions(suppress=True, precision=3)
    if log_wandb:
        wandb.login()
        run = wandb.init(
            project="VMPC",
            notes="trial experiment"
        )
        table_1 = wandb.Table(columns=["F_name", "Action", "Video", "Pixel Error", "Grasped State", "Grasped State Errror"])
        rows = []

    pprint.pprint(dict(cfg))
    lpips_official = lpips.LPIPS(net="alex").cuda()
    model = instantiate(cfg.model)

    # set up logging to json
    model_checkpoint_file = model.checkpoint_file
    print("model_checkpoint_file: ", model_checkpoint_file)
    model_checkpoint_dir = os.path.dirname(model_checkpoint_file)
    save_metrics_path = os.path.join(model_checkpoint_dir, "metrics.json")
    print(f"Save metrics path: {save_metrics_path}")
    # if already exists
    if os.path.isfile(save_metrics_path):
        with open(save_metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = dict()

    # format: is a dictionary where the keys are epoch numbers
    # and each entry is a dictionary with the format metric:value
    if "epoch" in cfg.model:
        epoch = cfg.model.epoch
    else:
        epoch = int(
            "".join(filter(str.isdigit, os.path.basename(model_checkpoint_file)))
        )

    # dataset_file = '/home/arpit/test_projects/vp2/vp2/robosuite_benchmark_tasks/5k_slice_rendered_256.hdf5'
    # dataset_name = 'robosuite_benchmark_tasks/combined'
    dataset_file = "/home/arpit/test_projects/OmniGibson/dynamics_model_dataset_test_2/dataset.hdf5"
    dataset_name = 'og'

    # print("cfg: ", cfg.keys())
    # dataset_name = "/".join(cfg.dataset.dataset_file.split("/")[-2:])
    # print("dataset_name: ", dataset_name)
    key = str(epoch) + "_" + dataset_name

    # if key in metrics:
    #     print(
    #         f"Key {key} in metrics already, check for results in {save_metrics_path}!"
    #     )

    # real_embeddings = []
    # predicted_embeddings = []
    # if "robosuite" in cfg.env._target_:
    #     view = "agentview_shift_2"
    # elif "robodesk" in cfg.env._target_:
    #     view = "camera"
    # else:
    #     raise ValueError(f"Unknown environment being used")

    test_data_loader = load_dataset_robomimic_torch(
        # [cfg.dataset.dataset_file],
        [dataset_file],
        batch_size=1,
        video_len=8,
        video_dims=(64, 64),
        phase=None,
        depth=False,
        normal=False,
        view='rgb',
        cache_mode=None,
        seg=False,
        only_depth=False,
        augmentation=None,
        shuffle=False
    )
    print("test_data_loader loaded")
    print("len(test_data_loader): ", len(test_data_loader))
    print("len(test_data_loader.dataset): ", len(test_data_loader.dataset))
    dataset = test_data_loader.dataset
    
    # # printing the obs of episode
    # ob = dataset[9]['obs']['rgb']
    # print("ob: ", ob.shape)
    # fig, ax = plt.subplots(2, 4)
    # ax[0][0].imshow(ob[0].transpose(1,2,0))
    # ax[0][1].imshow(ob[1].transpose(1,2,0))
    # ax[0][2].imshow(ob[2].transpose(1,2,0))
    # ax[0][3].imshow(ob[3].transpose(1,2,0))
    # ax[1][0].imshow(ob[4].transpose(1,2,0))
    # ax[1][1].imshow(ob[5].transpose(1,2,0))
    # ax[1][2].imshow(ob[6].transpose(1,2,0))
    # plt.show() 


    # for i in range(len(dataset)):
    #     print("i: ", dataset[i]['actions'].shape)
    
    mse = []
    lpips_all = []
    ssim_all = []
    pbar = tqdm.tqdm(total=N_BATCHES)
    for i, batch in enumerate(test_data_loader):
        # if i < 20:
        #     continue
        # print("batch: ", batch.keys())
        # print("batch (video, actions, rewards, segmentation): ", batch['video'].shape, batch['actions'].shape, batch['segmentation'])
        # print("--", batch['actions'].shape)
        # print("11: ", batch['actions'][0, :2])
        # testing what happens if all actions have zero delta orn
        # batch['actions'][:, :, 3:6] = torch.tensor([0.0, 0.0, 0.0])
        # print("22: ", batch['actions'][0, :2])
        # input()
        batch = dict_to_cuda(batch)
        batch["actions"] = batch["actions"].float()
        if isinstance(model, FitVidTorchModel):
            with torch.no_grad():
                _, eval_preds, eval_grasped_preds = model.model.evaluate(batch, compute_metrics=False)
        else:
            with torch.no_grad():
                batch_prepped = dict()
                batch_prepped["video"] = (
                    batch["video"].permute(0, 1, 3, 4, 2).cpu().numpy()[:, :2]
                )
                batch_prepped["actions"] = batch["actions"].cpu().numpy()
                outputs = model(batch_prepped)
                outputs["rgb"] = (
                    torch.tensor(outputs["rgb"][:, 1:]).permute(0, 1, 4, 2, 3).cuda()
                )
                eval_preds = dict(ag=outputs)
        gt_video = (batch["video"][:, 1:] * 255).to(torch.uint8)
        pred_video = (eval_preds[ag_mode]["rgb"] * 255).to(torch.uint8)
        gt_video_temp = gt_video.cpu()
        pred_video_temp = pred_video.cpu()
        print("gt_video, pred_video: ", gt_video.shape, pred_video.shape)

        # save video to disk
        folder_name = 'fitvid_predictions'
        os.makedirs(folder_name, exist_ok=True)
        concat_imgs = []
        for ind in range(len(gt_video_temp[0])):
            concat_img = hori_concatenate_image([gt_video_temp[0][ind].permute(1,2,0), pred_video_temp[0][ind].permute(1,2,0)])
            concat_imgs.append(concat_img)
        write_moviepy_video(concat_imgs, f'{folder_name}/{i:05d}.mp4')

        # get grasped preds
        gt_grasped = batch["grasped"][:, 1:]
        gt_grasped = np.squeeze(gt_grasped.cpu().numpy())
        pred_grasped = eval_grasped_preds[ag_mode]
        pred_grasped = np.squeeze(pred_grasped.cpu().numpy())
        pred_grasped = np.round(pred_grasped)
        print("gt_grasped, pred_grasped: ", gt_grasped, pred_grasped, np.sum(gt_grasped != pred_grasped))
        # input()

        if i > -1:
            fig, ax = plt.subplots(2, 6)
            ax[0][0].imshow(gt_video_temp[0][0].permute(1,2,0))
            ax[0][1].imshow(gt_video_temp[0][2].permute(1,2,0))
            ax[0][2].imshow(gt_video_temp[0][3].permute(1,2,0))
            ax[0][3].imshow(gt_video_temp[0][4].permute(1,2,0))
            ax[0][4].imshow(gt_video_temp[0][5].permute(1,2,0))
            ax[0][5].imshow(gt_video_temp[0][6].permute(1,2,0))
            ax[1][0].imshow(pred_video_temp[0][0].permute(1,2,0))
            ax[1][1].imshow(pred_video_temp[0][2].permute(1,2,0))
            ax[1][2].imshow(pred_video_temp[0][3].permute(1,2,0))
            ax[1][3].imshow(pred_video_temp[0][4].permute(1,2,0))
            ax[1][4].imshow(pred_video_temp[0][5].permute(1,2,0))
            ax[1][5].imshow(pred_video_temp[0][6].permute(1,2,0))
            plt.show()
        
        with torch.no_grad():

            lpips_official_score = lpips_official(
                flatten_image(gt_video / 255.0) * 2 - 1,
                flatten_image(pred_video / 255.0) * 2 - 1,
            )

        lpips_all.append(lpips_official_score.mean())
        ssim_all.append(
            piq.ssim(flatten_image(gt_video / 255.0), flatten_image(pred_video / 255.0))
        )
        mse.append(
            torch.mean(
                ((batch["video"][:, 1:] - eval_preds["ag"]["rgb"]) ** 2),
                dim=(1, 2, 3, 4),
            )
        )

        # for logging to wandb
        if log_wandb:
            # batch["actions"].cpu().numpy()[0]
            rows.append([f'{i:05d}', 
                         0, 
                         wandb.Video(f'{folder_name}/{i:05d}.mp4', fps=30, format="mp4"), 
                         np.array2string(mse[-1].cpu().numpy()), 
                         np.array2string(gt_grasped)+'\n'+np.array2string(pred_grasped), #np.array2string(np.array([gt_grasped.cpu().numpy(), pred_grasped.cpu().numpy()])), 
                         np.sum(gt_grasped != pred_grasped)])

        # real_embeddings.append(get_fvd_logits(gt_video).detach().cpu())
        # predicted_embeddings.append(get_fvd_logits(pred_video).detach().cpu())
        pbar.update(1)
        if i == N_BATCHES:
            break
    pbar.close()

    # # real_embeddings = torch.cat(real_embeddings, dim=0)
    # # predicted_embeddings = torch.cat(predicted_embeddings, dim=0)
    # # result = frechet_distance(real_embeddings, predicted_embeddings)
    # mse = torch.cat(mse, dim=0).mean()
    # ssim_all = torch.stack(ssim_all).mean()
    # lpips_all = torch.stack(lpips_all, dim=0).mean()
    # # print("FVD: {}".format(result.item()))
    # print("MSE: {}".format(mse.item()))
    # print("LPIPS: {}".format(lpips_all.item()))
    # print("SSIM: {}".format(ssim_all.item()))

    # metrics[key] = dict(
    #     # fvd=result.item(),
    #     mse=mse.item(),
    #     lpips=lpips_all.item(),
    #     ssim=ssim_all.item(),
    # )

    # with open(save_metrics_path, "w") as f:
    #     json.dump(metrics, f)

    if log_wandb:
        table_1 = wandb.Table(
            columns=table_1.columns, data=rows
        )
        run.log({"Videos": table_1}) 


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    compute_fvd()
