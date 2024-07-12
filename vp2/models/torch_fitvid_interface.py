import os
import json
from contextlib import ExitStack

import pprint
import torch
from vp2.mpc.utils import dict_to_numpy, slice_dict, cat_dicts
from vp2.models.model import VideoPredictionModel

from hydra.utils import to_absolute_path

try:
    from fitvid.model.fitvid import FitVid, GraspedModel
except ModuleNotFoundError:
    raise ModuleNotFoundError("FitVid is not installed. Please see the README for installation instructions.")


def slice_dict(d, start_idx, end_idx):
    return {k: v[start_idx:end_idx] for k, v in d.items()}


def cat_dicts(dicts):
    out = dict()
    for k in dicts[0].keys():
        out[k] = torch.cat([d[k] for d in dicts], dim=0)
    return out


class FitVidTorchModel(VideoPredictionModel):
    def __init__(
        self,
        checkpoint_dir,
        n_past,
        planning_modalities,
        max_batch_size=800,
        epoch=None,
        device='cuda:0',
        config_file=None
    ):
        hp = dict()
        # load HP from config file, if it exists
        self.checkpoint_file = self.get_checkpoint_file(checkpoint_dir, epoch)
        print("Loading checkpoint from", self.checkpoint_file)
        # config_path = os.path.join(os.path.dirname(self.checkpoint_file), "config.json")

        # change later
        # config_path = to_absolute_path(config_file)
        config_path = f'/home/arpit/test_projects/vp2/{config_file}'

        print("config_path: ", config_path)
        print("os.path.exists(config_path):", os.path.exists(config_path))
        if os.path.exists(config_path):
            print("Found config file! Loading hparams from it...")
            with open(config_path, "r") as config_file:
                hp = json.load(config_file)
        hp["is_inference"] = True
        self.planning_modalities = planning_modalities
        # self.num_context = n_past
        self.num_context = hp['model_kwargs']['n_past']
        self.max_batch_size = max_batch_size

        for predictor in ["depth_predictor", "normal_predictor"]:
            if hp.get(predictor, None) and not os.path.isabs(
                hp[predictor]["pretrained_weight_path"]
            ):
                hp[predictor]["pretrained_weight_path"] = to_absolute_path(
                    hp[predictor]["pretrained_weight_path"]
                )

        print("Loading FitVid model with hyperparameters:")
        pprint.pprint(hp)
        self.rgb_model = FitVid(**hp)

        num_video_channels = hp["model_kwargs"].get("video_channels", 3)
        if num_video_channels == 1:
            self.base_prediction_modality = "depth"
        elif num_video_channels == 3:
            self.base_prediction_modality = "rgb"
        self.device = device
        print("Load params")
        self.rgb_model.load_parameters(self.checkpoint_file)
        print(self.device)
        self.rgb_model.to(self.device)
        self.rgb_model.eval()

        # set up grasped model
        hp['model_kwargs']['grasped_dim'] = 1
        self.grasped_model = GraspedModel(**hp)
        self.grasped_model.set_video_prediction_model(self.rgb_model)

        grasped_model_ckpt_file = '/home/arpit/test_projects/fitvid/run_temp2/model_epoch375'
        self.grasped_model.load_parameters(grasped_model_ckpt_file)
        self.grasped_model.to(self.device)
        self.grasped_model.eval()

    def prepare_batch(self, xs):
        keys = ["video", "actions", "grasped"]
        batch = {
            k: torch.from_numpy(x).to(self.device).float() for k, x in xs.items() if k in keys
        }
        batch["video"] = torch.permute(batch["video"], (0, 1, 4, 2, 3))
        # batch['video'] = batch['video'][..., :3, :, :]
        return batch

    def format_model_epoch_filename(self, epoch):
        if epoch == "best":
            return "model_best"
        else:
            return f"model_epoch{epoch}"

    def __call__(self, batch, grad_enabled=False):
        preds = dict()
        with torch.no_grad() if not grad_enabled else ExitStack():
            batch = self.prepare_batch(batch)
            all_base_preds = list()
            all_grasped_preds = list()
            for compute_batch_idx in range(
                0, batch["video"].shape[0], self.max_batch_size
            ):
                base_preds, grasped_preds = self.grasped_model.test(
                    slice_dict(
                        batch,
                        compute_batch_idx,
                        compute_batch_idx + self.max_batch_size,
                    )
                )
                all_base_preds.append(base_preds)
                all_grasped_preds.append(grasped_preds)
            base_preds = torch.cat(all_base_preds, dim=0)
            grasped_preds = torch.cat(all_grasped_preds, dim=0)
            preds[self.base_prediction_modality] = base_preds
            preds['grasped'] = grasped_preds
            
            if "depth" in self.planning_modalities and "depth" not in preds:
                depth_preds = []
                for i in range(preds["rgb"].shape[1]):
                    depth_preds.append(
                        self.model.depth_predictor(
                            preds["rgb"][:, i, :3][:, None], time_axis=True
                        )
                    )
                depth_preds = torch.cat(depth_preds, dim=1)
                depth_preds = torch.permute(depth_preds, (0, 1, 3, 4, 2))
                preds["depth"] = depth_preds
            if "normal" in self.planning_modalities and "normal" not in preds:
                normal_preds = []
                for i in range(preds["rgb"].shape[1]):
                    normal_preds.append(
                        self.model.normal_predictor(
                            preds["rgb"][:, i, :3][:, None], time_axis=True
                        )
                    )
                normal_preds = torch.cat(normal_preds, dim=1)
                normal_preds = torch.permute(normal_preds, (0, 1, 3, 4, 2))
                preds["normal"] = normal_preds
        preds[self.base_prediction_modality] = torch.permute(
            preds[self.base_prediction_modality], (0, 1, 3, 4, 2)
        )
        if not grad_enabled:
            return dict_to_numpy(preds)
        else:
            return preds
