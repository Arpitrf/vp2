import abc
import numpy as np
import torch
import piq
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt

from vp2.mpc.utils import slice_dict
from vp2.util.conv_predictor import ConvPredictor


def sum(x, dim):
    # torch and np agnostic sum
    if isinstance(x, torch.Tensor):
        return x.sum(dim=dim)
    elif isinstance(x, np.ndarray):
        return x.sum(axis=dim)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def stack(x, dim=0):
    # torch and np agnostic stack
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=dim)
    elif isinstance(x[0], np.ndarray):
        return np.stack(x, axis=dim)
    else:
        raise ValueError(f"Unknown type {type(x)}")


class Objective(metaclass=abc.ABCMeta):
    def __init__(self, weight=1):
        self.weight = weight

    def compute_reward(self, prediction, goals):
        pass

    def __call__(self, predictions, goal, **kwargs):
        # IMPORTANT: All Objective subclasses compute REWARDS, not costs/losses.
        # This means that for any implemented objective, higher is better.
        return self.compute_reward(predictions, goal) * self.weight
    
class ClassMismatchError(Objective):
    def __init__(self, key, weight):
        super().__init__(weight)
        self.key = key

    def vectorized_sequence_ious(self, predictions, ground_truth):
        print("preductions, groud_truth shapes: ", predictions.shape, ground_truth.shape)
        # Convert to labels
        pred_labels = np.argmax(predictions, axis=-1)  # Shape: (200, 7, 64, 64)
        gt_labels = np.argmax(ground_truth, axis=-1)   # Shape: (7, 64, 64)
        
        # Expand ground truth to match the shape of predictions
        gt_labels = np.expand_dims(gt_labels, axis=0)  # Shape: (1, 7, 64, 64)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_labels == gt_labels, gt_labels > 0)  # Shape: (200, 7, 64, 64)
        union = np.logical_or(pred_labels == gt_labels, gt_labels > 0)          # Shape: (200, 7, 64, 64)
        print("intersection.shape, union.shape: ", intersection.shape, union.shape)

        # Sum across spatial dimensions (64, 64) to get counts
        intersection_sum = np.sum(intersection, axis=(2, 3))  # Shape: (200, 7)
        union_sum = np.sum(union, axis=(2, 3))                # Shape: (200, 7)
        
        # Avoid division by zero by adding a small epsilon where union_sum is zero
        epsilon = 1e-10
        iou = intersection_sum / (union_sum + epsilon)  # Shape: (200, 7)
        
        # Sum IoUs across the sequence frames (axis=1)
        iou_sums = np.sum(iou, axis=1)  # Shape: (200,)
        
        return iou_sums
    
    def mean_iou(self, seg1, seg2):
        # # Convert one-hot encoded segmentation maps to class labels
        # seg1_labels = np.argmax(seg1, axis=-1)
        # seg2_labels = np.argmax(seg2, axis=-1)
        
        iou_list = []
        final_iou = 0
        for s in range(seg1.shape[0]):
            seg1_labels = seg1[s]
            seg2_labels = seg2[s]
            for i in range(20):  # For each class
                intersection = np.logical_and(seg1_labels == i, seg2_labels == i).sum()
                union = np.logical_or(seg1_labels == i, seg2_labels == i).sum()
                if union != 0:
                    iou = intersection / union
                    iou_list.append(iou)
                else:
                    iou_list.append(np.nan)
            final_iou += np.nanmean(iou_list)
        return final_iou
    
    def calculate_pixel_mismatches(self, seg1, seg2):
        # # Convert one-hot encoded segmentation maps to class labels
        # seg1_labels = np.argmax(seg1, axis=-1)
        # seg2_labels = np.argmax(seg2, axis=-1)
        
        # Calculate mismatches per pixel
        mismatches = seg1 != seg2
        
        # Count the number of mismatches
        num_mismatches = mismatches.sum()
        
        # Calculate total number of pixels
        total_pixels = mismatches.size
        
        # Calculate percentage of mismatches
        mismatch_percentage = (num_mismatches / total_pixels) * 100
        
        return num_mismatches, mismatch_percentage
    
    def compute_reward(self, prediction, goal):
        reward = 0
        goal_seg_img = np.argmax(goal[self.key], axis=-1)
        prediction_seg_img = prediction[self.key]
        
        # fig, ax = plt.subplots(5, 7)
        # for i in range(7):
        #     ax[0][i].imshow(prediction_seg_img[0, i])
        #     ax[1][i].imshow(prediction_seg_img[14,i])
        #     ax[2][i].imshow(prediction_seg_img[29,i])
        #     ax[3][i].imshow(prediction_seg_img[33,i])
        #     ax[4][i].imshow(prediction_seg_img[199,i])
        # plt.show()
        
        # print(prediction_seg_img.shape, goal_seg_img.shape)
        # print("prediction example values: ", prediction[self.key][0, 0, :5, :5])

        # fig, ax = plt.subplots(2, 7)
        # for i in range(7):
        #     ax[0][i].imshow(goal_seg_img[i])
        #     ax[1][i].imshow(prediction_seg_img[0,i])
        # plt.show()

        # TODO: Change this logic later
        temp_dict = {
            2: 6,
            3: 3,
            5: 7,
            11: 10,
            9: 8
        }
        new_goal_seg_img = goal_seg_img.copy()
        for old_class, new_class in temp_dict.items():
            new_goal_seg_img[goal_seg_img == old_class] = new_class
        goal_seg_img = new_goal_seg_img.copy()

        # fig, ax = plt.subplots(2, 7)
        # for i in range(7):
        #     ax[0][i].imshow(goal_seg_img[i])
        #     ax[1][i].imshow(prediction_seg_img[0,i])
        # plt.show()

        reward = []
        for i in range(prediction_seg_img.shape[0]):
            # mean_iou = self.mean_iou(prediction_seg_img[i], goal_seg_img) 
            num_mismatches,  mismatch_percentage = self.calculate_pixel_mismatches(prediction_seg_img[i], goal_seg_img)
            reward.append(-num_mismatches)

        reward = np.array(reward)
        
        # 2. obtain dynamics model predictions that predict a "grasped" state equal to the expected "grasped" state will occur at some time step
        # and retain the corresponding actions
        ind_w_grasps = []
        for j in range(len(prediction['grasped'])):
            if any(np.squeeze(prediction['grasped'][j])):
                ind_w_grasps.append(j)
        if len(ind_w_grasps) > 0:
            for j in range(len(prediction['grasped'])):
                if j not in ind_w_grasps:
                    reward[j] = -1000000000
        
        return reward[:, None, None]


class SquaredError(Objective):
    def __init__(self, key, weight):
        super().__init__(weight)
        self.key = key

    def compute_reward(self, prediction, goal):
        # 1. compute pixel cost
        cost = (prediction[self.key] - goal[self.key]) ** 2
        # sum works much better than mean -- mean has small magnitudes (and floating point errors?)
        reward = -sum(cost, dim=(1, 2, 3, 4))
        
        # 2. obtain dynamics model predictions that predict a "grasped" state equal to the expected "grasped" state will occur at some time step
        # and retain the corresponding actions
        ind_w_grasps = []
        for j in range(len(prediction['grasped'])):
            if any(np.squeeze(prediction['grasped'][j])):
                ind_w_grasps.append(j)
        if len(ind_w_grasps) > 0:
            for j in range(len(prediction['grasped'])):
                if j not in ind_w_grasps:
                    reward[j] = -1000000000

        # 3. sort the filtered actions by pixel cost
        
        # print("reward: ", reward)
        return reward[:, None, None]


class CombinedObjective(Objective):
    def __init__(self, objectives, combine_method="sum"):
        super().__init__(weight=1)
        self.objectives = objectives
        self.combine_method = combine_method

    def compute_reward(self, prediction, goal):
        results = list()
        for name, objective in self.objectives.items():
            results.append(objective(prediction, goal))
        return sum(stack(results), dim=0)


class LPIPSError(Objective):
    def __init__(self, weight, key):
        super().__init__(weight)
        self.lpips = piq.LPIPS(reduction="none")
        self.key = key

    def flatten_image(self, im):
        leading_dims = im.shape[:-3]
        im = im.reshape(-1, *im.shape[-3:])
        return im, leading_dims

    def compute_reward(self, prediction, goal):
        prediction = prediction[self.key]
        goal = np.repeat(goal[self.key][None], prediction.shape[0], axis=0)
        goal = torch.tensor(goal).float().cuda()
        goal = torch.moveaxis(goal, -1, -3)
        prediction = torch.tensor(prediction).float().cuda()
        prediction = torch.moveaxis(prediction, -1, -3)
        lpips = []
        with torch.no_grad():
            for t in range(prediction.shape[1]):
                lpips_t = self.lpips(prediction[:, t], goal[:, t])
                lpips.append(lpips_t[..., None])
        lpips = torch.stack(lpips, dim=1)
        # [B, T, 1]
        lpips = lpips.mean(dim=(-1, -2), keepdim=True).cpu().detach().numpy()
        return -lpips


class ClassifierReward(Objective):
    def __init__(
        self,
        checkpoint_directory,
        weight,
        key,
        max_batch_size=1024,
        use_probs=False,
        use_gpu=True,
    ):
        """
        :param checkpoint_directory: directory containing classifier checkpoint
        :param weight: weight on this cost
        :param key: key containing image to classify (e.g. "rgb")
        :param max_batch_size: maximum batch size for each classifier forward pass.
        If the input has more samples than max_batch_size, the samples are split into
        batches of at most size max_batch_size.
        :param use_probs: if True, use sigmoid(logits) as the score, otherwise, directly use the logits
        :param use_gpu:
        """
        super().__init__(weight)
        self.checkpoint_directory = to_absolute_path(checkpoint_directory)
        self.key = key
        self.max_batch_size = max_batch_size
        self.use_probs = use_probs
        self.model = ConvPredictor()
        self.use_gpu = use_gpu
        print(f"Loading classifier reward predictor from {self.checkpoint_directory}")
        self.model.load_state_dict(torch.load(self.checkpoint_directory))
        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

    def compute_reward(self, prediction, goal):
        prediction = torch.tensor(prediction[self.key], dtype=torch.float32)
        if self.use_gpu:
            prediction = prediction.cuda()
        flattened_predictions = prediction.reshape(-1, *prediction.shape[-3:])
        # convert from BHWC to BCHW
        flattened_predictions = flattened_predictions.permute(0, 3, 1, 2)
        scores = []
        num_batches = int(np.ceil(flattened_predictions.shape[0] / self.max_batch_size))
        with torch.no_grad():
            for batch_num in range(num_batches):
                logits = self.model(
                    flattened_predictions[
                        batch_num
                        * self.max_batch_size : (batch_num + 1)
                        * self.max_batch_size
                    ]
                )
                if self.use_probs:
                    score = torch.sigmoid(logits)
                else:
                    score = logits
                scores.append(score)
        scores = torch.cat(scores, dim=0)
        scores = scores.view(prediction.shape[0], prediction.shape[1])
        scores = scores.sum(dim=1)
        scores = scores.cpu().numpy()
        return np.expand_dims(scores, (1, 2))


class PolicyFeatureDistance(Objective):
    def __init__(self, policy_feature_metrics, weight, image_key="rgb"):
        super().__init__(weight)
        if not isinstance(policy_feature_metrics, torch.nn.ModuleList):
            policy_feature_metrics = [policy_feature_metrics]
        self.policy_feature_metrics = policy_feature_metrics
        self.image_key = image_key

    def compute_features(self, imgs):
        """
        :param imgs: tensor of shape [..., C, H, W]
        :return: tensor of shape [..., D] where D is the shape of the feature dimension
        """
        imgs = torch.tensor(np.moveaxis(imgs, -1, -3)).float().cuda()
        features = [
            f.get_feature_activations(imgs) for f in self.policy_feature_metrics
        ]
        features = torch.cat(features, dim=-1)
        return features

    def compute_reward(self, prediction, goal):
        with torch.no_grad():
            pred_feats, goal_feats = (
                self.compute_features(prediction[self.image_key]),
                self.compute_features(goal[self.image_key]),
            )
            # both shapes are [B, T, D]
            cost = (pred_feats - goal_feats) ** 2
            # sum works much better than mean -- mean has small magnitudes (and floating point errors?)
            cost = cost.detach().cpu().numpy()
        reward = -np.sum(cost, axis=(1, 2))
        return np.expand_dims(reward, (1, 2))


class EnsembleObjective(Objective):

    # Compute objectives across an ensemble of models and aggregate.
    def __init__(self, objective, agg="mean", weight=1.0, lamb=0.0):
        super().__init__(weight=weight)
        self.objective = objective
        assert agg in [
            "mean",
            "min",
            "penalize_disagreement",
        ], "Only mean, min objective aggregation are supported!"
        self.agg = agg
        self.lamb = lamb

    def compute_reward(self, prediction, goal):
        rewards = list()
        ensemble_count = prediction[list(prediction.keys())[0]].shape[0]
        for i in range(ensemble_count):
            rew = self.objective.compute_reward(
                slice_dict(prediction, i, i + 1, squeeze=True), goal
            )
            rewards.append(rew)
        rewards = np.stack(rewards, axis=0)
        if self.agg == "mean":
            rewards = np.mean(rewards, axis=0)
        elif self.agg == "min":
            rewards = np.amin(rewards, axis=0)
        elif self.agg == "penalize_disagreement":
            # rewards = np.mean(rewards, axis=0)
            indices = np.random.randint(0, ensemble_count, size=(rewards.shape[1]))
            rewards = rewards[indices, np.arange(rewards.shape[1])]
            for key in prediction:
                mean = np.mean(prediction[key], axis=0)
                disagreements = np.abs(prediction[key] - mean).sum(axis=(2, 3, 4, 5))
                disagreements = np.amax(disagreements, axis=0)
                rewards -= self.lamb * np.expand_dims(disagreements, (1, 2))
        return rewards
