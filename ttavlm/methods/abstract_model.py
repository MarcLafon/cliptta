from typing import Any, Dict, Union, Iterable, Tuple, List, Optional
from typing_extensions import TypeAlias

from wandb.wandb_run import Run
from wandb_osh.hooks import TriggerWandbSyncHook

from copy import deepcopy

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian as torch_jacobian
import numpy
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from scipy.stats import chisquare
from skimage.filters import threshold_otsu as sk_otsu
from sklearn.manifold import TSNE

from ttavlm.optimizers import SAM
import ttavlm.lib as lib

Kwargs: TypeAlias = Dict[str, Any]
ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class AbstractOpenSetTTAModel(nn.Module):
    """
    Doc
    """

    def __init__(
        self,
        save_root: str,
        adaptation: str,
        model: nn.Module,
        clip_text_encoder: nn.Module,
        class_prototypes: Tensor,  # For clip it will be text_embeddings
        class_bias: Tensor,  # For clip it will be zero
        normalize_features: bool = False,
        update_text: bool = False,
        update_all_params: bool = False,
        optimizer_type: str = "sgd",
        steps: int = 1,
        episodic: bool = False,
        logit_scale: float = 100.0,
        id_score_type: str = "max_prob",
        ood_logit_scale: float = 1.0,
        use_ood_loss: bool = False,
        detect_ood: bool = False,
        score_type: str = "max_prob",
        loss_ood_type: str = "max_inter_var",
        use_weights: bool = False,
        update_alpha: bool = True,
        update_alpha_miss: bool = False,
        alpha: float = 0.5,
        beta_tta: float = 1.0,
        beta_reg: float = 0.1,
        beta_ood: float = 0.1,
        beta_miss: float = 0.0,
        gamma: float = 0.005,
        beta_schedule: str = "sequential",
        milestone: int = 2,
        lr: float = 1e-3,
        lr_miss: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        use_sam: bool = False,
        skip_top_layers: bool = False,
        max_iter: int = 79,
        use_batch_stats_only: bool = False,
        tta_reduction: str = "probs",
        negative_prototypes: Tensor = None,
        negative_bias: Tensor = None,
        distributed: bool = False,
        num_shots: int = 4,
        sample_size: int = 256,
        tsne: bool = False,
        measure_collapse: bool = False,
        measure_improvement: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_root = save_root
        self.adaptation = adaptation
        self.model = model
        self.clip_text_encoder = clip_text_encoder
        self.class_prototypes = class_prototypes
        self.class_bias = class_bias
        self.normalize_features = normalize_features
        self.steps = steps
        assert steps > 0, "Number of steps for adaptation must be >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.logit_scale = logit_scale
        self.id_score_type = id_score_type

        # OOD detection attributes
        self.use_ood_loss = use_ood_loss
        self.detect_ood = detect_ood
        self.score_type = score_type
        self.ood_logit_scale = ood_logit_scale
        self.loss_ood_type = loss_ood_type
        self.use_weights = use_weights
        self.update_alpha = update_alpha
        self.update_alpha_miss = update_alpha_miss
        self.alpha = nn.Parameter(torch.tensor([alpha]), requires_grad=update_alpha)
        self.alpha_miss = nn.Parameter(torch.tensor([alpha]), requires_grad=update_alpha_miss)
        self.beta_tta = beta_tta
        self.beta_reg = beta_reg
        self.beta_ood = beta_ood
        self.beta_miss = beta_miss
        self.gamma = gamma
        self.beta_schedule = beta_schedule
        self.milestone = milestone
        self.tta_reduction = tta_reduction
        self.negative_prototypes = negative_prototypes
        self.negative_bias = negative_bias
        self.tsne = tsne
        self.measure_collapse = measure_collapse
        self.measure_improvement = measure_improvement

        # Memory attributes
        self.num_shots = num_shots
        self.sample_size = sample_size

        # optimization attributes
        self.update_text = update_text
        assert self.clip_text_encoder is not None, "clip text encoder is required when update_text is True"
        self.update_all_params = update_all_params
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_miss = lr_miss
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_sam = use_sam
        self.skip_top_layers = skip_top_layers
        self.use_batch_stats_only = use_batch_stats_only
        self.max_iter = max_iter

        params = self.collect_params()
        self.configure_model()
        self.optimizer = self.setup_optimizer(params)

        self.distributed = distributed

        if self.distributed:
            self.model = lib.DataParallel(self.model)
            if self.update_text:
                self.clip_text_encoder = lib.DataParallel(self.clip_text_encoder)

        if self.update_alpha:
            self.optimizer.add_param_group({"params": self.alpha})
        if self.update_alpha_miss:
            self.optimizer.add_param_group({"params": self.alpha_miss, "lr": self.lr_miss})
        self.model_state = self.copy_state(self.model)
        self.optimizer_state = self.copy_state(self.optimizer)
        if self.update_text:
            self.clip_text_encoder_state = self.copy_state(self.clip_text_encoder)

    def get_features(
        self,
        images: List[Tensor],
        model: nn.Module = None,
    ) -> List[Tensor]:
        if model is None:
            model = self.model

        image_features = []
        for img in images:
            if hasattr(model, "proj"):
                img_features = model(img.type(model.dtype))
            else:
                img_features = model.forward_features(img.type(model.dtype))
                img_features = model.global_pool(img_features)
                img_features = img_features.view(img_features.size(0), -1)
            if self.normalize_features:
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            image_features.append(img_features)

        return image_features

    def get_logits(
        self,
        image_features: List[Tensor],
        class_prototypes: Tensor = None,
        class_bias: Tensor = None,
    ) -> List[Tensor]:
        class_prototypes = self.class_prototypes if class_prototypes is None else class_prototypes
        class_bias = self.class_bias if class_bias is None else class_bias

        logits = [img_features @ class_prototypes.t() + class_bias for img_features in image_features]

        return logits

    def forward(
        self,
        images: List[Tensor],
        labels: Tensor = None,
        **kwargs: Kwargs,
    ) -> Tuple[List[Tensor], Tensor]:
        if self.episodic:
            self.reset()

        self.before_adaptation(images)

        for step in range(self.steps):
            logits, scores = self.forward_and_adapt(images, step, labels)

        self.after_adaptation()

        return logits, scores

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        images: List[Tensor],
        step: int,
        labels: Tensor = None,
    ) -> Tuple[List[Tensor], Tensor]:
        pass

    def before_adaptation(
        self,
        images: List[Tensor],
        **kwargs: Kwargs,
    ):  # noqa: ANN201
        if self.use_ood_loss:
            self.init_otsu(images)
        else:
            pass

    def after_adaptation(
        self,
        **kwargs: Kwargs,
    ):  # noqa: ANN201
        pass

    def get_results(
        self,
        main_loader: DataLoader,
        ood_loader: Optional[DataLoader] = None,
        run_wandb: Optional[Run] = None,
        trigger_sync: Optional[TriggerWandbSyncHook] = None,
        display_progress: Optional[bool] = False,
    ) -> Tuple[float]:
        meter = lib.DictAverage()

        acc = 0.0
        good_bad_after = 0.0
        bad_good_after = 0.0
        n_bad_grad_pred = 0.0
        n_bad_grad_label = 0.0
        n_bad_grad_label_pred = 0.0
        n_bad_argmax_is_label = 0.0
        n_bad_argmax_is_pred = 0.0
        n_bad_initial = 1e-6
        n_good_initial = 1e-6
        score_id, score_ood, pred, labels_id = [], [], [], []
        ood_iter = iter(ood_loader) if ood_loader is not None else None
        method_name = self.__class__.__name__
        dataset_name = main_loader.dataset.__class__.__name__[:-7]
        shift_type = main_loader.dataset.shift_type
        count = 0
        closed_set = ood_loader is None
        track_loader = lib.track(main_loader, f"{method_name} running on {dataset_name} / {shift_type}")
        progress = lib.ProgressMeter(len(main_loader), meter, prefix=f"{method_name} running on {dataset_name} / {shift_type}")
        embeddings_before = [] if self.tsne else None
        embeddings_after = [] if self.tsne else None
        labels_ood = [] if self.tsne else None
        indexer = {}
        with torch.no_grad():
            for n, batch in enumerate(track_loader):
                images = [img.cuda(non_blocking=True) for img in batch["image"]]
                labels = batch["target"].cuda(non_blocking=True)
                n_id = labels.shape[0]

                if closed_set:
                    all_images = images
                else:
                    try:
                        batch_ood = next(ood_iter)
                    except StopIteration:
                        ood_iter = iter(ood_loader)
                        batch_ood = next(ood_iter)
                    images_ood = [img.cuda(non_blocking=True) for img in batch_ood["image"]]
                    all_images = [torch.cat((img, img_ood), dim=0) for img, img_ood in zip(images, images_ood)]

                labels_id.append(labels)
                # Collecting visual features before adaptation
                if self.tsne:
                    # labels_ood += labels.tolist()
                    labels_ood += [1] * images[0].shape[0]
                    labels_ood += [-1] * images_ood[0].shape[0]
                    image_features_before = self.get_features(all_images)
                    embeddings_before.append(image_features_before[0].detach().cpu())

                if self.measure_improvement:
                    # Get predictions before
                    indices = torch.arange(n_id).to(images[0].device)
                    image_features_before = self.get_features(all_images, model=self.model0)
                    logits_before = self.get_logits(image_features_before)
                    probs_before = [(self.logit_scale * lg).softmax(dim=-1) for lg in logits_before]
                    probs_before = torch.stack(probs_before, dim=0).mean(dim=0)
                    _, pred_before = probs_before.max(dim=1)

                    # Compute the gradient for each examples
                    if n in [25]:
                        indexer[n] = []
                        grad_stats = self.gradients_metrics(image_features_before[0], pred_before[:n_id], labels, indexer, n)
                        n_bad_grad_pred += grad_stats[0]
                        n_bad_grad_label += grad_stats[1]
                        n_bad_grad_label_pred += grad_stats[2]
                        n_bad_argmax_is_label += grad_stats[3]
                        n_bad_argmax_is_pred += grad_stats[4]

                        if run_wandb is not None:
                            run_wandb.log(
                                {
                                    "Bad_Grad_pred_aligned": n_bad_grad_pred,
                                    "Bad_Grad_label_aligned": n_bad_grad_label,
                                    "Bad_Grad_label_pred": n_bad_grad_label_pred,
                                    "Bad_argmax_is_label": n_bad_argmax_is_label,
                                    "Bad_argmax_is_pred": n_bad_argmax_is_pred,
                                },
                                step=n,
                            )

                    wrong_samples_before = indices[~(pred_before[:n_id] == labels)]
                    good_samples_before = indices[(pred_before[:n_id] == labels)]
                    n_good_before = good_samples_before.shape[0]
                    n_wrong_before = wrong_samples_before.shape[0]
                    n_bad_initial += n_wrong_before
                    n_good_initial += n_good_before



                if self.adaptation == "zero":
                    logits, scores, p_bar = self.forward_and_adapt(all_images, step=0, labels=labels)
                else:
                    logits, scores = self.forward(all_images, labels=labels)

                # Measuring class collapse
                if self.measure_collapse:
                    self.collapse_metric(logits, n, run_wandb)

                # Collecting visual features after adaptation
                if self.tsne:
                    image_features_after = self.get_features(all_images)
                    embeddings_after.append(image_features_after[0].detach().cpu())

                if self.adaptation == "zero":
                    _, pred_ = p_bar.max(dim=-1)
                else:
                    if self.tta_reduction == "logits":
                        logits = [(self.logit_scale * lg) for lg in logits]
                        logits = torch.stack(logits, dim=0).mean(dim=0)
                        probs = logits.softmax(dim=-1)
                    elif self.tta_reduction == "probs":
                        probs = [(self.logit_scale * lg).softmax(dim=-1) for lg in logits]
                        probs = torch.stack(probs, dim=0).mean(dim=0)
                    else:
                        raise NotImplementedError
                    _, pred_ = probs.max(dim=-1)

                if self.measure_improvement:
                    wrong_samples_after = indices[~(pred_[:n_id] == labels)]
                    good_samples_after = indices[(pred_[:n_id] == labels)]
                    good_bad_after += torch.isin(good_samples_before, wrong_samples_after).sum()
                    bad_good_after += torch.isin(wrong_samples_before, good_samples_after).sum()
                    if run_wandb is not None:
                        run_wandb.log({'Bad_good_after': bad_good_after}, step=n)
                        run_wandb.log({'Good_bad_after': good_bad_after}, step=n)
                        run_wandb.log({"bad_good_ratio": bad_good_after / n_bad_initial,
                                       "good_bad_ratio": good_bad_after / n_good_initial}, step=n)

                # Accuracy, for classification performance
                acc += (pred_[:n_id] == labels).float().sum()
                count += n_id
                log_dict = {
                    "acc": acc.item() / count,
                    "batch_acc": (pred_[:n_id] == labels).float().sum().item() / n_id,
                }
                meter.update(
                    {
                        "acc": (pred_[:n_id] == labels).float().sum().item() / n_id,
                    }
                )

                if not closed_set:
                    # OOD scores, for Open-Set detection
                    score_id.append(scores[:n_id])
                    score_ood.append(scores[n_id:])
                    pred.append(pred_[:n_id])

                    meter.update(
                        {
                            "running_auc": lib.get_auroc(scores[:n_id].cpu().numpy(), scores[n_id:].cpu().numpy()),
                            "running_fpr": lib.get_fpr(scores[:n_id].cpu().numpy(), scores[n_id:].cpu().numpy()),
                        }
                    )
                    log_dict["running_auc"] = meter["running_auc"].avg
                    log_dict["running_fpr"] = meter["running_fpr"].avg

                track_loader.set_postfix(log_dict)
                # Logging in W&B
                if run_wandb is not None:
                    run_wandb.log(log_dict, step=n)
                    # trigger_sync(logdir=run_wandb.dir)

                if n % 10 == 0 and display_progress:
                    progress.display(n)

        if not closed_set:
            labels_id = torch.cat(labels_id).cpu().numpy()
            score_id = torch.cat(score_id).detach().cpu().numpy()
            score_ood = torch.cat(score_ood).detach().cpu().numpy()
            pred = torch.cat(pred).detach().cpu().numpy()

        # Compute metrics
        accuracy = acc.item() / len(main_loader.dataset)
        if closed_set:
            auc, fpr, oscr = 0.0, 0.0, 0.0
        else:
            auc = lib.get_auroc(score_id, score_ood)
            fpr = lib.get_fpr(score_id, score_ood)
            oscr = lib.get_oscr(score_id, score_ood, pred, labels_id)

            # Plotting t-SNE of visual embeddings
            if self.tsne:
                self.plot(embeddings_before, embeddings_after, labels_ood)
                b_embeddings = torch.cat(embeddings_before, dim=0)
                a_embeddings = torch.cat(embeddings_after, dim=0)
                tsne_info = {'before': b_embeddings, 'after': a_embeddings, 'labels': labels_ood}
                torch.save(tsne_info, os.path.join(self.save_root, 'plots', 'tsne_' + self.adaptation + '_results.pth'))

        return accuracy, auc, fpr, oscr

    def init_otsu(
        self,
        images: List[Tensor],
    ) -> None:
        # Initializing threshold from batch predictions
        with torch.no_grad():
            image_features = self.get_features(images)
            logits = self.get_logits(image_features)
            scores = self.get_scores(logits, image_features)
            np_scores = scores.detach().cpu().numpy()
            threshold = torch.tensor([sk_otsu(np_scores)], requires_grad=self.update_alpha)
            threshold = threshold.to(images[0].device)
            self.alpha.data = threshold

    def get_scores(
        self,
        logits: List[Tensor],
        image_features: Tensor = None,
        labels: Tensor = None,
        score_type: Optional[str] = None,
    ) -> Tensor:
        """
        The incoming data maybe a list of tensors in case we are using Test-Time Augmentation techniques,
        In that case we should specify a reduction mechanism to compute the ood score. For the `max_logit` and `logsumexp` cases
        a simple average over the logits should be sufficient. In the cases whe have to compute probability vectors like for `max_prob`, `entropy` and `tcp`
        we have two options:
                1) averaging the logits and then compute the probability vector,
                2) computing each probability vector and then average them (stamp does this).
        We will stick with the way stamp does it to keep it simple.

        The code is refactored to treat both cases (Tensor and List[Tensor])
        """

        logits = [lg * self.ood_logit_scale for lg in logits]
        if score_type is None:
            score_type = self.score_type

        if score_type == "max_logit":
            logits = torch.stack(logits, dim=0).mean(dim=0)
            scores = logits.max(1).values
        elif score_type == "max_prob":
            probs = [lg.softmax(dim=-1) for lg in logits]
            probs = torch.stack(probs, dim=0).mean(dim=0)
            scores = probs.max(dim=1).values
        elif score_type == "max_log_prob":
            log_probs = [lg.softmax(dim=-1).log() for lg in logits]
            log_probs = torch.stack(log_probs, dim=0).mean(dim=0)
            scores = log_probs.max(dim=1).values
        elif score_type == "logsumexp":
            logits = torch.stack(logits, dim=0).mean(dim=0)
            scores = logits.logsumexp(dim=1)
        elif score_type == "entropy":
            probs = [lg.softmax(dim=-1) for lg in logits]
            probs = torch.stack(probs, dim=0).mean(dim=0)
            scores = -lib.entropy(probs)
        elif score_type == "sim":
            scores = ((image_features @ image_features.t()) * (1 - torch.eye(image_features.shape[0]).to(image_features.device))).mean(1)
        elif score_type == "tcp":
            batch_size = int(logits.shape[0] / 2)
            logits = torch.stack(logits, dim=0).mean(dim=0)
            id_logits = logits[:batch_size]
            ood_logits = logits[batch_size:]
            id_scores = id_logits[torch.arange(batch_size), labels]
            ood_logits = ood_logits.max(1).values
            scores = torch.cat([id_scores, ood_logits])
        elif score_type == "neglabel":
            neg_logits = self.get_logits(image_features, self.negative_prototypes, self.negative_bias)[0] * self.ood_logit_scale
            probs = torch.cat([logits[0], neg_logits], dim=1).softmax(dim=-1)
            scores = probs[:, : self.class_prototypes.shape[0]].max(dim=1).values
            # scores = probs[:,:self.class_prototypes.shape[0]].sum(dim=1)
        elif score_type == "clipartt":
            values, pred = logits[0].topk(1, 1, True, True)
            text_features = self.class_prototypes[pred[:, 0]]
            img_sim = (image_features @ image_features.t()) * (1 - torch.eye(image_features.shape[0]).to(image_features.device))
            txt_sim = (text_features @ text_features.t()).mean(1)
            scores = -((img_sim + txt_sim) / 2.0).mean(1)

        return scores

    def get_otsu_loss(
        self,
        scores: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        w = torch.sigmoid((scores - self.alpha) / self.gamma)
        loss = self.otsu_loss(w, scores)

        return loss

    def otsu_loss(self, w: Tensor, scores: Tensor) -> Tensor:
        p_in, p_out = w.mean(), (1 - w).mean()
        mu_in = torch.dot(w, scores) / torch.sum(w)
        mu_out = torch.dot(1 - w, scores) / torch.sum(1 - w)
        if self.loss_ood_type == "max_inter_var":
            loss_ood = -p_in * p_out * (mu_in - mu_out).pow(2)
        elif self.loss_ood_type == "min_intra_var":
            loss_ood = -p_out * mu_out.pow(2) - p_in * mu_in.pow(2)
        elif self.loss_ood_type == "avg_contrastive":
            loss_ood = -(mu_in - mu_out).pow(2)
        else:
            raise NotImplementedError

        return loss_ood

    def filter_id(
        self,
        images: List[Tensor],
        image_features: List[Tensor],
        scores: Tensor,
        threshold: Optional[nn.Parameter] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # If threshold is passed (alpha in Otsu method) do nothing, else compute it with otsu
        if threshold is None:
            threshold = torch.tensor([sk_otsu(scores.detach().cpu().numpy())], requires_grad=False)

        threshold = threshold.to(scores.device)
        id_images = images[0][scores > threshold]
        id_image_features = image_features[0][scores > threshold]

        return [id_images], [id_image_features]

    def update_betas(self, step: int) -> None:
        if self.beta_schedule == "sequential":
            self.beta_tta = 0.1 + (0.9 - 0.1) * (numpy.exp(step / (self.steps - 1)) - 1) / (numpy.exp(1) - 1)
            self.beta_ood = 0.9 - (0.9 - 0.1) * (numpy.exp(step / (self.steps - 1)) - 1) / (numpy.exp(1) - 1)
        elif self.beta_schedule == "step":
            if step > self.milestone:
                self.beta_tta = 1.0
                self.beta_ood = 0.0
                self.beta_reg = 1.0
        elif self.beta_schedule == "none":
            pass
        else:
            raise NotImplementedError

    def _reset(self) -> None:
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        if self.update_text:
            self.clip_text_encoder.load_state_dict(self.clip_text_encoder_state, strict=True)

    def _reset_extra(self) -> None:
        """Use this to reset any additionnal model or optimizer your specfic tta method needs"""
        pass

    def reset(self) -> None:
        self._reset()
        self._reset_extra()

    def copy_state(
        self,
        model_or_optimizer: Union[nn.Module, Optimizer],
    ) -> None:
        """Copy the model or optimizer states for resetting after adaptation."""
        model_or_optimizer_state = deepcopy(model_or_optimizer.state_dict())
        return model_or_optimizer_state

    def collect_params(self) -> Tuple[List[Parameter], List[str]]:
        """Collect the affine scale + shift parameters from normalization layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        """
        params = []
        if not self.update_all_params:
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    if self.skip_top_layers:
                        if any(subnm in nm for subnm in ["layer4", "conv5_x", "blocks.9", "blocks.10", "blocks.11", "ln_post"]):
                            lib.LOGGER.info(f"Skipping {nm} params.")
                            continue
                        if nm in ["norm"]:
                            lib.LOGGER.info(f"Skipping {nm} params.")
                            continue
                    for np, p in m.named_parameters():
                        if np in ["weight", "bias"]:  # weight is scale, bias is shift
                            params.append(p)
        else:
            for np, p in self.model.named_parameters():
                params.append(p)

        if self.update_text:
            lib.LOGGER.info("Adding text encoder's parameters to params")
            if not self.update_all_params:
                for nm, m in self.clip_text_encoder.named_modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                        for np, p in m.named_parameters():
                            if np in ["weight", "bias"]:  # weight is scale, bias is shift
                                params.append(p)
            else:
                for np, p in self.clip_text_encoder.named_parameters():
                    params.append(p)

        return params

    def configure_model(self) -> None:
        """Set model's parameters requires_grad to True for adaptation"""
        self.model.train()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                m.requires_grad_(True)
            elif self.update_all_params:
                m.requires_grad_(True)

        if self.update_text:
            lib.LOGGER.info("Configuring text encoder's layers")
            self.clip_text_encoder.train()
            # disable grad, to (re-)enable only what we update
            self.clip_text_encoder.requires_grad_(False)
            for m in self.clip_text_encoder.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.requires_grad_(True)
                elif self.update_all_params:
                    m.requires_grad_(True)

    def setup_optimizer(self, params: List[Parameter]) -> Optimizer:
        if self.optimizer_type.lower() == "adam":
            if self.use_sam:
                optimizer = SAM(params, torch.optim.Adam, lr=self.lr, weight_decay=self.weight_decay, eps=1e-6)
            else:
                optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-6)
        elif self.optimizer_type.lower() == "adamw":
            if self.use_sam:
                optimizer = SAM(params, torch.optim.AdamW, lr=self.lr, weight_decay=self.weight_decay, eps=1e-6)
            else:
                optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-6)

        elif self.optimizer_type.lower() == "sgd":
            if self.use_sam:
                optimizer = SAM(
                    params,
                    torch.optim.SGD,
                    lr=self.lr,
                    momentum=self.momentum,
                )
            else:
                optimizer = torch.optim.SGD(params, self.lr, momentum=self.momentum)
        else:
            raise NotImplementedError

        return optimizer

    def plot(self,
             embeddings_before: List[Tensor],
             embeddings_after: List[Tensor],
             labels_ood: List,
             ) -> None:
        labels_ood = numpy.array(labels_ood)
        embeddings_before = torch.cat(embeddings_before).detach().cpu().numpy()
        embeddings_after = torch.cat(embeddings_after).detach().cpu().numpy()
        tsne_before = TSNE(n_components=2).fit_transform(embeddings_before)
        tsne_after = TSNE(n_components=2).fit_transform(embeddings_after)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        tsne_before_ood = tsne_before[labels_ood == -1]
        tsne_before_id = tsne_before[labels_ood != -1]
        tsne_after_ood = tsne_after[labels_ood == -1]
        tsne_after_id = tsne_after[labels_ood != -1]
        labels_good = labels_ood[labels_ood != -1]

        ax1.scatter(tsne_before_ood[:, 0], tsne_before_ood[:, 1], marker='.', s=10, c='purple', label='OOD samples')
        ax1.scatter(tsne_before_id[:, 0], tsne_before_id[:, 1], marker='.', s=10, c=labels_good)
        ax1.set_ylabel('Before adaptation')
        ax2.scatter(tsne_after_ood[:, 0], tsne_after_ood[:, 1], marker='.', s=10, c='purple', label='OOD samples')
        ax2.scatter(tsne_after_id[:, 0], tsne_after_id[:, 1], marker='.', s=10, c=labels_good)
        ax2.set_ylabel('After adaptation')
        ax2.yaxis.set_label_position("right")
        ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax1.legend()
        ax2.legend()
        save_path = os.path.join(self.save_root, 'plots')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'tsne_' + self.adaptation + '.png'), bbox_inches='tight', dpi=300, transparent=True)
        fig.savefig(os.path.join(save_path, 'tsne_' + self.adaptation + '_before.png'),
                    bbox_inches='tight',
                    pad_inches=0.1,
                    transparent=True)
        extent_ax1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(save_path, 'tsne_' + self.adaptation + '_before.png'), bbox_inches=extent_ax1, transparent=True, dpi=300)
        extent_ax2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(save_path, 'tsne_' + self.adaptation + '_after.png'), bbox_inches=extent_ax2, transparent=True, dpi=300)

    def collapse_metric(self, logits: Tensor, step: int, logger: Optional[Run]) -> None:
        n_classes = logits[0].shape[1]
        with torch.no_grad():
            mean_prob = (logits[0] + 1e-6).softmax(1).mean(0)
        uniform = (torch.ones(n_classes) / n_classes).to(logits[0].device)

        # Cross-entropy between distributions
        ent_metric = lib.softmax_mean_entropy(logits[0] * self.logit_scale)
        ce_metric = F.cross_entropy(mean_prob, uniform)

        # Chi-2 between distributions
        mean_prob = mean_prob.cpu().detach().numpy()
        uniform = uniform.cpu().detach().numpy()
        chi_metric = chisquare(mean_prob).statistic

        logger.log({'Collapse CE': ce_metric}, step=step)
        logger.log({'Collapse Entropy': ent_metric}, step=step)
        logger.log({'Collapse Chi': chi_metric}, step=step)

    def gradients_metrics(self, image_features: Tensor, preds: Tensor, labels: Tensor, indexer: Dict, step=int) -> Tuple[Tensor, Tensor, Tensor]:
        pred_text_features = self.class_prototypes[preds]
        label_text_features = self.class_prototypes[labels]

        with torch.no_grad():
            gradients = torch_jacobian(self.compute_loss, image_features)

        # Compute scalar product between gradient and each class, take argmax:

        argmax_grad_dir = (-gradients @ self.class_prototypes.t()).argmax(dim=-1)

        bad_grad_pred = 0
        bad_grad_label = 0
        bad_grad_label_pred = 0
        bad_argmax_is_label = 0
        bad_argmax_is_pred = 0

        for i in range(preds.shape[0]):
            if preds[i] != labels[i]:
                if gradients[i] @ pred_text_features[i].t() < 0.0:
                    bad_grad_pred += 1

                if gradients[i] @ label_text_features[i].t() < 0.0:
                    bad_grad_label += 1

                if (gradients[i] @ label_text_features[i].t()) < (gradients[i] @ pred_text_features[i].t()) < 0.0:
                    indexer[step].append(i)
                    bad_grad_label_pred += 1

                if argmax_grad_dir[i] == labels[i]:
                    bad_argmax_is_label += 1

                if argmax_grad_dir[i] == preds[i]:
                    bad_argmax_is_pred += 1

        return bad_grad_pred, bad_grad_label, bad_grad_label_pred, bad_argmax_is_label, bad_argmax_is_pred
