# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

from data.ag import AG, cuda_collate_fn, ag_get_statistics
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_mR_from_evaluator_list,
)
from lib.pytorch_misc import argsort_desc
from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentorNoCrop,
)
from model.egtr import DetrForSceneGraphGeneration
from util.box_ops import rescale_bboxes
from util.misc import use_deterministic_algorithms
from sklearn.model_selection import train_test_split

seed_everything(42, workers=True)

# Reference: https://github.com/yuweihao/KERN/blob/master/models/eval_rels.py
def evaluate_batch(
    outputs,
    targets,
    multiple_sgg_evaluator,
    multiple_sgg_evaluator_list,
    single_sgg_evaluator,
    single_sgg_evaluator_list,
    num_labels,
    max_topk=100,
):
    for j, target in enumerate(targets):
        # Pred
        pred_logits = outputs["logits"][j]
        obj_scores, pred_classes = torch.max(
            pred_logits.softmax(-1)[:, :num_labels], -1
        )
        sub_ob_scores = torch.outer(obj_scores, obj_scores)
        sub_ob_scores[
            torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
        ] = 0.0  # prevent self-connection

        pred_boxes = outputs["pred_boxes"][j]
        pred_rel = torch.clamp(outputs["pred_rel"][j], 0.0, 1.0)
        if "pred_connectivity" in outputs:
            pred_connectivity = torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0)
            pred_rel = torch.mul(pred_rel, pred_connectivity)

        # GT
        orig_size = target["orig_size"]
        target_labels = target["class_labels"]  # [num_objs]
        target_boxes = target["boxes"]  # [num_objs, 4]
        target_rel = target["rel"].nonzero()  # [num_rels, 3(s, o, p)]

        gt_entry = {
            "gt_relations": target_rel.clone().numpy(),
            "gt_boxes": rescale_bboxes(target_boxes, torch.flip(orig_size, dims=[0]))
            .clone()
            .numpy(),
            "gt_classes": target_labels.clone().numpy(),
        }

        if multiple_sgg_evaluator is not None:
            triplet_scores = torch.mul(pred_rel, sub_ob_scores.unsqueeze(-1))
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 3(s,o,p)]
            rel_scores = (
                pred_rel.cpu()
                .clone()
                .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
            )  # [pred_rels]

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }
            multiple_sgg_evaluator["sgdet"].evaluate_scene_graph_entry(
                gt_entry, pred_entry
            )

            for pred_id, _, evaluator_rel in multiple_sgg_evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                if gt_entry_rel["gt_relations"].shape[0] == 0:
                    continue
                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                    gt_entry_rel, pred_entry
                )

        if single_sgg_evaluator is not None:
            triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
            pred_rel_inds = argsort_desc(triplet_scores.cpu().clone().numpy())[
                :max_topk, :
            ]  # [pred_rels, 2(s,o)]
            rel_scores = (
                pred_rel.cpu().clone().numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1]]
            )  # [pred_rels, 50]

            pred_entry = {
                "pred_boxes": rescale_bboxes(
                    pred_boxes.cpu(), torch.flip(orig_size, dims=[0])
                )
                .clone()
                .numpy(),
                "pred_classes": pred_classes.cpu().clone().numpy(),
                "obj_scores": obj_scores.cpu().clone().numpy(),
                "pred_rel_inds": pred_rel_inds,
                "rel_scores": rel_scores,
            }

            print("gt_entry : ", gt_entry)
            print("pred_entry : ", pred_entry)
            single_sgg_evaluator["sgdet"].evaluate_scene_graph_entry(
                gt_entry, pred_entry
            )
            for pred_id, _, evaluator_rel in single_sgg_evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel["gt_relations"][:, -1], pred_id)
                gt_entry_rel["gt_relations"] = gt_entry_rel["gt_relations"][mask, :]
                if gt_entry_rel["gt_relations"].shape[0] == 0:
                    continue
                evaluator_rel["sgdet"].evaluate_scene_graph_entry(
                    gt_entry_rel, pred_entry
                )



class SGG(pl.LightningModule):
    def __init__(
        self,
        architecture,
        backbone_dirpath,
        auxiliary_loss,
        lr,
        lr_backbone,
        lr_initialized,
        weight_decay,
        pretrained,
        main_trained,
        from_scratch,
        id2label,
        rel_loss_coefficient,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        rel_categories,
        multiple_sgg_evaluator,
        multiple_sgg_evaluator_list,
        single_sgg_evaluator,
        single_sgg_evaluator_list,
        coco_evaluator,
        feature_extractor,
        num_queries,
        ce_loss_coefficient,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
        use_freq_bias,
        fg_matrix,
        use_log_softmax,
        freq_bias_eps,
        connectivity_loss_coefficient,
        logit_adjustment,
        logit_adj_tau,
    ):

        super().__init__()
        # replace COCO classification head with custom head
        config = DeformableDetrConfig.from_pretrained(pretrained)
        config.architecture = architecture
        config.auxiliary_loss = auxiliary_loss
        config.from_scratch = from_scratch
        config.num_rel_labels = len(rel_categories)
        config.num_labels = max(id2label.keys()) + 1
        config.num_queries = num_queries
        config.rel_loss_coefficient = rel_loss_coefficient
        config.smoothing = smoothing
        config.rel_sample_negatives = rel_sample_negatives
        config.rel_sample_nonmatching = rel_sample_nonmatching
        config.ce_loss_coefficient = ce_loss_coefficient
        config.pretrained = pretrained
        config.rel_sample_negatives_largest = rel_sample_negatives_largest
        config.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest

        config.connectivity_loss_coefficient = connectivity_loss_coefficient
        config.use_freq_bias = use_freq_bias
        config.use_log_softmax = use_log_softmax
        config.freq_bias_eps = freq_bias_eps

        config.logit_adjustment = logit_adjustment
        config.logit_adj_tau = logit_adj_tau
        self.config = config

        if config.from_scratch:
            assert backbone_dirpath
            self.model = DetrForSceneGraphGeneration(config=config, fg_matrix=fg_matrix)
            self.model.model.backbone.load_state_dict(
                torch.load(f"{backbone_dirpath}/{config.backbone}.pt")
            )
            self.initialized_keys = []
        else:
            self.model, load_info = DetrForSceneGraphGeneration.from_pretrained(
                pretrained,
                config=config,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                fg_matrix=fg_matrix,
            )
            self.initialized_keys = load_info["missing_keys"] + [
                _key for _key, _, _ in load_info["mismatched_keys"]
            ]

        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_initialized = lr_initialized
        self.weight_decay = weight_decay
        self.multiple_sgg_evaluator = multiple_sgg_evaluator
        self.multiple_sgg_evaluator_list = multiple_sgg_evaluator_list
        self.single_sgg_evaluator = single_sgg_evaluator
        self.single_sgg_evaluator_list = single_sgg_evaluator_list
        self.coco_evaluator = coco_evaluator
        self.feature_extractor = feature_extractor

    
    def forward(self, pixel_values, pixel_mask):

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )
        return outputs

    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]  # [num_frames, C, H, W]
        labels = batch["labels"]
        num_frames = pixel_values.size(0)
        
        total_loss = 0
        total_loss_dict = {}
    
        # Process each frame sequentially
        for frame_idx in range(num_frames):
            # Get current frame
            frame_pixel_values = pixel_values[frame_idx:frame_idx+1]
            
            # labels의 구조를 list로 유지하되 단일 요소만 포함하도록 수정
            frame_labels = [{
                "class_labels": labels[0]["class_labels"][frame_idx:frame_idx+1],
                "boxes": labels[0]["boxes"][frame_idx:frame_idx+1],
                "rel": labels[0]["rel"]  # relationship tensor
            }]  # 리스트로 감싸서 단일 배치 형태 유지
            
            # Forward pass for current frame
            outputs = self.model(
                pixel_values=frame_pixel_values,
                pixel_mask=None,
                labels=frame_labels,  # 수정된 형태로 전달
                output_attentions=False,
                output_attention_states=True,
                output_hidden_states=True,
            )
            
            # Accumulate losses
            total_loss += outputs.loss
            for k, v in outputs.loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0) + v
            
            del outputs
    
        # Average losses over all frames
        total_loss = total_loss / num_frames
        total_loss_dict = {k: v/num_frames for k, v in total_loss_dict.items()}
        
        return total_loss, total_loss_dict

    

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "training_loss": loss.item(),
        }
        log_dict.update({f"training_{k}": v.item() for k, v in loss_dict.items()})
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        loss_dict["loss"] = loss
        del loss
        return loss_dict

    def validation_epoch_end(self, outputs):
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        for k in outputs[0].keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x[k] for x in outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)

    @rank_zero_only
    def on_train_start(self) -> None:
        self.config.save_pretrained(self.logger.log_dir)
        return super().on_train_start()


    def test_step(self, batch, batch_idx):
        # get the inputs
        self.model.eval()

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        targets = [{k: v.cpu() for k, v in label.items()} for label in batch["labels"]]

        with torch.no_grad():
            outputs = self.forward(pixel_values, pixel_mask)
            # eval SGG
            evaluate_batch(
                outputs,
                targets,
                self.multiple_sgg_evaluator,
                self.multiple_sgg_evaluator_list,
                self.single_sgg_evaluator,
                self.single_sgg_evaluator_list,
                self.config.num_labels,
            )
            # eval OD
            if self.coco_evaluator is not None:
                orig_target_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                )
                results = self.feature_extractor.post_process(
                    outputs, orig_target_sizes.to(self.device)
                )  # convert outputs of model to COCO api
                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results)
                }
                self.coco_evaluator.update(res)
    

    def test_epoch_end(self, outputs):
        log_dict = {}
        # log OD
        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            log_dict.update({"AP50": self.coco_evaluator.coco_eval["bbox"].stats[1]})

        # log SGG
        if self.multiple_sgg_evaluator is not None:
            recall = self.multiple_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.single_sgg_evaluator is not None:
            recall = self.single_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.single_sgg_evaluator_list, "sgdet", multiple_preds=False
            )
            recall = dict(zip(["(single)" + x for x in recall.keys()], recall.values()))
            mean_recall = dict(
                zip(["(single)" + x for x in mean_recall.keys()], mean_recall.values())
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]

        if self.lr_initialized is not None:  # rel_predictor
            initialized_lr_params = self.initialized_keys
        else:
            initialized_lr_params = []
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params))
                    and (not any(nd in n for nd in initialized_lr_params))
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in diff_lr_params) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        if initialized_lr_params:
            param_dicts.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in initialized_lr_params)
                        and p.requires_grad
                    ],
                    "lr": self.lr_initialized,
                }
            )
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--data_path", type=str, default="/mnt/ssd2tb/haewon/action-genome/")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--backbone_dirpath", type=str, default=""
    )  # required when from_scratch is True

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--auxiliary_loss", type=str2bool, default=False)
    parser.add_argument(
        "--from_scratch", type=str2bool, default=False
    )  # whether to train without pretrained detr
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
    )  # set to "architecture" when from_scratch is True

    # Hyperparameters
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--ce_loss_coefficient", type=float, default=2.0)
    parser.add_argument("--rel_loss_coefficient", type=float, default=15.0)
    parser.add_argument(
        "--connectivity_loss_coefficient", type=float, default=30.0
    )  # OI: 90
    parser.add_argument("--smoothing", type=float, default=1e-14)
    parser.add_argument("--rel_sample_negatives", type=int, default=80)
    parser.add_argument("--rel_sample_nonmatching", type=int, default=80)
    parser.add_argument(
        "--rel_sample_negatives_largest", type=str2bool, default=True
    )  # OI: True
    parser.add_argument(
        "--rel_sample_nonmatching_largest", type=str2bool, default=True
    )  # OI: False

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_epochs_finetune", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=2e-7)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lr_initialized", type=float, default=2e-4)  # for pretrained
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)

    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finetune", type=str2bool, default=True)

    parser.add_argument(
        "--filter_duplicate_rels", type=str2bool, default=True
    )  # for OI
    parser.add_argument("--filter_multiple_rels", type=str2bool, default=True)  # for OI
    parser.add_argument("--use_freq_bias", type=str2bool, default=True)
    parser.add_argument("--use_log_softmax", type=str2bool, default=False)

    # Evaluation
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_when_train_end", type=str2bool, default=True)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)

    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])

    args = parser.parse_args()
    if args.from_scratch:
        args.pretrained = args.architecture

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )
    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
            args.architecture, size=800, max_size=1333
        )
    )

    
    AG.initialize_shared_cache()

    train_dataset = AG(mode="train", datasize = "mini", feature_extractor=feature_extractor_train, data_path="/mnt/ssd2tb/haewon/action-genome/", filter_nonperson_box_frame=True,
                      filter_small_box=False)

    # 데이터셋을 train/validation으로 나누기 (80% train, 20% validation)
    train_data, val_data = train_test_split(list(train_dataset), test_size=0.5, random_state=42)

    # 새로운 train dataset과 val dataset을 생성
    train_dataset = AG(mode="train", datasize="mini", feature_extractor=feature_extractor_train, data_path="/mnt/ssd2tb/haewon/action-genome/",
                    filter_nonperson_box_frame=True, filter_small_box=False, custom_data=train_data)
    val_dataset = AG(mode="train", datasize="mini", feature_extractor=feature_extractor, data_path="/mnt/ssd2tb/haewon/action-genome/",
                    filter_nonperson_box_frame=True, filter_small_box=False, custom_data=val_data)

    # DataLoader 생성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8,
                                                collate_fn=cuda_collate_fn, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8,
                                                collate_fn=cuda_collate_fn, pin_memory=True)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
                                                
    # 객체 클래스 정보를 딕셔너리로 매핑
    id2label = {i: name for i, name in enumerate(train_dataset.object_classes)} 

    # 전경 행렬 생성
    fg_matrix = ag_get_statistics(train_dataset, must_overlap=True)

    # Evaluator
    rel_categories = train_dataset.rel_categories
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    coco_evaluator = None


    multiple_sgg_evaluator_list = []
    single_sgg_evaluator_list = []
    if args.eval_when_train_end:
        if args.eval_multiple_preds:
            multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=True
            )  # R@k
            for index, name in enumerate(rel_categories):
                multiple_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=True),
                    )
                )
        if args.eval_single_preds:
            single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=False
            )  # R@k
            for index, name in enumerate(rel_categories):
                single_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=False),
                    )
                )

        
        coco_evaluator = CocoEvaluator(
            val_dataset, ["bbox"]
        )  # initialize evaluator with ground truths


    # Logger setting
    save_dir = f"{args.output_path}/egtr__{'/'.join(args.pretrained.split('/')[-3:]).replace('/', '__')}"
    if args.from_scratch:
        save_dir += "__from_scratch"
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr__{args.lr_backbone}_{args.lr}_{args.lr_initialized}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    # Trainer setting
    logger = TensorBoardLogger(save_dir, name=name, version=version)
    if os.path.exists(f"{logger.log_dir}/checkpoints"):
        if os.path.exists(f"{logger.log_dir}/checkpoints/last.ckpt"):
            ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
        else:
            ckpt_path = sorted(
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]
    else:
        ckpt_path = None

    # Module
    module = SGG(
        architecture=args.architecture,
        backbone_dirpath=args.backbone_dirpath,
        auxiliary_loss=args.auxiliary_loss,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        lr_initialized=args.lr_initialized,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        main_trained="",
        from_scratch=args.from_scratch,
        id2label=id2label,
        rel_loss_coefficient=args.rel_loss_coefficient,
        smoothing=args.smoothing,
        rel_sample_negatives=args.rel_sample_negatives,
        rel_sample_nonmatching=args.rel_sample_nonmatching,
        rel_categories=rel_categories,
        multiple_sgg_evaluator=multiple_sgg_evaluator,
        multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
        single_sgg_evaluator=single_sgg_evaluator,
        single_sgg_evaluator_list=single_sgg_evaluator_list,
        coco_evaluator=coco_evaluator,
        feature_extractor=feature_extractor,
        num_queries=args.num_queries,
        ce_loss_coefficient=args.ce_loss_coefficient,
        rel_sample_negatives_largest=args.rel_sample_negatives_largest,
        rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
        use_freq_bias=args.use_freq_bias,
        fg_matrix=fg_matrix,
        use_log_softmax=args.use_log_softmax,
        freq_bias_eps=1e-12,
        connectivity_loss_coefficient=args.connectivity_loss_coefficient,
        logit_adjustment=args.logit_adjustment,
        logit_adj_tau=args.logit_adj_tau,
    )

    # Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        filename="{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=args.patience, verbose=True, mode="min"
    )

    # Train
    trainer = None
    if not args.skip_train:
        # Main training
        if not Path(
            TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            ).log_dir
        ).exists():
            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                gpus=args.gpus,
                max_epochs=args.max_epochs,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Main training")
            trainer.fit(module, ckpt_path=ckpt_path)

            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

        if args.finetune:
            ckpt_path = sorted(  # load best model
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]

            # Finetune trainer setting
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )
            if os.path.exists(f"{logger.log_dir}/checkpoints"):
                finetune_ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
            else:
                finetune_ckpt_path = None

            # Finetune module
            module = SGG(
                architecture=args.architecture,
                backbone_dirpath=args.backbone_dirpath,
                auxiliary_loss=args.auxiliary_loss,
                lr=args.lr * 0.1,
                lr_backbone=args.lr_backbone * 0.1,
                lr_initialized=args.lr_initialized * 0.1,
                weight_decay=args.weight_decay,
                pretrained=args.pretrained,
                main_trained=ckpt_path,
                from_scratch=args.from_scratch,
                id2label=id2label,
                rel_loss_coefficient=args.rel_loss_coefficient,
                smoothing=args.smoothing,
                rel_sample_negatives=args.rel_sample_negatives,
                rel_sample_nonmatching=args.rel_sample_nonmatching,
                rel_categories=rel_categories,
                multiple_sgg_evaluator=multiple_sgg_evaluator,
                multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
                single_sgg_evaluator=single_sgg_evaluator,
                single_sgg_evaluator_list=single_sgg_evaluator_list,
                coco_evaluator=coco_evaluator,
                feature_extractor=feature_extractor,
                num_queries=args.num_queries,
                ce_loss_coefficient=args.ce_loss_coefficient,
                rel_sample_negatives_largest=args.rel_sample_negatives_largest,
                rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
                use_freq_bias=args.use_freq_bias,
                fg_matrix=fg_matrix,
                use_log_softmax=args.use_log_softmax,
                freq_bias_eps=1e-12,
                connectivity_loss_coefficient=args.connectivity_loss_coefficient,
                logit_adjustment=args.logit_adjustment,
                logit_adj_tau=args.logit_adj_tau,
            )

            # Finetune callback
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                filename="{epoch:02d}-{validation_loss:.2f}",
                save_last=True,
            )
            early_stop_callback = EarlyStopping(
                monitor="validation_loss",
                patience=args.patience,
                verbose=True,
                mode="min",
            )

            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                max_epochs=args.max_epochs_finetune,
                gpus=args.gpus,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Finetune with smaller lr")
            trainer.fit(module, ckpt_path=finetune_ckpt_path)

        if trainer is not None:
            torch.distributed.destroy_process_group()
            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

    # Evaluation
    if args.eval_when_train_end and (trainer is None or trainer.is_global_zero):
        if args.skip_train and args.finetune:
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )

        # Load best model
        ckpt_path = sorted(
            glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)  # load best model

        # Eval
        trainer = Trainer(
            precision=args.precision, logger=logger, gpus=1, max_epochs=-1
        )

        test_dataset = AG(mode="test", datasize = "mini", feature_extractor=feature_extractor, data_path="/mnt/ssd2tb/haewon/action-genome/", filter_nonperson_box_frame=True, filter_small_box=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                                    collate_fn=cuda_collate_fn, pin_memory=True)

        
        if trainer.is_global_zero:
            print("### Evaluation")
        metric = trainer.test(module, dataloaders=test_dataloader)

        # Save eval metric
        metric = metric[0]
        device = "".join(torch.cuda.get_device_name(0).split()[1:2])
        filename = f'{ckpt_path.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
        if args.logit_adjustment:
            filename += f"__la_{args.logit_adj_tau}"
        metric["eval_arg"] = args.__dict__
        with open(f"{filename}.json", "w") as f:
            json.dump(metric, f)
        print("metric is saved in", f"{filename}.json")
