import sys
sys.path.append('../')
import cv2
import argparse
import os
from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
import matplotlib.pyplot as plt
from torch.optim import Adam

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def train_and_save_best_scales():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.single_gpu)
    logger.configure(args.log_root)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    if args.classifier_path != "":
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    classifier_scales = th.tensor(args.classifier_scales, dtype=th.float32, device=dist_util.dev(), requires_grad=True)
    optimizer = Adam([classifier_scales], lr=0.01)
    num_iterations = 5
    best_classifier_scales = [None] * len(args.category_name_list)
    best_scores = [float('inf')] * len(args.category_name_list)

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            classifier_scale = classifier_scales[y.view(-1)].to(dist_util.dev())
            return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale.view(-1, 1, 1, 1)

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    for iteration in range(num_iterations):
        logger.log(f"Iteration {iteration + 1} / {num_iterations}")

        logger.log("sampling...")
        all_images = []
        all_labels = []
        num_samples = sum(args.category_num_list)
        lis = []
        for i in range(len(args.category_num_list)):
            lis.extend([i] * args.category_num_list[i])

        while len(all_images) * args.batch_size < num_samples:
            model_kwargs = {}

            classes = th.tensor(
                lis[len(all_images) * args.batch_size:len(all_images) * args.batch_size + args.batch_size],
                device=dist_util.dev())

            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model_fn,
                (len(classes), 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if classifier_scales is not None else None,
                device=dist_util.dev(),
                cfg=args.cfg
            )

            if args.get_image:
                sample = ((sample + 1) / 2).clamp(0, 1)
                tvu.save_image(sample, os.path.join(logger.get_dir(), f"output_{iteration}_{len(all_images)}.png"))
                sample = sample.to(th.uint8)
                break
            else:
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: num_samples]
        if dist.get_rank() == 0:
            if args.get_images:
                if not os.path.exists(os.path.join(args.dataset_dir, args.dataset_name)):
                    os.makedirs(os.path.join(args.dataset_dir, args.dataset_name))
                    for i in args.category_name_list:
                        os.makedirs(os.path.join(args.dataset_dir, args.dataset_name, i))
                for i in range(len(arr)):
                    im = Image.fromarray(arr[i])
                    im.save(os.path.join(args.dataset_dir, args.dataset_name, args.category_name_list[label_arr[i]],
                                         f"{iteration}_{i}.png"))
            else:
                shape_str = "x".join([str(x) for x in arr.shape])
                out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
                logger.log(f"saving to {out_path}")
                np.savez(out_path, arr, label_arr)

        dist.barrier()

        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(2, 4, i + 1)
            plt.imshow(arr[i])
            plt.axis('off')
        plt.show()

        scores = []
        for category in args.category_name_list:
            score = float(input(f"请输入类别 {category} 的图像质量评分 (0-100): "))
            scores.append(score)

        total_loss = 0.0
        for i, score in enumerate(scores):
            target_score = 100.0
            current_score = score / target_score
            #loss = F.mse_loss(classifier_scales[i], th.tensor(current_score, dtype=th.float32, device=dist_util.dev()))
            loss = (100-score)*F.mse_loss((classifier_scales[i])/10, th.tensor(current_score, dtype=th.float32, device=dist_util.dev()))
            total_loss += loss.item()
            logger.log(f"loss for category {args.category_name_list[i]} is {loss.item()}")

            if loss.item() < best_scores[i]:
                best_scores[i] = loss.item()
                best_classifier_scales[i] = classifier_scales[i].clone().detach().cpu().numpy()
                np.save(os.path.join(logger.get_dir(), f"best_classifier_scales_{args.category_name_list[i]}.npy"),
                        best_classifier_scales[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        logger.log("sampling complete")
        logger.log(f"classifier_scales is {classifier_scales}")


    logger.log("Generating 200 images for each category using the best classifier scales...")
    logger.log(f"the best classifier_scales is {best_classifier_scales}")
    for category_idx, category in enumerate(args.category_name_list):
        best_classifier_scale = np.load(os.path.join(logger.get_dir(), f"best_classifier_scales_{category}.npy"))
        classifier_scales.data[category_idx] = th.tensor(best_classifier_scale, dtype=th.float32, device=dist_util.dev()).data

        all_images = []
        all_labels = []
        num_samples = 150
        lis = [category_idx] * num_samples

        while len(all_images) * args.batch_size < num_samples:
            model_kwargs = {}

            classes = th.tensor(
                lis[len(all_images) * args.batch_size:len(all_images) * args.batch_size + args.batch_size],
                device=dist_util.dev())

            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model_fn,
                (len(classes), 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if classifier_scales is not None else None,
                device=dist_util.dev(),
                cfg=args.cfg
            )

            if args.get_image:
                sample = ((sample + 1) / 2).clamp(0, 1)
                tvu.save_image(sample, os.path.join(logger.get_dir(), f"final_output_{category}_{len(all_images)}.png"))
                sample = sample.to(th.uint8)
                break
            else:
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples for category {category}")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: num_samples]
        if dist.get_rank() == 0:
            if args.get_images:
                category_dir = os.path.join(args.dataset_dir, args.dataset_name, category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                for i in range(len(arr)):
                    im = Image.fromarray(arr[i])
                    im.save(os.path.join(category_dir, f"{category}_{i}.png"))
            else:
                shape_str = "x".join([str(x) for x in arr.shape])
                out_path = os.path.join(logger.get_dir(), f"final_samples_{category}_{shape_str}.npz")
                logger.log(f"saving to {out_path}")
                np.savez(out_path, arr, label_arr)

        dist.barrier()
    logger.log("Final image generation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=10,
        use_ddim=True,
        model_path="../logs_mt1/model080000.pt",
        classifier_path='./../logs_clas_mt1/model150000.pt',
        classifier_scales=[10, 10, 10, 10, 10],
        log_root='./../logs_150_mt',
        dataset_dir='./../logs_expert150_mt',
        category_name_list=['Blowhole', 'Break', 'Crack','Fray', 'Uneven'],
        category_num_list=[1, 1, 1, 1, 1],
        dataset_name='dataname',
        get_image=False,
        get_images=True,
        single_gpu=False,
        cfg=0,)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    train_and_save_best_scales()