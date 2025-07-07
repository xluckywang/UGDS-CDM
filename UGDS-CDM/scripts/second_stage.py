import sys
sys.path.append('../')
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_source_data_for_domain_translation,
    get_image_filenames_for_label
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

import torchvision.utils as tvu


def main():
    classifier_scales = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    guides = ["Entropy"]

    for scale in classifier_scales:
        for guide in guides:
            run_experiment(scale, guide)


def run_experiment(classifier_scale, which_guide):
    args = create_argparser().parse_args()

    # Update args with new classifier_scale and which_guide
    args.classifier_scale = classifier_scale
    args.which_guide = which_guide
    args.save_dir = f'../{classifier_scale}_150_{which_guide}_mt/dataname'
    args.log_root = f'../{classifier_scale}_150_{which_guide}_mt'

    dist_util.setup_dist(single_gpu=args.single_gpu)
    logger.configure(args.log_root)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
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

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            if args.which_guide == "classifier":
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
            elif args.which_guide == "Entropy":
                uncertainties = -(probs * log_probs).sum()
                return th.autograd.grad(uncertainties, x_in)[0] * args.classifier_scale


    source = [int(v) for v in args.source.split(",")]
    target = [int(v) for v in args.target.split(",")]
    source_to_target_mapping = {s: t for s, t in zip(source, target)}

    logger.log("running image translation...")
    data = load_source_data_for_domain_translation(
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.val_dir,
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        for i in target:
            os.makedirs(os.path.join(args.save_dir, str(i)))

    for i, (batch, extra) in enumerate(data):
        logger.log(f"translating batch {i}, shape {batch.shape}.")

        batch = batch.to(dist_util.dev())

        source_y = dict(y=extra["y"].to(dist_util.dev()))
        target_y_list = [source_to_target_mapping[v.item()] for v in extra["y"]]
        target_y = dict(y=th.tensor(target_y_list).to(dist_util.dev()))

        logger.log("encoding the source images.")
        noise = diffusion.ddim_reverse_sample_loop(
            model,
            batch,
            clip_denoised=False,
            model_kwargs=source_y,
            device=dist_util.dev(),
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        sample = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=target_y,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            eta=args.eta
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        images = []
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(images) * args.batch_size} samples")

        logger.log("saving translated images.")
        images = np.concatenate(images, axis=0)

        for index in range(images.shape[0]):
            filepath = os.path.join(args.save_dir, str(target_y_list[index]), str(int(args.classifier_scale))+"_"+extra["filepath"][index])
            image = Image.fromarray(images[index])
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

    dist.barrier()
    logger.log(f"domain translation complete for classifier_scale={classifier_scale} and which_guide={which_guide}")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=10,
        classifier_scale=0,
        eta=0.0,
        save_dir='../0_0/dataname',
        which_guide="Entropy",
        log_root='../0_0',
        single_gpu=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="../logs_mt1/model080000.pt",
        help="Path to the diffusion model weights."
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default='./../logs_clas_mt1/model150000.pt',
        help="Path to the classifier model weights."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0,1,2,3,4",
        help="Source domains."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="0,1,2,3,4",
        help="Target domains."
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default='../logs_expert150_mt/dataname',
        help="The local directory containing ImageNet validation dataset, "
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()