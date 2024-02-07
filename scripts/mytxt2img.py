# updated by jingxiang zhang
import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from imwatermark import WatermarkEncoder
from einops import rearrange
from pytorch_lightning import seed_everything
import json
import random

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    print(f"Loading finished")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def cache_conditioning(prompts, model, cache_dir="cache", release=False):
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    index_file_path = os.path.join(cache_dir, "index.json")
    
    # Load or initialize the index file
    if os.path.isfile(index_file_path):
        with open(index_file_path, 'r') as f:
            index = json.load(f)
    else:
        index = {}
    
    # Check if the prompt is in the index
    if prompts in index:
        # Load the cached feature matrix
        feature_path = os.path.join(cache_dir, index[prompts])
        condition = torch.load(feature_path)
    else:
        model.init_cond_stage_forward()
        # Generate the conditioning if not cached
        condition = model.get_learned_conditioning([prompts])
        if release:
            del model.cond_stage_model
            torch.cuda.empty_cache()

        # Save the feature matrix
        feature_file_name = f"feature_{len(index)}.pt"
        feature_path = os.path.join(cache_dir, feature_file_name)
        torch.save(condition, feature_path)
        
        # Update the index and save
        index[prompts] = feature_file_name
        with open(index_file_path, 'w') as f:
            json.dump(index, f)
    
    return condition


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/myv1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=913654895,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()

    if opt.seed == 913654895:
        opt.seed = random.randint(1, 2000000000)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prompts = opt.prompt
    assert prompts is not None
    prompts = [prompts]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    with torch.no_grad():
        with model.ema_scope():
            print("get prompt encode")
            uc = cache_conditioning("", model, cache_dir="cache")
            c = cache_conditioning(prompts[0], model, cache_dir="cache", release=True)
            print("finish prompt encode")
            c = c.to("cuda")
            uc = uc.to("cuda")
            model.model = model.model.half().to("cuda")

            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            print("start DDPM sampling")
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                            conditioning=c,
                                            batch_size=1,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta)
            print("finish DDPM sampling")
            # remove the DDPM model from GPU memory
            del model.model
            torch.cuda.empty_cache()

            print("upsampling to 512 * 512")
            model.first_stage_model = model.first_stage_model.to("cuda")
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            # remove the upsampling model from GPU memory
            del model.first_stage_model
            torch.cuda.empty_cache()

            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

            # save image
            for x_sample in x_checked_image_torch:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1
                

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()


# python scripts/mytxt2img.py --prompt "an apple and a banana" --outdir "outputs/txt2img" --W 512 --H 768