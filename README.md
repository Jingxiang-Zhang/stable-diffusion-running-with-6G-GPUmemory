# Stable Diffusion v1 Running with 6G GPU Memory
This project is forked from https://github.com/CompVis/stable-diffusion, and aim to run text to image generator in a 6G memory GPU. 

Stable diffusion for text to image generator contain 3 models:

1. CLIP text encoder: embed the input text prompt into condition c

2. U-Net: generate a small size image by DDPM, typically 64 × 64, or 96 × 64

3. Upsampling decoder: upsample the small size image to a large size image, typically by factor of 8

To minimize the GPU memory cost and improve efficiency, there are several updates:

1. Cache the condition c for the same prompt: Only use CLIP text encoder for the first time of using the prompt, and then cache it into the file system. Each prompt cache takes 232 KB.
2. Load the model and data to CUDA only when used. After using the model, remove it from CUDA.
3. Use half-precision for U-Net (diffusion part)



## How to use?

1. System environment recommendation: git, anaconda, vscode

2. Download this version of stable diffusion by:

   ```
   git clone https://github.com/Jingxiang-Zhang/stable-diffusion-running-with-6G-GPUmemory.git
   ```

3. Create a conda environment by:

   ```
   conda env create -f environment.yaml
   conda activate ldm
   ```

4. Similar to stable diffusion, you need to download pretrained weight from Hugging Face: [link](https://huggingface.co/CompVis), please use  stable-diffusion-v-1-4-original version without full-ema (which is much larger). Or, click [download v1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt?download=true) to start.

5. Run the test scenario:

   ```
   python scripts/mytxt2img.py --prompt "a photograph of an astronaut riding a horse" --outdir "outputs/txt2img" --W 512 --H 768
   ```

Mind that in the first time, this project need to download CLIP model in C:\Users\username\\.cache\huggingface\transformers, which is slow.

Limitations:

1. Currently only support DDIM sampler. use `--plms` will cause errors.
2. Currently only support text to image generation.

Caveats:

1. The output shape of DDPM is (H/f, W/f), the maximum size that 6G memory GPU can support is (96, 64), which means by default f = 8, the maximum output image is (768, 512)



## Update details

1. mytxt2img.py script in /scripts/ folder

2. /ldm/modules/diffusionmodules/util.py, line 261, `forward(x.float())` -> `forward(x)`. x.float() will convert the data into float32 type, but the type of our input x and model is float16.
3. /ldm/modules/diffusionmodules/openaimodel.py, line 725, append `t_emb = t_emb.half()`. The default time embedding value is float32, we need to convert it to float16.
4. /ldm/modules/diffusionmodules/openaimodel.py, line 732, `h = x.type(self.dtype)` -> `h = x.type(torch.float16)`.
5. /ldm/models/diffusion/ddpm.py:
   - Line 437: append `init_cond_stage_config_immediately=True` as input parameter. We can set this value to False to disable loading CLIP model (if the prompt is in cache and don't need to run CLIP text encoder).
   - Line 464: load CLIP model conditionally.
   - Line 485: append a new function to manually load CLIP model.
6. /ldm/models/diffusion/ddim.py:
   - Line 121: `device = "cuda"` force the model run in GPU.
   - Line 171: set all the input values to half precision float16.
7. /configs/stable-diffusion/myv1-inference.yaml: copy from /configs/stable-diffusion/v1-inference.yaml, append `init_cond_stage_config_immediately: False` to disable automatically load CLIP model.



## Citation
    @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {10684-10695}
    }

*This model card was written by: Robin Rombach and Patrick Esser and is based on the [DALL-E Mini model card](https://huggingface.co/dalle-mini/dalle-mini).*
