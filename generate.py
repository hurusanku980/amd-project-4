import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a futuristic datacenter with AMD GPU servers, cyberpunk style")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--images", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # ROCm optimizations
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()

    start = time.time()
    for i in range(args.images):
        image = pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=7.5).images[0]
        image.save(f"output_{i}.png")
        print(f"Generated image {i+1}/{args.images}")

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s ({elapsed/args.images:.1f}s per image)")

if __name__ == "__main__":
    main()
