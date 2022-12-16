#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from daemon import Daemon

__version__ = "0.8.8"


class AIDaemon(Daemon):
    def initialize(self, model_path: str):
        MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
        if MODEL_DEVICE == "cuda" and not torch.cuda.is_available():
            MODEL_DEVICE = "cpu"

        self.device = torch.device(MODEL_DEVICE)
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", local_files_only=True
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", local_files_only=True
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", local_files_only=True
        ).to(self.device)
        self.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )

    def ai(self, source_file, prepared_file, **metadata):
        try:
            MODEL_PATH = os.environ.get(
                "MODEL_PATH", "/opt/app/stable-diffusion"
            )

            MODEL_GUIDANCE_SCALE = float(
                os.environ.get("MODEL_GUIDANCE_SCALE", 7.5)
            )

            # Initialize
            prompt = metadata.get("prompt", "")
            samples = int(metadata.get("samples", 1))
            height = metadata.get("image_height", 512)
            width = metadata.get("image_width", 512)
            inference_steps = metadata.get("inference_steps", 20)
            generator = torch.manual_seed(42)

            prompt = [prompt] * samples

            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * samples,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            latents = torch.randn(
                (samples, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(self.device)

            self.scheduler.set_timesteps(inference_steps)
            latents = latents * self.scheduler.init_noise_sigma

            for inference_step in range(inference_steps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input
                )

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        inference_step,
                        encoder_hidden_states=text_embeddings,
                    ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + MODEL_GUIDANCE_SCALE * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, inference_step, latents
                ).prev_sample

            # scale and decode the image latents with vae
            start = time.perf_counter()
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            for idx, image in enumerate(images):
                image_file = prepared_file.parent / (
                    prepared_file.stem
                    + "_{0:02x}".format(idx)
                    + prepared_file.suffix
                )
                cv2.imwrite(str(image_file), image)

            results = []
            json_file = prepared_file.with_suffix(".json")
            with json_file.open("w") as f:
                json.dump({"results": results}, f)

        except Exception as e:
            pass

        source_file.unlink()
        sys.exit()

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        MAX_FORK = int(os.environ.get("MAX_FORK", 8))
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 4096))

        staged_files = sorted(
            [
                f
                for f in Path(STAGED_PATH).glob("*")
                if f.is_file() and f.suffix != ".json"
            ],
            key=lambda f: f.stat().st_mtime,
        )
        source_files = [f for f in Path(SOURCE_PATH).glob("*") if f.is_file()]
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop(0)

            meta_file = staged_file.with_suffix(".json")
            if meta_file.is_file():
                with meta_file.open("r") as fp:
                    try:
                        image_metadata = json.load(fp)
                    except:
                        image_metadata = {}
            image_metadata = {
                **{
                    "extension": staged_file.suffix,
                    "background": "",
                },
                **image_metadata,
            }

            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) / (
                staged_file.stem + image_metadata["extension"]
            )

            with staged_file.open("rb") as src_fp, source_file.open(
                "wb"
            ) as dst_fp:
                while True:
                    chunk = src_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_fp.write(chunk)

            staged_file.unlink()
            self.ai(source_file, prepared_file, **image_metadata)

    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.queue()
            time.sleep(1.0)


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
