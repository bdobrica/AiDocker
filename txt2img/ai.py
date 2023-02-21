#!/usr/bin/env python3
import os
import traceback
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from daemon import AiBatch as Batch
from daemon import AiBatchDaemon as Daemon

__version__ = "0.8.12"


class AiBatch(Batch):
    @staticmethod
    def flatten_list(list_of_lists: List[List]) -> List:
        return [item for sublist in list_of_lists for item in sublist]

    @staticmethod
    def load_source(source_file: Path) -> str:
        with open(source_file, "r") as fp:
            return fp.read().strip()

    @staticmethod
    def save_image(image: np.ndarray, prepared_file: Path, idx: int) -> None:
        image_file = prepared_file.parent / (
            prepared_file.stem + "_{0:02x}".format(idx) + prepared_file.suffix
        )
        cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @property
    def height(self) -> int:
        return max([meta.get("image_height", 512) for meta in self.metadata])

    @property
    def width(self) -> int:
        return max([meta.get("image_width", 512) for meta in self.metadata])

    @property
    def inference_steps(self) -> int:
        return sum([meta.get("inference_steps", 20) for meta in self.metadata])

    def prepare(self) -> List[str]:
        return AiBatch.flatten_list(
            [
                [AiBatch.load_source(source_file)] * meta.get("samples", 1)
                for source_file, meta in zip(self.source_files, self.metadata)
            ]
        )

    def serve(self, inference_data: np.ndarray) -> None:
        prepared_files = AiBatch.flatten_list(
            [
                [prepared_file] * meta.get("samples", 1)
                for prepared_file, meta in zip(
                    self.prepared_files, self.metadata
                )
            ]
        )
        indices = AiBatch.flatten_list(
            [list(range(meta.get("samples", 1))) for meta in self.metadata]
        )
        _ = [
            AiBatch.save_image(data, prepared_files[idx], indices[idx])
            for idx, data in enumerate(inference_data)
        ]


class AIDaemon(Daemon):
    def load(self) -> None:
        MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/app/stable-diffusion")
        MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
        if MODEL_DEVICE.startswith("cuda") and not torch.cuda.is_available():
            MODEL_DEVICE = "cpu"

        self.device = torch.device(MODEL_DEVICE)

        self.vae = AutoencoderKL.from_pretrained(
            MODEL_PATH, subfolder="vae", local_files_only=True
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", local_files_only=True
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            MODEL_PATH, subfolder="unet", local_files_only=True
        ).to(self.device)
        self.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )

    def ai(self, batch: AiBatch) -> None:
        try:
            MODEL_GUIDANCE_SCALE = float(
                os.environ.get("MODEL_GUIDANCE_SCALE", 7.5)
            )

            generator = torch.manual_seed(42)

            prompt = batch.prepare()
            samples = len(prompt)

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
                (
                    samples,
                    self.unet.in_channels,
                    batch.height // 8,
                    batch.width // 8,
                ),
                generator=generator,
            )
            latents = latents.to(self.device)

            self.scheduler.set_timesteps(
                batch.inference_steps, device=self.device
            )
            latents = latents * self.scheduler.init_noise_sigma

            for inference_step in self.scheduler.timesteps:
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
            latents = 1 / 0.18215 * latents
            output = self.vae.decode(latents).sample
            output = (output / 2 + 0.5).clamp(0, 1)
            output = output.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (output * 255).round().astype("uint8")
            batch.serve(images)
            batch.update_metadata({"processed": "true"})

        except Exception as e:
            batch.update_metadata({"processed": "error"})
            if os.environ.get("DEBUG", "false").lower() in ("true", "1", "on"):
                print(traceback.format_exc())


if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile=PIDFILE_PATH, chroot=CHROOT_PATH).start()
