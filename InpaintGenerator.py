import torch
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image


class InpaintGenerator(ABC):
    def preprocess(self, image, mask, vae, device, dtype):
        img_arr = np.array(image.resize((512, 512))).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        mask = mask.resize((64, 64), resample=Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = 1.0 - mask
        mask = torch.from_numpy(mask).to(device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        return latents, mask

    def postprocess(self, latents, vae):
        latents = latents / 0.18215
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)

        return Image.fromarray(image)

    def step(self, scheduler, pred_noise, latent, t, t_prev):
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(latent.device)

        pred_original_sample = (latent - (1 - alpha_prod_t) ** 0.5 * pred_noise) / (alpha_prod_t ** 0.5)

        beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        coeff_x0 = (alpha_prod_t_prev ** 0.5 * beta_t) / (1 - alpha_prod_t)
        coeff_xt = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_prod_t)

        mean = coeff_x0 * pred_original_sample + coeff_xt * latent

        if t > 0:
            noise = torch.randn_like(latent)
            # חישוב השונות (Posterior Variance)
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * beta_t
            return mean + (variance ** 0.5) * noise

        return mean

    def add_noise(self, scheduler, latents, t):
        noise = torch.randn_like(latents)
        alpha_prod = scheduler.alphas_cumprod[t]
        return (alpha_prod ** 0.5) * latents + ((1 - alpha_prod) ** 0.5) * noise

    @abstractmethod
    def generate(self, mask, image, prompt, pipe, scheduler, cfg):
        pass


# --- 1. Vanilla Implementation ---
class VanillaGenerator(InpaintGenerator):
    def generate(self, mask, image, prompt, pipe, scheduler, cfg):
        # This uses your existing logic: preprocess -> loop -> postprocess
        latents_x0, mask_latent = self.preprocess(image, mask, pipe.vae, pipe.device, pipe.unet.dtype)
        latents = torch.randn_like(latents_x0)

        for i, t in enumerate(scheduler.timesteps):
            t_val = t.item()
            t_prev = scheduler.timesteps[i + 1].item() if i < len(scheduler.timesteps) - 1 else 0

            latents = self._single_step(latents, latents_x0, mask_latent, prompt, t_val, t_prev, pipe, scheduler, cfg)

        return self.postprocess(latents, pipe.vae)

    def _single_step(self, latents, latents_x0, mask, prompt, t, t_prev, pipe, scheduler, cfg):
        # Core DDPM step + Stitching
        model_input = torch.cat([latents] * 2) if cfg > 0 else latents
        with torch.no_grad():
            noise_pred = pipe.unet(model_input, t, encoder_hidden_states=prompt).sample

        if cfg > 0:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg * (noise_text - noise_uncond)

        latents_unknown = self.step(scheduler, noise_pred, latents, t, t_prev)
        latents_known = self.add_noise(scheduler, latents_x0, t_prev)

        return (mask * latents_known) + ((1.0 - mask) * latents_unknown)


# --- 2. Resampling (RePaint) Implementation ---
class ResamplingGenerator(VanillaGenerator):
    def __init__(self, jump_length=10, jump_inn_sample=10):
        self.jump_length = jump_length  # How many steps to go back
        self.jump_n_sample = jump_n_sample  # How many times to repeat the jump

    def generate(self, mask, image, prompt, pipe, scheduler, cfg):
        latents_x0, mask_latent = self.preprocess(image, mask, pipe.vae, pipe.device, pipe.unet.dtype)
        latents = torch.randn_like(latents_x0)

        timesteps = scheduler.timesteps
        n_steps = len(timesteps)

        i = 0
        while i < n_steps:
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item() if i < n_steps - 1 else 0

            # Standard Step Forward
            latents = self._single_step(latents, latents_x0, mask_latent, prompt, t, t_prev, pipe, scheduler, cfg)

            # Jumping Logic (Time Travel)
            # Check if we should jump back (don't jump on the very last steps)
            if i > 0 and i % self.jump_length == 0 and i < n_steps - self.jump_length:
                # Perform Resampling
                for _ in range(self.jump_n_sample):
                    # Jump Back: Add noise to go from t_prev back to t
                    # Mathematically: x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*epsilon
                    beta = 1 - (scheduler.alphas_cumprod[t] / scheduler.alphas_cumprod[t_prev])
                    noise = torch.randn_like(latents)
                    latents = (1 - beta) ** 0.5 * latents + (beta ** 0.5) * noise

                    # Denoise again
                    latents = self._single_step(latents, latents_x0, mask_latent, prompt, t, t_prev, pipe, scheduler,
                                                cfg)

            i += 1

        return self.postprocess(latents, pipe.vae)


# --- Factory / Wrapper ---
class InpaintContext:
    def __init__(self, mode="vanilla", **kwargs):
        if mode == "vanilla":
            self.generator = VanillaGenerator()
        elif mode == "resampling":
            self.generator = ResamplingGenerator(**kwargs)
        else:
            raise ValueError("Unknown mode")

    def run(self, mask, image, prompt, pipe, scheduler, cfg):
        return self.generator.generate(mask, image, prompt, pipe, scheduler, cfg)