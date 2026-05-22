# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Tuple
from gdf import BaseSchedule, DDIMSampler, DDPMSampler, GDF, SimpleSampler
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import torchvision.transforms as T
from train import WurstCoreC
from utils import setup_csd


transform = T.ToPILImage()


class RBM(GDF):
  """
  Sampling with reference-based modulation. 
  """
  def StableCascade_inversion(
    self,
    input_latent, 
    model,
    model_inputs: Dict[str, Any],
    shape: Tuple,
    unconditional_inputs: Dict[str, Any] = None,
    sampler=None, 
    schedule=None, 
    t_start=1.0, 
    t_end=0.0, 
    timesteps=20, 
    x_init=None, 
    cfg=3.0, 
    cfg_t_stop=None, 
    cfg_t_start=None, 
    cfg_rho=0.7, 
    sampler_params=None, 
    shift=1, 
    device="cpu",

    inversion_timesteps: int = 20
    ):

    timesteps = inversion_timesteps

    sampler_params = {} if sampler_params is None else sampler_params
    if sampler is None:
        sampler = DDPMSampler(self)
    r_range = torch.linspace(t_end, t_start, timesteps+1)
    schedule = self.schedule if schedule is None else schedule
    logSNR_range, alpha_cumprod = schedule.call_var(r_range, shift=shift)
    logSNR_range = logSNR_range[:, None].expand(
        -1, shape[0] if x_init is None else x_init.size(0)
    ).to(device)
    alpha_cumprod = alpha_cumprod[:, None].expand(
        -1, shape[0] if x_init is None else x_init.size(0)
    ).to(device)

    x = input_latent.clone()
    if cfg is not None:
        if unconditional_inputs is None:
            unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
        model_inputs = {
            k: torch.cat([v, v_u], dim=0) if isinstance(v, torch.Tensor) 
            else [torch.cat([vi, vi_u], dim=0) if isinstance(vi, torch.Tensor) and isinstance(vi_u, torch.Tensor) else None for vi, vi_u in zip(v, v_u)] if isinstance(v, list)
            else {vk: torch.cat([v[vk], v_u.get(vk, torch.zeros_like(v[vk]))], dim=0) for vk in v} if isinstance(v, dict)
            else None for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())
        }
    for i in range(0, timesteps):

        noise_cond = self.noise_cond(logSNR_range[i])

        if cfg is not None and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop) and (cfg_t_start is None or r_range[i].item() <= cfg_t_start):
            cfg_val = cfg
            if isinstance(cfg_val, (list, tuple)):
                assert len(cfg_val) == 2, "cfg must be a float or a list/tuple of length 2"
                cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (1-r_range[i].item())
            pred, pred_unconditional = model(torch.cat([x, x], dim=0), noise_cond.repeat(2), **model_inputs).chunk(2)
            pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
            if cfg_rho > 0:
                std_pos, std_cfg = pred.std(),  pred_cfg.std()
                pred = cfg_rho * (pred_cfg * std_pos/(std_cfg+1e-9)) + pred_cfg * (1-cfg_rho)
            else:
                pred = pred_cfg
        else:
            pred = model.orig_sample(x, noise_cond, **model_inputs)
        
        x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)

        #x = sampler(x, x0, epsilon, logSNR_range[i], logSNR_range[i+1], **sampler_params)
        x = alpha_cumprod[i+1].sqrt() * x0 + (1 - alpha_cumprod[i+1]).sqrt() * epsilon

        altered_vars = yield (x0, x, pred)

        # Update some running variables if the user wants
        if altered_vars is not None:
            cfg = altered_vars.get('cfg', cfg)
            cfg_rho = altered_vars.get('cfg_rho', cfg_rho)
            sampler = altered_vars.get('sampler', sampler)
            model_inputs = altered_vars.get('model_inputs', model_inputs)
            x = altered_vars.get('x', x)
            x_init = altered_vars.get('x_init', x_init)

  def orig_sample(
    self, model, model_inputs, shape, unconditional_inputs=None,
    sampler=None, schedule=None, t_start=1.0, t_end=0.0, timesteps=20,
    x_init=None, cfg=3.0, cfg_t_stop=None, cfg_t_start=None,
    cfg_rho=0.7, sampler_params=None, shift=1, device="cpu",
  ):
      sampler_params = {} if sampler_params is None else sampler_params
      if sampler is None:
          sampler = DDPMSampler(self)
      r_range = torch.linspace(t_start, t_end, timesteps+1)
      schedule = self.schedule if schedule is None else schedule
      logSNR_range = schedule(r_range, shift=shift)[:, None].expand(
          -1, shape[0] if x_init is None else x_init.size(0)
      ).to(device)

      x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
      if cfg is not None:
          if unconditional_inputs is None:
              unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
          model_inputs = {
              k: torch.cat([v, v_u], dim=0) if isinstance(v, torch.Tensor) 
              else [torch.cat([vi, vi_u], dim=0) if isinstance(vi, torch.Tensor) and isinstance(vi_u, torch.Tensor) else None for vi, vi_u in zip(v, v_u)] if isinstance(v, list)
              else {vk: torch.cat([v[vk], v_u.get(vk, torch.zeros_like(v[vk]))], dim=0) for vk in v} if isinstance(v, dict)
              else None for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())
          }
      for i in range(0, timesteps):
        noise_cond = self.noise_cond(logSNR_range[i])
        if cfg is not None and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop) and (cfg_t_start is None or r_range[i].item() <= cfg_t_start):
            cfg_val = cfg
            if isinstance(cfg_val, (list, tuple)):
                assert len(cfg_val) == 2, "cfg must be a float or a list/tuple of length 2"
                cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (1-r_range[i].item())
            pred, pred_unconditional = model(torch.cat([x, x], dim=0), noise_cond.repeat(2), **model_inputs).chunk(2)
            pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
            if cfg_rho > 0:
                std_pos, std_cfg = pred.std(),  pred_cfg.std()
                pred = cfg_rho * (pred_cfg * std_pos/(std_cfg+1e-9)) + pred_cfg * (1-cfg_rho)
            else:
                pred = pred_cfg
        else:
            pred = model(x, noise_cond, **model_inputs)
        
        x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)
        x = sampler(x, x0, epsilon, logSNR_range[i], logSNR_range[i+1], **sampler_params)
        altered_vars = yield (x0, x, pred)

        # Update some running variables if the user wants
        if altered_vars is not None:
            cfg = altered_vars.get('cfg', cfg)
            cfg_rho = altered_vars.get('cfg_rho', cfg_rho)
            sampler = altered_vars.get('sampler', sampler)
            model_inputs = altered_vars.get('model_inputs', model_inputs)
            x = altered_vars.get('x', x)
            x_init = altered_vars.get('x_init', x_init)

  def sample(
    self,
    model: torch.nn.Module,
    model_inputs: Dict[str, Any],
    shape: Tuple,
    unconditional_inputs: Dict[str, Any] = None,
    sampler: SimpleSampler = None,
    schedule: BaseSchedule = None,
    t_start: float = 1.0,
    t_end: float = 0.0,
    timesteps: float = 20,
    x_init: torch.Tensor = None,
    cfg: float = 3.0,
    cfg_t_stop: int = None,
    cfg_t_start: int = None,
    cfg_rho: float = 0.7,
    sampler_params: Dict[str, Any] = None,
    shift: int = 1,
    device: str = "cpu",
    x0_forward: torch.Tensor = None,
    x0_style_forward: torch.Tensor = None,
    num_iter: int = 3,
    eta: float = 4,
    tau: int = 20,
    eval_csd: bool = False,
    eval_sub_csd: bool = False,
    apply_pushforward: bool = False,
    tau_pushforward: int = 0,
    tau_pushforward_csd: int = 0,
    lam_content: float = 1.0,
    lam_style: float = 1.0,
    lam_txt_alignment: float = 0.0,
    use_attn_mask: bool = False,
    save_attn_mask: bool = False,
    models: WurstCoreC.Models = None,
    extras: WurstCoreC.Extras = None,
    sam_mask: float = 1.0,
    sam_prompt: str = None,
    use_sam_mask: bool = False,
    use_ddim_sampler: bool = False,

    Lambda: float = 0.9,
    t_guidance_start:float = 0.8,
    t_guidance_end: float = 0.2,
    inverted_noise: torch.Tensor = None,
    device_2: str = None
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device_2 = device if device_2 == None else device_2

    sampler_params = {} if sampler_params is None else sampler_params
    if sampler is None:
        sampler = DDPMSampler(self)
    r_range = torch.linspace(t_start, t_end, timesteps+1)
    schedule = self.schedule if schedule is None else schedule
    logSNR_range, alpha_cumprod = schedule.call_var(r_range, shift=shift)
    logSNR_range = logSNR_range[:, None].expand(
        -1, shape[0] if x_init is None else x_init.size(0)
    ).to(device)
    alpha_cumprod = alpha_cumprod[:, None].expand(
        -1, shape[0] if x_init is None else x_init.size(0)
    ).to(device)

    x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
    if cfg is not None:
        if unconditional_inputs is None:
            unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
        model_inputs = {
            k: torch.cat([v, v_u], dim=0) if isinstance(v, torch.Tensor) 
            else [torch.cat([vi, vi_u], dim=0) if isinstance(vi, torch.Tensor) and isinstance(vi_u, torch.Tensor) else None for vi, vi_u in zip(v, v_u)] if isinstance(v, list)
            else {vk: torch.cat([v[vk], v_u.get(vk, torch.zeros_like(v[vk]))], dim=0) for vk in v} if isinstance(v, dict)
            else None for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())
        }
    
    csd_model = setup_csd(device=device)
    csd_model = csd_model.to(device=device_2)
    cosine_loss = torch.nn.CosineSimilarity(dim=1)

    x0_style_forward = x0_style_forward.to(device_2)
    x0_forward = x0_forward.to(device_2)

    models.previewer.to(device=device_2)
    org_style = models.previewer(x0_style_forward)
    _, _, org_style_embedding = csd_model(
      extras.clip_preprocess(org_style)
    )
    
    org_content = models.previewer(x0_forward)
    _, org_content_embedding, _ = csd_model(
      extras.clip_preprocess(org_content)
    )

    org_style_embedding = org_style_embedding.to(device=device_2)
    org_content_embedding = org_content_embedding.to(device=device_2)
    
    for i in range(0, timesteps):

        noise_cond = self.noise_cond(logSNR_range[i])

        x = x.detach()
        x.requires_grad_(False)

        u = torch.zeros_like(x)

        if cfg is not None and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop) and (cfg_t_start is None or r_range[i].item() <= cfg_t_start):
            cfg_val = cfg
            if isinstance(cfg_val, (list, tuple)):
                assert len(cfg_val) == 2, "cfg must be a float or a list/tuple of length 2"
                cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (1-r_range[i].item())
            pred, pred_unconditional = model(torch.cat([x, x], dim=0), noise_cond.repeat(2), **model_inputs).chunk(2)
            pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
            if cfg_rho > 0:
                std_pos, std_cfg = pred.std(),  pred_cfg.std()
                pred = cfg_rho * (pred_cfg * std_pos/(std_cfg+1e-9)) + pred_cfg * (1-cfg_rho)
            else:
                pred = pred_cfg
        else:
            pred = model(x, noise_cond, **model_inputs)
        
        if (t_guidance_end < noise_cond) and (noise_cond < t_guidance_start):
          x = x.to(device_2)
          x.requires_grad_(True)
          x0_hat, epsilon = self.undiffuse(x, logSNR_range[i].to(device_2), pred.to(device_2))
        else:
          x0_hat, epsilon = self.undiffuse(x, logSNR_range[i], pred)

        print(x0_hat.device, x.device)

        print(f"\n\nTimestep: {i}\n-Noise_cond = {noise_cond}")

        if (t_guidance_end < noise_cond) and (noise_cond < t_guidance_start):

          x0_hat_previewer = models.previewer(x0_hat)
          _, this_content, this_style = csd_model(
              extras.clip_preprocess(x0_hat_previewer)
          )

          loss_style = torch.linalg.norm(org_style_embedding - this_style, ord=2) 
        #   loss_style = 2*((1 - cosine_loss(org_style_embedding, this_style)).abs().mean())
          g_s = torch.autograd.grad(loss_style, x, retain_graph=True)[0]
          del loss_style
          g_s = 1 / (torch.linalg.norm(g_s) + 1e-8) * g_s

          loss_content = torch.linalg.norm(org_content_embedding - this_content, ord=2)
          g_c = torch.autograd.grad(loss_content, x, retain_graph=False)[0]
          del loss_content
          g_c = 1 / (torch.linalg.norm(g_c) + 1e-8) * g_c
          del x0_hat_previewer
          
          g_s = g_s.to(device=device)
          g_c = g_c.to(device=device)
          x0_hat = x0_hat.detach().to(device=device)
          x = x.detach().to(device=device)
          dot_product = torch.sum(g_s * g_c)
          if dot_product < 0: 
            u = g_s - Lambda * dot_product * g_c
          else:
            u = g_s 
          
        #   u = u - torch.sum(u * pred) / (torch.linalg.norm(pred)**2 + 1e-8) * pred
          u = u * (torch.linalg.norm(pred) / (torch.linalg.norm(u) + 1e-8))
        #   max_val = torch.quantile(torch.abs(pred).float(), 0.99)
        #   max_val = 0.9 * torch.max(torch.abs(pred))
        #   u = torch.clamp(u, min=-max_val, max=max_val)

          epsilon = pred - eta * (1 - alpha_cumprod[i]).sqrt() * u
          print(f"\n- max_u = {u.max().item()}, min_u = {u.min().item()}")
          print(f"\n-max_pred = {pred.max().item()}, min_pred = {pred.min().item()}")
          print(f"\n-max_epsilon_tilde = {epsilon.max().item()}, min_epsilon_tilde = {epsilon.min().item()}")
        if apply_pushforward and x.shape[0] > 1 and i < tau_pushforward:
          xt_forward, epsilon_forward, pos, scale = self.diffuse_forward(x0_style_forward, logSNR_range[i], epsilon)
          mean_src = x.mean(axis=0).unsqueeze(dim=0)
          std_src = x.std(axis=0).unsqueeze(dim=0)
          x_norm = (x - mean_src) / (std_src + 1e-15)
          x_ada_in = x_norm * scale + x0_style_forward * pos 
          x = (1 - gamma_pushforward) * x + gamma_pushforward * x_ada_in
        
        x = sampler(x, x0_hat, epsilon, logSNR_range[i], logSNR_range[i+1], **sampler_params).requires_grad_(False)

        altered_vars = yield (x0_hat, x, epsilon)

        # Update some running variables if the user wants
        if altered_vars is not None:
            cfg = altered_vars.get('cfg', cfg)
            cfg_rho = altered_vars.get('cfg_rho', cfg_rho)
            sampler = altered_vars.get('sampler', sampler)
            model_inputs = altered_vars.get('model_inputs', model_inputs)
            x = altered_vars.get('x', x)
            x_init = altered_vars.get('x_init', x_init)  

#   def sampleRBM(
#       self,
#       model: torch.nn.Module,
#       model_inputs: Dict[str, Any],
#       shape: Tuple,
#       unconditional_inputs: Dict[str, Any] = None,
#       sampler: SimpleSampler = None,
#       schedule: BaseSchedule = None,
#       t_start: float = 1.0,
#       t_end: float = 0.0,
#       timesteps: float = 20,
#       x_init: torch.Tensor = None,
#       cfg: float = 3.0,
#       cfg_t_stop: int = None,
#       cfg_t_start: int = None,
#       cfg_rho: float = 0.7,
#       sampler_params: Dict[str, Any] = None,
#       shift: int = 1,
#       device: str = "cpu",
#       x0_forward: torch.Tensor = None,
#       x0_style_forward: torch.Tensor = None,
#       num_iter: int = 3,
#       eta: float = 1e-1,
#       tau: int = 20,
#       eval_csd: bool = False,
#       eval_sub_csd: bool = False,
#       apply_pushforward: bool = False,
#       tau_pushforward: int = 0,
#       tau_pushforward_csd: int = 0,
#       lam_content: float = 1.0,
#       lam_style: float = 1.0,
#       lam_txt_alignment: float = 0.0,
#       use_attn_mask: bool = False,
#       save_attn_mask: bool = False,
#       models: WurstCoreC.Models = None,
#       extras: WurstCoreC.Extras = None,
#       sam_mask: float = 1.0,
#       sam_prompt: str = None,
#       use_sam_mask: bool = False,
#       use_ddim_sampler: bool = False,
#   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     Implementation of Stochastic Optimal Control for reference-based modulation.

#     Args:
#         model: StageC model in StableCascade.
#         model_inputs: Input keyword arguments to the model.
#         shape: Shape of the StageC model's latent space.
#         unconditional_inputs: Unconditional input keyword arguments to the
#           model. Defaults to None.
#         sampler: Sampler used in the reverse diffusion process. Defaults to
#           None.
#         schedule: Scheduler used in the reverse diffusion process. Defaults to
#           None.
#         t_start: LogSNR start time. Defaults to 1.0.
#         t_end: LogSNR end time. Defaults to 0.0.
#         timesteps: Number of diffusion timesteps. Defaults to 20.
#         x_init: Initialized latents. Defaults to None.
#         cfg: Configuration parameter for guidance. Defaults to 3.0.
#         cfg_t_stop: Time step to stop configuration. Defaults to None.
#         cfg_t_start: Time step to start configuration. Defaults to None.
#         cfg_rho: Configuration rho parameter for guidance. Defaults to 0.7.
#         sampler_params: Additional parameters for the sampler. Defaults to None.
#         shift: Shift parameter for the schedule. Defaults to 1.
#         device: Device to run the sampling on ("cpu" or "cuda"). Defaults to
#           "cpu".
#         x0_forward: Initial forward latents. Defaults to None.
#         x0_style_forward: Initial style forward latents. Defaults to None.
#         num_iter: Number of iterations for latent refinement. Defaults to 3.
#         eta: Learning rate for latent refinement. Defaults to 1e-1.
#         tau: Number of timesteps for latent refinement. Defaults to 20.
#         eval_csd: Flag to evaluate content-style decomposition. Defaults to
#           False.
#         eval_sub_csd: Flag to evaluate subject-style decomposition. Defaults to
#           False.
#         apply_pushforward: Flag to apply pushforward transformation. Defaults to
#           False.
#         tau_pushforward: Number of timesteps for pushforward transformation.
#           Defaults to 0.
#         tau_pushforward_csd: Number of timesteps for content-style pushforward
#           transformation. Defaults to 0.
#         lam_content: Weight for content loss. Defaults to 1.0.
#         lam_style: Weight for style loss. Defaults to 1.0.
#         lam_txt_alignment: Weight for faithfulness to the original dynamics. Defaults to 0.0.
#         use_attn_mask: Flag to use attention mask. Defaults to False.
#         save_attn_mask: Flag to save attention mask. Defaults to False.
#         models: Models used in the framework. Defaults to None.
#         extras: Extra configurations and utilities. Defaults to None.
#         sam_mask: SAM mask value. Defaults to 1.0.
#         sam_prompt: SAM prompt for mask generation. Defaults to None.
#         use_sam_mask: Flag to use SAM mask. Defaults to False.
#         use_ddim_sampler: Flag to use DDIM sampler instead of DDPM sampler.
#           Defaults to False.

#     Yields:
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
#         latent variable (x0), the current state (x), and the prediction (pred).
    # """
    # sampler_params = {} if sampler_params is None else sampler_params
    # if sampler is None:
    #   sampler = DDPMSampler(self)
    # if use_ddim_sampler:
    #   sampler = DDIMSampler(self)
    # r_range = torch.linspace(t_start, t_end, timesteps + 1)
    # schedule = self.schedule if schedule is None else schedule
    # logSNR_range = (
    #     schedule(r_range, shift=shift)[:, None]
    #     .expand(-1, shape[0] if x_init is None else x_init.size(0))
    #     .to(device)
    # )

    # x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
    # if cfg is not None:
    #   if unconditional_inputs is None:
    #     unconditional_inputs = {
    #         k: torch.zeros_like(v) for k, v in model_inputs.items()
    #     }
    #   model_inputs = {
    #       k: (
    #           torch.cat([v, v_u], dim=0)
    #           if isinstance(v, torch.Tensor)
    #           else (
    #               [
    #                   (
    #                       torch.cat([vi, vi_u], dim=0)
    #                       if isinstance(vi, torch.Tensor)
    #                       and isinstance(vi_u, torch.Tensor)
    #                       else None
    #                   )
    #                   for vi, vi_u in zip(v, v_u)
    #               ]
    #               if isinstance(v, list)
    #               else (
    #                   {
    #                       vk: torch.cat(
    #                           [v[vk], v_u.get(vk, torch.zeros_like(v[vk]))],
    #                           dim=0,
    #                       )
    #                       for vk in v
    #                   }
    #                   if isinstance(v, dict)
    #                   else None
    #               )
    #           )
    #       )
    #       for (k, v), (k_u, v_u) in zip(
    #           model_inputs.items(), unconditional_inputs.items()
    #       )
    #   }
    # csd_model = setup_csd(device=device)
    # cosine_loss = torch.nn.CosineSimilarity(dim=1)
    # sam_model = LangSAM()

    # for i in range(0, timesteps):
    #   noise_cond = self.noise_cond(logSNR_range[i])
    #   if (
    #       cfg is not None
    #       and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
    #       and (cfg_t_start is None or r_range[i].item() <= cfg_t_start)
    #   ):
    #     cfg_val = cfg
    #     if isinstance(cfg_val, (list, tuple)):
    #       assert (
    #           len(cfg_val) == 2
    #       ), "cfg must be a float or a list/tuple of length 2"
    #       cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (
    #           1 - r_range[i].item()
    #       )
    #     ## Generate predictions.
    #     with torch.no_grad():
    #       with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #         pred, pred_unconditional = model(
    #             torch.cat([x, x], dim=0),
    #             noise_cond.repeat(2),
    #             **model_inputs,
    #         ).chunk(2)
    #     pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
    #     if cfg_rho > 0:
    #       std_pos, std_cfg = pred.std(), pred_cfg.std()
    #       pred = cfg_rho * (
    #           pred_cfg * std_pos / (std_cfg + 1e-9)
    #       ) + pred_cfg * (1 - cfg_rho)
    #     else:
    #       pred = pred_cfg
    #   else:
    #     pred = model(x, noise_cond, **model_inputs)
    #   x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)

    #   #######################################################
    #   ## Stochastic Optimal Control block
    #   #######################################################
    #   if i < tau:
    #     if eval_csd:
    #       z0 = x0.clone().detach()
    #       z0.requires_grad = True
    #       optimizer = torch.optim.Adam(
    #           [z0], lr=eta * (1.0 - i / timesteps)
    #       )  # decreasing stepsize schedule

    #       org_style = models.previewer(x0_style_forward)
    #       bb_feats2, content_embeddings2, style_embeddings2 = csd_model(
    #           extras.clip_preprocess(org_style)
    #       )

    #       for _ in range(num_iter):
    #         pred_image = models.previewer(z0)
    #         bb_feats1, content_embeddings1, style_embeddings1 = csd_model(
    #             extras.clip_preprocess(pred_image)
    #         )
    #         # Measure style similarity.
    #         style_loss = (
    #             (1 - cosine_loss(style_embeddings1, style_embeddings2))
    #             .abs()
    #             .mean()
    #         )
    #         # Measure faithfulness to original dynamics.
    #         txt_alignment_loss = nnf.mse_loss(z0, x0.detach())

    #         loss = (
    #             lam_style * style_loss + lam_txt_alignment * txt_alignment_loss
    #         )

    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()

    #         style_correlation = -(
    #             style_embeddings1 @ style_embeddings2.T
    #         ).mean()
    #         content_correlation = -(
    #             content_embeddings1 @ content_embeddings2.T
    #         ).mean()

    #       x0 = z0.detach()
    #     elif eval_sub_csd:
    #       z0 = x0.clone().detach()
    #       z0.requires_grad = True
    #       optimizer = torch.optim.Adam(
    #           [z0], lr=eta * (1.0 - i / timesteps)
    #       )

    #       # 16 x 24 x 24
    #       org_image = models.previewer(x0_forward)
    #       org_style = models.previewer(x0_style_forward)
    #       # 3 x 192 x 192
    #       if use_sam_mask:
    #         bb_feats1, content_embeddings1, style_embeddings1 = csd_model(
    #             extras.clip_preprocess(org_image * sam_mask)
    #         )
    #       else:
    #         bb_feats1, content_embeddings1, style_embeddings1 = csd_model(
    #             extras.clip_preprocess(org_image)
    #         )
    #       bb_feats2, content_embeddings2, style_embeddings2 = csd_model(
    #           extras.clip_preprocess(org_style)
    #       )

    #       for _ in range(num_iter):
    #         pred_image = models.previewer(z0)
    #         ##############################################
    #         ## use sam mask for the predicted image (optional)
    #         ##############################################
    #         if use_attn_mask and x.shape[0] == 1:
    #           attn_mask, boxes, phrases, logits = sam_model.predict(
    #               transform(pred_image[0].detach().to(torch.float32)),
    #               sam_prompt,
    #           )
    #           if len(boxes):
    #             if len(boxes) > 1:
    #               attn_mask = attn_mask[:1]
    #             attn_mask = attn_mask.detach().unsqueeze(dim=0).to(device)
    #             if save_attn_mask:
    #               plt.imsave(
    #                   f"results/sam_mask_pred_step_{i}.png",
    #                   (attn_mask).float().cpu().clamp(0, 1).numpy()[0, 0],
    #               )
    #           _, content_embeddings, _ = csd_model(
    #             extras.clip_preprocess(pred_image * attn_mask)
    #           )
    #         ##############################################
    #         else:
    #           _, content_embeddings, _ = csd_model(extras.clip_preprocess(pred_image))
    #         bb_feats, _, style_embeddings = csd_model(
    #             extras.clip_preprocess(pred_image)
    #         )
    #         content_loss = (
    #             (1 - cosine_loss(content_embeddings, content_embeddings1))
    #             .abs()
    #             .mean()
    #         )
    #         style_loss = (
    #             (1 - cosine_loss(style_embeddings, style_embeddings2))
    #             .abs()
    #             .mean()
    #         )
    #         txt_alignment_loss = nnf.mse_loss(z0, x0.detach())

    #         ## Compose subject+style loss.
    #         loss = (
    #             lam_content * content_loss
    #             + lam_style * style_loss
    #             + lam_txt_alignment * txt_alignment_loss
    #         )

    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()

    #       x0 = z0.detach()

    #   # ############################################################
    #   # Pushforward block (optional)
    #   # ############################################################
    #   if apply_pushforward and x.shape[0] > 1:
    #     if eval_sub_csd:
    #       if i < tau_pushforward:
    #         xt_forward, epsilon_forward, pos, scale = self.diffuse_forward(
    #             x0_forward, logSNR_range[i], epsilon
    #         )
    #         mean_src = x.mean(axis=0).unsqueeze(dim=0)
    #         std_src = x.std(axis=0).unsqueeze(dim=0)
    #         x_norm = (x - mean_src) / (std_src + 1e-15)
    #         x = x_norm * scale + x0_forward * pos
    #       elif i < tau_pushforward_csd:
    #         xt_forward, epsilon_forward, pos, scale = self.diffuse_forward(
    #             x0_style_forward, logSNR_range[i], epsilon
    #         )
    #         mean_src = x.mean(axis=0).unsqueeze(dim=0)
    #         std_src = x.std(axis=0).unsqueeze(dim=0)
    #         x_norm = (x - mean_src) / (std_src + 1e-15)
    #         x = x_norm * scale + x0_style_forward * pos
    #     elif eval_csd and i < tau_pushforward:
    #       xt_forward, epsilon_forward, pos, scale = self.diffuse_forward(
    #           x0_style_forward, logSNR_range[i], epsilon
    #       )
    #       mean_src = x.mean(axis=0).unsqueeze(dim=0)
    #       std_src = x.std(axis=0).unsqueeze(dim=0)
    #       x_norm = (x - mean_src) / (std_src + 1e-15)
    #       x = x_norm * scale + x0_style_forward * pos

    #   x = sampler(
    #       x, x0, epsilon, logSNR_range[i], logSNR_range[i + 1], **sampler_params
    #   )
    #   altered_vars = yield (x0, x, pred)

    #   # Update some running variables if the user wants.
    #   if altered_vars is not None:
    #     cfg = altered_vars.get("cfg", cfg)
    #     cfg_rho = altered_vars.get("cfg_rho", cfg_rho)
    #     sampler = altered_vars.get("sampler", sampler)
    #     model_inputs = altered_vars.get("model_inputs", model_inputs)
    #     x = altered_vars.get("x", x)
    #     x_init = altered_vars.get("x_init", x_init)
