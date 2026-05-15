from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from gdf import BaseSchedule, DDIMSampler, DDPMSampler, GDF, SimpleSampler
from train import WurstCoreC
from utils import setup_csd


@dataclass
class AJDCConfig:
    # diffusion / sampling
    timesteps: int = 20
    t_start: float = 1.0
    t_end: float = 0.0
    shift: int = 1
    use_ddim_sampler: bool = False
    device: str = "cuda"

    # classifier-free guidance
    cfg: Optional[float] = 3.0
    cfg_t_stop: Optional[int] = None
    cfg_t_start: Optional[int] = None
    cfg_rho: float = 0.7

    # adaptive jump control
    jump_candidates: Tuple[int, ...] = (1, 2, 4)
    internal_substeps: int = 1

    # finite-difference sensitivity
    fd_eps: float = 5e-3

    # objective weights
    lambda_content: float = 1.0
    lambda_leakage: float = 1.0
    beta: float = 1e-2
    alpha_jump: float = 0.05
    eta_err: float = 0.1

    # safety
    leakage_threshold: float = 0.0
    safety_margin: float = 0.02
    a_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class AJDC(GDF):
    """
    Adaptive Jump-and-Drift Control for training-free diffusion stylization.

    Action at each decision step:
        1) choose jump length Delta
        2) choose safe drift a = [a_style, a_content, a_leak]

    Notes:
        - This is a clean scaffold meant to replace the inner optimization loop
          of RB-Modulation.
        - You will likely still need to adapt tensor shapes and previewer I/O
          to the exact repo version you are using.
    """

    def sample(
        self,
        model: torch.nn.Module,
        model_inputs: Dict[str, Any],
        shape: Tuple[int, ...],
        unconditional_inputs: Optional[Dict[str, Any]] = None,
        sampler: Optional[SimpleSampler] = None,
        schedule: Optional[BaseSchedule] = None,
        sampler_params: Optional[Dict[str, Any]] = None,
        x_init: Optional[torch.Tensor] = None,
        x0_content_forward: Optional[torch.Tensor] = None,
        x0_style_forward: Optional[torch.Tensor] = None,
        models: Optional[WurstCoreC.Models] = None,
        extras: Optional[WurstCoreC.Extras] = None,
        config: Optional[AJDCConfig] = None,
    ):
        assert models is not None, "models is required"
        assert extras is not None, "extras is required"
        assert x0_content_forward is not None, "x0_content_forward is required"
        assert x0_style_forward is not None, "x0_style_forward is required"

        cfg = config or AJDCConfig()
        sampler_params = {} if sampler_params is None else sampler_params

        if sampler is None:
            sampler = DDIMSampler(self) if cfg.use_ddim_sampler else DDPMSampler(self)

        schedule = self.schedule if schedule is None else schedule
        r_range = torch.linspace(cfg.t_start, cfg.t_end, cfg.timesteps + 1, device=cfg.device)
        logsnr_range = schedule(r_range, shift=cfg.shift)[:, None]

        if x_init is None:
            x = sampler.init_x(shape).to(cfg.device)
        else:
            x = x_init.clone().to(cfg.device)

        if cfg.cfg is not None:
            unconditional_inputs = self._prepare_unconditional_inputs(
                model_inputs, unconditional_inputs
            )
            model_inputs = self._merge_cfg_inputs(model_inputs, unconditional_inputs)

        csd_model = setup_csd(device=cfg.device)
        cosine = torch.nn.CosineSimilarity(dim=1)

        # fixed anchors from content / style images
        with torch.no_grad():
            content_img = models.previewer(x0_content_forward)
            style_img = models.previewer(x0_style_forward)

            _, content_target, _ = csd_model(extras.clip_preprocess(content_img))
            _, style_content_target, style_target = csd_model(
                extras.clip_preprocess(style_img)
            )

        i = 0
        while i < cfg.timesteps:
            best_candidate: Optional[Dict[str, Any]] = None

            for jump in self._valid_jumps(i, cfg.timesteps, cfg.jump_candidates):
                j = i + jump
                candidate = self._evaluate_candidate_jump(
                    model=model,
                    x=x,
                    i=i,
                    j=j,
                    r_range=r_range,
                    logsnr_range=logsnr_range,
                    model_inputs=model_inputs,
                    sampler=sampler,
                    sampler_params=sampler_params,
                    models=models,
                    extras=extras,
                    csd_model=csd_model,
                    cosine=cosine,
                    style_target=style_target,
                    content_target=content_target,
                    style_content_target=style_content_target,
                    cfg=cfg,
                )

                if best_candidate is None or candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate

            assert best_candidate is not None

            x = best_candidate["x_next"]
            i = best_candidate["next_idx"]

            altered_vars = yield (
                best_candidate["x0_ctrl"],
                x,
                best_candidate["pred_ctrl"],
            )
            if altered_vars is not None:
                x = altered_vars.get("x", x)

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidate_jump(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        i: int,
        j: int,
        r_range: torch.Tensor,
        logsnr_range: torch.Tensor,
        model_inputs: Dict[str, Any],
        sampler: SimpleSampler,
        sampler_params: Dict[str, Any],
        models: WurstCoreC.Models,
        extras: WurstCoreC.Extras,
        csd_model: torch.nn.Module,
        cosine: torch.nn.Module,
        style_target: torch.Tensor,
        content_target: torch.Tensor,
        style_content_target: torch.Tensor,
        cfg: AJDCConfig,
    ) -> Dict[str, Any]:
        # uncontrolled rollout
        x_tau_0, x0_tau_0, pred_tau_0 = self._rollout_indices(
            model=model,
            x_start=x,
            i=i,
            j=j,
            r_range=r_range,
            logsnr_range=logsnr_range,
            model_inputs=model_inputs,
            sampler=sampler,
            sampler_params=sampler_params,
            basis=None,
            a_star=None,
            models=models,
            extras=extras,
            csd_model=csd_model,
            cosine=cosine,
            style_target=style_target,
            content_target=content_target,
            style_content_target=style_content_target,
            cfg=cfg,
        )

        # gradients and values at uncontrolled terminal state
        grad_style, grad_content, grad_leak, F0, L0 = self._objective_grads(
            x0_latent=x0_tau_0,
            models=models,
            extras=extras,
            csd_model=csd_model,
            cosine=cosine,
            style_target=style_target,
            content_target=content_target,
            style_content_target=style_content_target,
            cfg=cfg,
        )

        basis = self._make_basis(grad_style, grad_content, grad_leak)

        # finite-difference sensitivity wrt a = [a_style, a_content, a_leak]
        M = self._estimate_sensitivity_fd(
            model=model,
            x=x,
            i=i,
            j=j,
            r_range=r_range,
            logsnr_range=logsnr_range,
            model_inputs=model_inputs,
            sampler=sampler,
            sampler_params=sampler_params,
            basis=basis,
            base_x0=x0_tau_0.detach(),
            models=models,
            extras=extras,
            csd_model=csd_model,
            cosine=cosine,
            style_target=style_target,
            content_target=content_target,
            style_content_target=style_content_target,
            cfg=cfg,
        )

        grad_F = (grad_style - cfg.lambda_content * grad_content).reshape(-1)
        grad_L = grad_leak.reshape(-1)

        g = M.transpose(0, 1) @ grad_F
        d = M.transpose(0, 1) @ grad_L
        H = self._approx_curvature(M, beta=cfg.beta)

        b = cfg.leakage_threshold - L0 - cfg.safety_margin
        a_star = self._solve_safe_tiny_qp(
            g=g,
            H=H,
            d=d,
            b=b,
            device=x.device,
            a_max=cfg.a_max,
        )

        # controlled rollout using chosen a_star
        x_ctrl, x0_ctrl, pred_ctrl = self._rollout_indices(
            model=model,
            x_start=x,
            i=i,
            j=j,
            r_range=r_range,
            logsnr_range=logsnr_range,
            model_inputs=model_inputs,
            sampler=sampler,
            sampler_params=sampler_params,
            basis=basis,
            a_star=a_star,
            models=models,
            extras=extras,
            csd_model=csd_model,
            cosine=cosine,
            style_target=style_target,
            content_target=content_target,
            style_content_target=style_content_target,
            cfg=cfg,
        )

        F_ctrl, L_ctrl = self._objective_values(
            x0_latent=x0_ctrl,
            models=models,
            extras=extras,
            csd_model=csd_model,
            cosine=cosine,
            style_target=style_target,
            content_target=content_target,
            style_content_target=style_content_target,
            cfg=cfg,
        )

        local_err = self._local_error_estimate(
            model=model,
            x=x,
            i=i,
            j=j,
            r_range=r_range,
            logsnr_range=logsnr_range,
            model_inputs=model_inputs,
            sampler=sampler,
            sampler_params=sampler_params,
            basis=basis,
            a_star=a_star,
            models=models,
            extras=extras,
            csd_model=csd_model,
            cosine=cosine,
            style_target=style_target,
            content_target=content_target,
            style_content_target=style_content_target,
            cfg=cfg,
        )

        jump_len = float(j - i)
        score = (
            F_ctrl
            - cfg.lambda_leakage * L_ctrl
            + cfg.alpha_jump * jump_len
            - cfg.eta_err * local_err
            - cfg.beta * float((a_star ** 2).sum().item())
        )

        return {
            "score": score,
            "jump": j - i,
            "next_idx": j,
            "a_star": a_star,
            "x_next": x_ctrl,
            "x0_ctrl": x0_ctrl,
            "pred_ctrl": pred_ctrl,
        }

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def _objective_values(
        self,
        x0_latent: torch.Tensor,
        models: WurstCoreC.Models,
        extras: WurstCoreC.Extras,
        csd_model: torch.nn.Module,
        cosine: torch.nn.Module,
        style_target: torch.Tensor,
        content_target: torch.Tensor,
        style_content_target: torch.Tensor,
        cfg: AJDCConfig,
    ) -> Tuple[float, float]:
        with torch.no_grad():
            pred_img = models.previewer(x0_latent)
            _, content_emb, style_emb = csd_model(extras.clip_preprocess(pred_img))

            style_dist = (1.0 - cosine(style_emb, style_target)).mean()
            content_dist = (1.0 - cosine(content_emb, content_target)).mean()

            # leakage: becoming too similar to structure/content of style image
            leakage = (1.0 - cosine(content_emb, style_content_target)).mean()

            F_val = float((-style_dist - cfg.lambda_content * content_dist).item())
            L_val = float(leakage.item())

        return F_val, L_val

    def _objective_grads(
        self,
        x0_latent: torch.Tensor,
        models: WurstCoreC.Models,
        extras: WurstCoreC.Extras,
        csd_model: torch.nn.Module,
        cosine: torch.nn.Module,
        style_target: torch.Tensor,
        content_target: torch.Tensor,
        style_content_target: torch.Tensor,
        cfg: AJDCConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        z = x0_latent.detach().clone().requires_grad_(True)
        pred_img = models.previewer(z)
        _, content_emb, style_emb = csd_model(extras.clip_preprocess(pred_img))

        style_dist = (1.0 - cosine(style_emb, style_target)).mean()
        content_dist = (1.0 - cosine(content_emb, content_target)).mean()
        leakage = (1.0 - cosine(content_emb, style_content_target)).mean()

        style_obj = -style_dist
        content_obj = content_dist
        leak_obj = leakage
        total_obj = style_obj - cfg.lambda_content * content_obj

        grad_style = torch.autograd.grad(style_obj, z, retain_graph=True)[0].detach()
        grad_content = torch.autograd.grad(content_obj, z, retain_graph=True)[0].detach()
        grad_leak = torch.autograd.grad(leak_obj, z, retain_graph=True)[0].detach()

        F_val = float(total_obj.item())
        L_val = float(leak_obj.item())
        return grad_style, grad_content, grad_leak, F_val, L_val

    # ------------------------------------------------------------------
    # Basis / drift
    # ------------------------------------------------------------------

    def _make_basis(
        self,
        grad_style: torch.Tensor,
        grad_content: torch.Tensor,
        grad_leak: torch.Tensor,
    ) -> torch.Tensor:
        """
        Basis U = [g_style, -g_content, -g_leak], normalized.
        Returns shape [3, *latent_shape]
        """
        b0 = self._normalize_tensor(grad_style)
        b1 = -self._normalize_tensor(grad_content)
        b2 = -self._normalize_tensor(grad_leak)
        return torch.stack([b0, b1, b2], dim=0)

    def _inject_drift_into_x0(
        self,
        x0: torch.Tensor,
        basis: torch.Tensor,
        a_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        x0 + a1*b_style + a2*b_content + a3*b_leak
        """
        drift = (
            a_star[0] * basis[0]
            + a_star[1] * basis[1]
            + a_star[2] * basis[2]
        )
        return x0 + drift

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_indices(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        i: int,
        j: int,
        r_range: torch.Tensor,
        logsnr_range: torch.Tensor,
        model_inputs: Dict[str, Any],
        sampler: SimpleSampler,
        sampler_params: Dict[str, Any],
        basis: Optional[torch.Tensor],
        a_star: Optional[torch.Tensor],
        models: WurstCoreC.Models,
        extras: WurstCoreC.Extras,
        csd_model: torch.nn.Module,
        cosine: torch.nn.Module,
        style_target: torch.Tensor,
        content_target: torch.Tensor,
        style_content_target: torch.Tensor,
        cfg: AJDCConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x_start.clone()
        last_x0: Optional[torch.Tensor] = None
        last_pred: Optional[torch.Tensor] = None

        for k in range(i, j):
            pred = self._predict_eps(
                model=model,
                x=x,
                i=k,
                r_range=r_range,
                logsnr_range=logsnr_range,
                model_inputs=model_inputs,
                cfg=cfg.cfg,
                cfg_t_stop=cfg.cfg_t_stop,
                cfg_t_start=cfg.cfg_t_start,
                cfg_rho=cfg.cfg_rho,
            )
            x0, eps = self.undiffuse(x, logsnr_range[k], pred)

            if basis is not None and a_star is not None:
                x0 = self._inject_drift_into_x0(x0, basis, a_star)

            x = sampler(
                x,
                x0,
                eps,
                logsnr_range[k],
                logsnr_range[k + 1],
                **sampler_params,
            )

            last_x0 = x0
            last_pred = pred

        assert last_x0 is not None
        assert last_pred is not None
        return x, last_x0, last_pred

    # ------------------------------------------------------------------
    # Sensitivity estimation
    # ------------------------------------------------------------------

    def _estimate_sensitivity_fd(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        i: int,
        j: int,
        r_range: torch.Tensor,
        logsnr_range: torch.Tensor,
        model_inputs: Dict[str, Any],
        sampler: SimpleSampler,
        sampler_params: Dict[str, Any],
        basis: torch.Tensor,
        base_x0: torch.Tensor,
        models: WurstCoreC.Models,
        extras: WurstCoreC.Extras,
        csd_model: torch.nn.Module,
        cosine: torch.nn.Module,
        style_target: torch.Tensor,
        content_target: torch.Tensor,
        style_content_target: torch.Tensor,
        cfg: AJDCConfig,
    ) -> torch.Tensor:
        """
        Finite-difference sensitivity of terminal x0 wrt a = [a_style, a_content, a_leak].
        Returns matrix M of shape [N, 3], where N = numel(x0).
        """
        eye = torch.eye(3, device=x.device, dtype=x.dtype)
        cols: List[torch.Tensor] = []

        for q in range(3):
            a = cfg.fd_eps * eye[q]

            _, x0_pert, _ = self._rollout_indices(
                model=model,
                x_start=x,
                i=i,
                j=j,
                r_range=r_range,
                logsnr_range=logsnr_range,
                model_inputs=model_inputs,
                sampler=sampler,
                sampler_params=sampler_params,
                basis=basis,
                a_star=a,
                models=models,
                extras=extras,
                csd_model=csd_model,
                cosine=cosine,
                style_target=style_target,
                content_target=content_target,
                style_content_target=style_content_target,
                cfg=cfg,
            )

            col = ((x0_pert - base_x0) / cfg.fd_eps).reshape(-1).detach()
            cols.append(col)

        return torch.stack(cols, dim=1)

    def _approx_curvature(self, M: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Simple PSD approximation:
            H = M^T M + 2 beta I
        """
        eye = torch.eye(M.shape[1], device=M.device, dtype=M.dtype)
        return M.transpose(0, 1) @ M + 2.0 * beta * eye

    def _local_error_estimate(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        i: int,
        j: int,
        r_range: torch.Tensor,
        logsnr_range: torch.Tensor,
        model_inputs: Dict[str, Any],
        sampler: SimpleSampler,
        sampler_params: Dict[str, Any],
        basis: torch.Tensor,
        a_star: torch.Tensor,
        models: WurstCoreC.Models,
        extras: WurstCoreC.Extras,
        csd_model: torch.nn.Module,
        cosine: torch.nn.Module,
        style_target: torch.Tensor,
        content_target: torch.Tensor,
        style_content_target: torch.Tensor,
        cfg: AJDCConfig,
    ) -> float:
        """
        Cheap local error surrogate:
            compare one jump i->j with split jump i->mid->j
        """
        if j - i <= 1:
            return 0.0

        mid = i + (j - i) // 2

        x_a, _, _ = self._rollout_indices(
            model, x, i, j, r_range, logsnr_range, model_inputs, sampler,
            sampler_params, basis, a_star, models, extras, csd_model, cosine,
            style_target, content_target, style_content_target, cfg
        )

        x_b, _, _ = self._rollout_indices(
            model, x, i, mid, r_range, logsnr_range, model_inputs, sampler,
            sampler_params, basis, a_star, models, extras, csd_model, cosine,
            style_target, content_target, style_content_target, cfg
        )
        x_b, _, _ = self._rollout_indices(
            model, x_b, mid, j, r_range, logsnr_range, model_inputs, sampler,
            sampler_params, basis, a_star, models, extras, csd_model, cosine,
            style_target, content_target, style_content_target, cfg
        )

        return float(((x_a - x_b) ** 2).mean().item())

    # ------------------------------------------------------------------
    # Safe tiny QP
    # ------------------------------------------------------------------

    def _solve_safe_tiny_qp(
        self,
        g: torch.Tensor,
        H: torch.Tensor,
        d: torch.Tensor,
        b: float,
        device: torch.device,
        a_max: Sequence[float],
    ) -> torch.Tensor:
        """
        Solve:
            max_a g^T a - 1/2 a^T H a
            s.t.  d^T a <= b,  0 <= a <= a_max
        via closed-form KKT + box projection.
        """
        H_inv = torch.linalg.inv(H)
        a_unc = H_inv @ g

        if float(d @ a_unc) <= b:
            a = a_unc
        else:
            denom = float(d @ (H_inv @ d)) + 1e-8
            numer = float(d @ (H_inv @ g)) - b
            mu = max(0.0, numer / denom)
            a = H_inv @ (g - mu * d)

        a = torch.clamp(a, min=0.0)
        a_hi = torch.tensor(a_max, device=device, dtype=a.dtype)
        a = torch.minimum(a, a_hi)
        return a

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _valid_jumps(
        self,
        i: int,
        timesteps: int,
        jumps: Iterable[int],
    ) -> List[int]:
        return [j for j in jumps if i + j <= timesteps]

    def _normalize_tensor(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        denom = torch.sqrt((x ** 2).mean()) + eps
        return x / denom

    def _prepare_unconditional_inputs(
        self,
        model_inputs: Dict[str, Any],
        unconditional_inputs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if unconditional_inputs is not None:
            return unconditional_inputs

        prepared = {}
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = torch.zeros_like(v)
            else:
                prepared[k] = v
        return prepared

    def _merge_cfg_inputs(
        self,
        model_inputs: Dict[str, Any],
        unconditional_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = {}
        for k, v in model_inputs.items():
            u = unconditional_inputs[k]
            if isinstance(v, torch.Tensor):
                merged[k] = torch.cat([v, u], dim=0)
            else:
                merged[k] = v
        return merged

    def _predict_eps(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        i: int,
        r_range: torch.Tensor,
        logsnr_range: torch.Tensor,
        model_inputs: Dict[str, Any],
        cfg: Optional[float],
        cfg_t_stop: Optional[int],
        cfg_t_start: Optional[int],
        cfg_rho: float,
    ) -> torch.Tensor:
        noise_cond = self.noise_cond(logsnr_range[i])

        use_cfg = (
            cfg is not None
            and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
            and (cfg_t_start is None or r_range[i].item() <= cfg_t_start)
        )

        if use_cfg:
            with torch.no_grad():
                pred, pred_uncond = model(
                    torch.cat([x, x], dim=0),
                    noise_cond.repeat(2),
                    **model_inputs,
                ).chunk(2)

                pred_cfg = torch.lerp(pred_uncond, pred, cfg)

                if cfg_rho > 0:
                    std_pos = pred.std()
                    std_cfg = pred_cfg.std()
                    pred = cfg_rho * (pred_cfg * std_pos / (std_cfg + 1e-9)) + (1.0 - cfg_rho) * pred_cfg
                else:
                    pred = pred_cfg
        else:
            pred = model(x, noise_cond, **model_inputs)

        return pred