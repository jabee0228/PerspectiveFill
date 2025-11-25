import os
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch
from .ldm.models.diffusion.ddim import DDIMSampler
from .inpaint_utils import seed_everything, make_batch
from .ldm.util import instantiate_from_config
from contextlib import suppress
from typing import Optional
seed_everything(42)

# ---------------------------------------------------------
# Stable Diffusion Inpainting backend helper functions
# ---------------------------------------------------------
_INPAINT_MODEL = None
_DDIM_SAMPLER = None
_INPAINT_DEVICE = None
_INPAINT_CFG = None
_INPAINT_EMA = False

# 預設的 config/ckpt 來源，可用環境變數覆蓋
DEFAULT_YAML = os.getenv('INPAINT_YAML', './module/StableDiffsuion/configs/latent-diffusion/inpainting_example_overfit.yaml')
DEFAULT_CKPT = os.getenv('INPAINT_CKPT', './module/StableDiffsuion/weight/sd.ckpt')  # 沒提供就要求呼叫者先設定

def init_inpaint_model(ckpt_path: str,
                       yaml_profile: str,
                       device: str = 'cpu',
                       use_ema: bool = False) -> None:
    """Initialize and cache the inpainting model & sampler.
    Safe to call multiple times; will reuse existing model if params unchanged.
    """
    global _INPAINT_MODEL, _DDIM_SAMPLER, _INPAINT_DEVICE, _INPAINT_CFG, _INPAINT_EMA

    # If already initialized with same parameters, skip
    if (_INPAINT_MODEL is not None and _INPAINT_CFG == yaml_profile and
            _INPAINT_DEVICE == device and _INPAINT_EMA == use_ema):
        return

    # Resolve device
    if torch.cuda.is_available() and device.startswith('cuda'):
        dev = torch.device(device)
    else:
        dev = torch.device('cpu')

    if not os.path.isfile(yaml_profile):
        raise FileNotFoundError(f"yaml_profile not found: {yaml_profile}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    cfg = OmegaConf.load(yaml_profile)
    cfg.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(cfg.model)

    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[init_inpaint_model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[init_inpaint_model] Unexpected keys: {len(unexpected)}")

    model = model.to(dev)
    sampler = DDIMSampler(model)

    _INPAINT_MODEL = model
    _DDIM_SAMPLER = sampler
    _INPAINT_DEVICE = dev
    _INPAINT_CFG = yaml_profile
    _INPAINT_EMA = use_ema


def run_inpaint(origin_image_path: str,
                ref_image_path: str,
                mask_path: str,
                ckpt_path: str,
                yaml_profile: str,
                steps: int = 50,
                resize: int = 512,
                device: str = 'cpu',
                use_ema: bool = False,
                white_part: bool = False,
                save_path: str = './uploads/inpainted.jpg') -> bytes:
    """Run a single inpainting inference.

    Parameters:
      origin_image_path: path to original image (background base)
      ref_image_path: path to reference (used as masked_image when white_part=False)
      mask_path: path to binary mask (white region to inpaint). Will be binarized internally.
      ckpt_path / yaml_profile: model weight & config
      steps: DDIM sampling steps
      resize: images resized to square size (e.g., 512)
      device: cpu | cuda | cuda:0 ...
      use_ema: whether to use EMA scope
      white_part: if True, ignore ref_image_path and generate masked_image from origin *(1-mask)*image
      save_path: output file path; saved as JPEG

    Returns:
      JPEG bytes of the composited inpainted image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize model if needed (or re-init if params differ)
    init_inpaint_model(ckpt_path=ckpt_path, yaml_profile=yaml_profile, device=device, use_ema=use_ema)

    model = _INPAINT_MODEL
    sampler = _DDIM_SAMPLER
    dev = _INPAINT_DEVICE

    # Build batch (make_batch handles mask binarization). When white_part=False we pass ref_image_path.
    batch = make_batch(origin_image_path, mask_path, ref_image_path, device=str(dev), resize_to=resize, white_part=white_part)

    # Encode masked image
    c_masked = model.cond_stage_model.encode(batch["masked_image"])  # latent of masked input
    # Resize mask to latent size
    cc_mask = torch.nn.functional.interpolate(batch["mask"], size=c_masked.shape[-2:])
    # Conditioning concat
    c = torch.cat((c_masked, cc_mask), dim=1)

    shape = (3,) + c_masked.shape[2:]

    scope_ctx = model.ema_scope if use_ema else suppress
    with torch.no_grad():
        with scope_ctx("Sampling"):
            samples_ddim, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)

            # Normalize tensors to [0,1]
            image = torch.clamp((batch["image"] + 1.0) / 2.0, 0.0, 1.0)
            mask = torch.clamp((batch["mask"] + 1.0) / 2.0, 0.0, 1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)

            # Composite: keep non-masked original, fill masked with prediction
            inpainted = (1 - mask) * image + mask * predicted_image

    # To numpy uint8
    inpainted_np = (inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)

    # Save
    Image.fromarray(inpainted_np).save(save_path, format='JPEG')

    # Encode to JPEG bytes
    from io import BytesIO
    bio = BytesIO()
    Image.fromarray(inpainted_np).save(bio, format='JPEG')
    return bio.getvalue()


def run_inpaint_simple(origin_image_path: str,
                       ref_image_path: str,
                       mask_path: str,
                       steps: int = 50,
                       resize: int = 512,
                       white_part: bool = False,
                       device: str = 'cpu',
                       use_ema: bool = False) -> bytes:
    """Simplified backend-friendly inference.

    Arguments:
      origin_image_path: 原始影像 (保持未遮罩區域)
      ref_image_path: 參考影像 (white_part=False 時作為被遮罩輸入來源)
      mask_path: 二值遮罩 (白=需要生成, 黑=保留原圖) 可為灰階將以 0.5 門檻二值化
      steps: DDIM 取樣步數 (品質/速度權衡)
      resize: 將輸入影像與遮罩等比例縮放到的邊長 (正方形)
      white_part: True -> masked_image = (1-mask)*origin；False -> 使用 ref 影像取代
      device: 'cpu' 或 'cuda[:index]'，自動 fallback 至 CPU 若不可用
      use_ema: 是否使用 EMA 權重範圍 (若模型支援)

    Returns:
      生成後合成完成的最終 inpaint 圖像之 JPEG bytes。

    Side effects:
      會將結果儲存於 ./upload/inpainted.jpg 方便前端直接讀取檔案。
    """
    if not os.path.isfile(origin_image_path):
        raise FileNotFoundError(f"origin_image_path not found: {origin_image_path}")
    if not os.path.isfile(ref_image_path):
        raise FileNotFoundError(f"ref_image_path not found: {ref_image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"mask_path not found: {mask_path}")

    if DEFAULT_CKPT is None:
        raise RuntimeError("必須設定環境變數 INPAINT_CKPT 指向 Stable Diffusion inpaint 權重檔 (.ckpt)。")

    ckpt_path = DEFAULT_CKPT
    yaml_path = DEFAULT_YAML

    model = _INPAINT_MODEL
    sampler = _DDIM_SAMPLER
    dev = _INPAINT_DEVICE
    if model is None or sampler is None:
        raise RuntimeError("Inpaint model 尚未初始化成功。")

    batch = make_batch(origin_image_path, mask_path, ref_image_path, device=str(dev), resize_to=resize, white_part=white_part)

    # 取樣流程
    scope_ctx = model.ema_scope if (use_ema and hasattr(model, 'ema_scope')) else suppress
    with torch.no_grad():
        with scope_ctx():  # ema_scope 不一定需要參數; suppress() 忽略例外
            c_masked = model.cond_stage_model.encode(batch['masked_image'])
            cc_mask = torch.nn.functional.interpolate(batch['mask'], size=c_masked.shape[-2:])
            c = torch.cat((c_masked, cc_mask), dim=1)
            shape = (3,) + c_masked.shape[2:]
            samples_ddim, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False
            )
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            image = torch.clamp((batch['image'] + 1.0) / 2.0, 0.0, 1.0)
            mask = torch.clamp((batch['mask'] + 1.0) / 2.0, 0.0, 1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)
            inpainted = (1 - mask) * image + mask * predicted_image

    # 轉為 numpy / 儲存與回傳
    inpainted_np = (inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)
    save_dir = './upload'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'inpainted.jpg')
    Image.fromarray(inpainted_np).save(save_path, format='JPEG')

    from io import BytesIO
    bio = BytesIO()
    Image.fromarray(inpainted_np).save(bio, format='JPEG')
    return bio.getvalue()

__all__ = ['init_inpaint_model', 'run_inpaint', 'run_inpaint_simple']
