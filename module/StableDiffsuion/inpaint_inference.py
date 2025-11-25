import argparse, os, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
# CHANGE WITH YOUR PATH
from .main_inpainting import instantiate_from_config
from .ldm.models.diffusion.ddim import DDIMSampler
from .inpaint_utils import seed_everything
from .inpaint_utils import make_batch,plot_row_original_mask_output
from contextlib import suppress
seed_everything(42)


# RUN SCRIPT
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting")
    
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
        required=True
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        required=True
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="path of weights to load",
        required=True        
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path of weights to load",
        required=True        
    )
    
    parser.add_argument(
        "--yaml_profile",
        type=str,
        help="yaml file describing the model to initialize",
        required=True        
    )


    parser.add_argument(
        "--resize",
        type=int,
        default = 512,
        help="resize to ",
    )
      
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="specify the device for inference (cpu, cuda, cuda:x)",
    )
      
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    
    parser.add_argument(
        "--ema",
        action='store_true',
        help="use ema weights",
    )
    parser.add_argument(
        "--white",
        type=str2bool,
        default=False,
        help="use white part or fixed",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="csv file of the dataset",
    )

    opt = parser.parse_args()

    if opt.csv is not None:
        import pandas as pd
        df = pd.read_csv(opt.csv)
        #df = df[df["partition"]=="validation"] # filter partition

        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(opt.indir, x))
        df["fixed_path"] = df["fixed_path"].apply(lambda x: os.path.join(opt.indir, x))
        df["mask_path"] = df["mask_path"].apply(lambda x: os.path.join(opt.indir, x))
        # df.to_csv(os.path.join(opt.indir, "dataset.csv"), index=False)
        masks = df["mask_path"].tolist()
        images = df["image_path"].tolist()
        fixeds = df["fixed_path"].tolist()
    else:
        masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.*")))

        images = [x.replace("_mask.jpg", "_raw.jpg") for x in masks]
        fixeds = [x.replace("_mask.jpg", "_fixed.jpg") for x in masks]

    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load(opt.yaml_profile)
    config.model.params.ckpt_path = opt.ckpt
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(opt.ckpt)["state_dict"],
                            strict=False)

    print("Loading modeling from %s" % opt.ckpt)
    
    device = torch.device(opt.device) if torch.cuda.is_available() and opt.device is not "cpu" else torch.device("cpu")
    
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    
    scope = model.ema_scope if opt.ema else suppress
    ema_prefix = "EMA" if opt.ema else "NOT_EMA"
    
    with torch.no_grad():
        with scope("Sampling"):
            for image, mask, fixed in tqdm(zip(images, masks, fixeds)):
                #outpath = os.path.join(opt.outdir, "%s_%s_%s_%s.jpg" % (os.path.split(image)[1].split(".")[0], opt.prefix, ema_prefix, os.path.basename(opt.ckpt)))
                outpath = os.path.join(opt.outdir, os.path.basename(image))
                batch = make_batch(image, mask, fixed, device=device, resize_to=opt.resize, white_part=opt.white)
                
                c_masked = model.cond_stage_model.encode(batch["masked_image"])
                                
                cc_mask = torch.nn.functional.interpolate(batch["mask"],
                                                        size=c_masked.shape[-2:])

                c = torch.cat((c_masked,cc_mask), dim=1)

                shape = (3,) + c_masked.shape[2:] # same

                
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                    conditioning=c,
                                                    batch_size=c.shape[0],
                                                    shape=shape,
                                                    verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                
                masked_image = torch.clamp((batch["masked_image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                min=0.0, max=1.0)


                inpainted = (1-mask)*image+mask*predicted_image
                
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                print("Save in %s" % outpath)
                
                mask = mask.cpu().numpy().transpose(0,2,3,1)[0]*255
                image = image.cpu().numpy().transpose(0,2,3,1)[0]*255
                masked_image = masked_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                
                # image_to_print = plot_row_original_mask_output([{"masked_image":masked_image, "image":image, "predicted_image":predicted_image}], image_size = 512)
                # Image.fromarray(image_to_print.astype(np.uint8)).save(outpath)
                Image.fromarray(predicted_image.astype(np.uint8)).save(outpath)