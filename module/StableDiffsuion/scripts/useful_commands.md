# IN INPAITING FOLDER
wget -O models/ldm/inpainting_big/model.ckpt https://ommer-lab.com/files/latent-diffusion/inpainting_big.zip --no-check-certificate

## DYNAFILL
python scripts/inpaint_ControlNet_csv.py --csv_file "data/open_source_samples/dynafill/subset/full_to_target.csv" --image_dir "data/open_source_samples/dynafill/subset/" --outdir "data/samples/DYNAFILL/" --ckpt "logs/2023-02-08_custom_keyboard_training_CONTROLNET_DYNAFILL/checkpoints/epoch=000077.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_Inference.yaml" --device cuda:1 --prefix "DYNAFILL_CN_1"

python scripts/inpaint_ControlNet_csv.py --csv_file "data/open_source_samples/dynafill/subset/full_to_target.csv" --image_dir "data/open_source_samples/dynafill/subset/" --outdir "data/samples/DYNAFILL_CORRECT/" --ckpt "logs/2023-02-08_custom_keyboard_training_CONTROLNET_DYNAFILL_CORRECT/checkpoints/epoch=000077.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_Inference.yaml" --device cuda:1 --prefix "DYNAFILL_CN"


python scripts/inpaint_ControlNet_csv.py --csv_file "data/open_source_samples/dynafill/subset/full_to_target.csv" --image_dir "data/open_source_samples/dynafill/subset/" --outdir "data/samples/DYNAFILL_CORRECT/" --ckpt "logs/2023-02-08_custom_keyboard_training_CONTROLNET_DYNAFILL_CORRECT_FULL/checkpoints/xxxxx" --yaml_profile "configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_Inference.yaml" --device cuda:1 --prefix "DYNAFILL_CN"


python scripts/inpaint_ControlNet_csv.py --csv_file "data/modules/DYNAFILL/full_to_target.csv" --image_dir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/" --outdir "data/samples/DYNAFILL_REAL_VALIDATION/" --ckpt "logs/2023-02-08_custom_keyboard_training_CONTROLNET_DYNAFILL_CORRECT_FULL/checkpoints/epoch=000000.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_Inference.yaml" --device cuda:1 --prefix "DYNAFILL_CN"


python scripts/inpaint_ControlNet.py --indir "data/samples/Car_custom" --outdir "data/samples/DYNAFILL_CORRECT_CUSTOM/" --ckpt "logs/2023-02-08_custom_keyboard_training_CONTROLNET_DYNAFILL_CORRECT/checkpoints/epoch=000077.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_Inference.yaml" --device cuda:1 --prefix "DYNAFILL_CN"


## DYNAFILL NO CONTROL: RGB AND MASK VS ONLY MASK

python scripts/inpaint_RGB_CHANNELS_csv.py --csv_file "data/modules/DYNAFILL/full_to_target.csv" --image_dir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/" --outdir "data/samples/DYNAFILL_RGB_REAL_VALIDATION/" --ckpt "logs/2023-02-08_SEG_DYNAFILL_FULLARRANGE/checkpoints/epoch=000000.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_DYNAFILL_SEG_INPAINT.yaml" --device cuda:1 --prefix "DYNAFILL_CN"



python scripts/inpaint_RGB_CHANNELS_csv.py --csv_file "data/modules/DYNAFILL/full_to_target.csv" --image_dir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/" --outdir "data/samples/DYNAFILL_RGB_REAL_VALIDATION_SEG_LEARNED/" --ckpt "logs/2023-02-08_SEG_DYNAFILL_FULLARRANGE_SD_NEW_KEYS_SEG_LOW_RES/checkpoints/epoch=000000.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_DYNAFILL_SEG_INPAINTARRANGE_SD_WEIGHTS_256.yaml" --device cuda:1 --prefix "DYNAFILL_CN"


python scripts/inpaint_runaway_correct_from_csv.py --csv_file "data/modules/DYNAFILL/full_to_target.csv" --image_dir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/Datasets/DynaFill/" --outdir "data/samples/DYNAFILL_RGB_REAL_VALIDATION_CLASSIC/" --ckpt "logs/2023-02-08_CLASSICAL_DYNAFILL_CORRECT_FULL_LOW_RES/checkpoints/epoch=000002.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_DYNAFILL_SD_CLASSICAL_256.yaml" --device cuda:1 --prefix "DYNAFILL_CN"




python scripts/inpaint_runaway_correct_folder.py --indir "data/samples/Car_custom2" --outdir "data/samples/DYNAFILL_CORRECT_CUSTOM/" --ckpt "logs/2023-02-08_CLASSICAL_DYNAFILL_CORRECT_FULL_LOW_RES/checkpoints/epoch=000002.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_DYNAFILL_SD_CLASSICAL_256.yaml" --device cuda:1 --prefix "DYNAFILL_CN" --resize 256






## INTERIORNET

python scripts/inpaint_runaway_correct.py --indir "data/samples/interiornet_test/" --outdir "data/samples/interiornet_output/SD/" --ckpt "models/ldm/inpainting_big/model_compvis.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:1 --prefix "interiornet_SD"


python scripts/inpaint_ControlNet.py --indir "data/samples/interiornet_test/" --outdir "data/samples/interiornet_output/CN/" --ckpt "logs/2023-02-08_custom_keyboard_training_CONTROLNET_LAMA/checkpoints/epoch=000002.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_interiornet_ControlNet_Inference.yaml" --device cuda:1 --prefix "interiornet_CN_1"


python scripts/inpaint_runaway_correct.py --indir "data/samples/interiornet_test/" --outdir "data/samples/interiornet_output/LAMA/" --ckpt "logs/2023-02-08_custom_interionet_FULL_TRAINING_LAMA/checkpoints/epoch=000021.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation_lama"



python scripts/inpaint_runaway_correct_from_csv.py --csv_file "data/open_source_samples/dataframe_interiornet_lama_validation.csv" --outdir "data/samples/interiornet_output/validation_lama/" --ckpt "logs/2023-02-08_custom_interionet_FULL_TRAINING_LAMA/checkpoints/epoch=000021.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation_lama"



python scripts/inpaint_runaway_correct.py --indir "data/samples/interiornet_test_artificial/" --outdir "data/samples/interiornet_output/artificial/" --ckpt "logs/2023-02-08_custom_interionet_FULL_TRAINING_LAMA/checkpoints/epoch=000021.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_artificial"


### Validation set

python scripts/inpaint_runaway_correct_from_csv.py --csv_file "data/open_source_samples/dataframe_filtered_test.csv" --outdir "data/samples/interiornet_output/validation/" --ckpt "models/ldm/inpainting_big/model_compvis.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation_compvis"

python scripts/inpaint_runaway_correct_from_csv.py --csv_file "data/open_source_samples/dataframe_filtered_test.csv" --outdir "data/samples/interiornet_output/validation/" --ckpt "logs/2023-02-08_custom_interionet_FULL_TRAINING_fixed_LR/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation"

python scripts/inpaint_runaway_correct_from_csv.py --csv_file "data/open_source_samples/dataframe_filtered_test.csv" --outdir "data/samples/interiornet_output/validation_lr/" --ckpt "logs/2023-02-08_custom_interionet_FULL_TRAINING_test_scheduler/checkpoints/epoch=000115.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation_lr"



### AVERAGE Validation set

python scripts/inpaint_runaway_average.py --csv_file "data/open_source_samples/dataframe_filtered_test.csv" --outdir "data/samples/interiornet_output/validation_average/" --ckpt "models/ldm/inpainting_big/model_compvis.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation_compvis"


### INTERPOLATED Validation set

python scripts/inpaint_runaway_average_interpolation.py --csv_file "data/open_source_samples/dataframe_filtered_test.csv" --outdir "data/samples/interiornet_output/validation_interpolation/" --ckpt "models/ldm/inpainting_big/model_compvis.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "interiornet_validation_compvis"


## KEYBOARD NOT EMA VS EMA

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_FULL_TRAINING/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key2"

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_FULL_TRAINING2/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key2"

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_different_samplerSAMESEEDNOTEMA/checkpoints/epoch=000269.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key"

python scripts/inpaint_runaway_correct.py --indir "data/samples/custom_inpainting/mouse_keyboard/" --outdir "data/samples/custom_inpainting/output/" --ckpt "logs/2023-02-08_custom_keyboard_training_FULL_SAMESEEDNOTEMA/checkpoints/last.ckpt" --yaml_profile "configs/latent-diffusion/inpainting_runaway_inference.yaml" --device cuda:0 --prefix "trained_key_FULL"



## DEBUG

python scripts/INPAINTDEBUG.py --indir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/custom_inpainting/" --outdir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/output_images_debug/"

python scripts/inpaint_runaway_correctBIG.py --indir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/custom_inpainting/" --outdir "/data01/lorenzo.stacchio/TU GRAZ/Stable_Diffusion_Inpaiting/stable-diffusion_custom_inpaint/data/INPAINTING/output_images_debug/"




