from ldm.util import instantiate_from_config
model = instantiate_from_config(config.model)
print(model.state_dict().keys())