import wandb
import yaml

with open("sweep_fc.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, entity="marvinpeng", project="checkthat")
print(sweep_id)