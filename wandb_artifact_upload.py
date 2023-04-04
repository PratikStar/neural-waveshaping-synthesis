import wandb
import os

path = "/root/nws"

wandb.init(project="nws", name="artifact_update")
for f in os.listdir():
    if not f.startswith("timbre"):
        continue
    artifact = wandb.Artifact(f, type="model")
    artifact.add_dir(f)
    wandb.log_artifact()

