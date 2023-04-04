import wandb
import os

path = "/root/nws"

wandb.init(project="nws", name="artifact_upload")
for f in os.listdir(path):
    if not f.startswith("timbre"):
        continue
    print(f"For: {f}")
    artifact = wandb.Artifact(f, type="model")
    artifact.add_dir(path + "/" + f)
    wandb.log_artifact(artifact)


