watch -n5 ./watch-acp.sh

# 1. Rsync repo
cd /Users/pratik/repos/neural-waveshaping-synthesis
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/neural-waveshaping-synthesis\" w:/work/gk77/k77021/repos"

nohup watch -d -n5 rsync -av --exclude-from=".rsyncignore_upload" "/Users/pratik/repos/neural-waveshaping-synthesis" w:/work/gk77/k77021/repos 0<&- &> /dev/null &

# 2. Rsync data
cd /Users/pratik/data/A_sharp_3
rsync -avz "/Users/pratik/data/A_sharp_3" w:/work/gk77/k77021/data
rsync -avz "/Users/pratik/data/single_note_distorted" w:/work/gk77/k77021/data
rsync -avz "/Users/pratik/data/di_1_one_clip" w:/work/gk77/k77021/data
rsync -avz "/Users/pratik/Downloads/monophonic-4secchunks-di_f0" w:/work/gk77/k77021/nws

rsync -avz "/Users/pratik/data/timbre_A4" w:/work/gk77/k77021/data

# Rsync checkpoints and data
rsync -avz w:/work/gk77/k77021/nws "/Users/pratik"

# from wisteria
rsync -av w:/work/gk77/k77021/data/A_sharp_3 "/Users/pratik/Downloads"

rsync -av w:/work/gk77/k77021/nws/monophonic-4secchunks-di_f0-44032hz/checkpoints "/Users/pratik/Downloads/nws/monophonic-4secchunks-di_f0-44032hz" 




# GCP
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-west1-b instance-3
## pratikrsutar
gcloud --project ddsp2-374016 compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east4-c instance-gpu2
## sutarprateeeeek. just use cloudshell
gcloud --project nws1-382311 compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east4-c instance-gpu


## scp data
gcloud compute scp /home/olympus_shared/monophonic-4secchunks instance-1:/home/olympus_shared/data/timbre --zone us-east4-c --recurse --compress
rsync -r /home/olympus_shared/data/timbre /root/data/

gcloud compute scp instance-gpu2:/home/pratik/data/nws/timbre_A4-16k-f0_di /Users/pratik/data/nws/timbre_A4-16k-f0_di --zone us-east4-c --recurse --compress

## checkpoint transfer. gcp to local
cp /root/nws/timbre_A4-16k-f0_hardcoded-static_dynamic_z16/checkpoints/last-v1.ckpt /home/pratik/nws/timbre_A4-16k-f0_hardcoded-static_dynamic_z16/checkpoints/last-v1.ckpt
gcloud compute scp instance-gpu2:/home/pratik/nws/timbre_A4-16k-f0_hardcoded-static_dynamic_z16/checkpoints/last-v1.ckpt /Users/pratik/nws/timbre_A4-16k-f0_hardcoded-static_dynamic_z16/checkpoints/last-v1.ckpt --zone us-east4-c --recurse --compress



```shell
sudo su
apt-get install git wget
ulimit -Sn 10000
# install miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3<tab>
source /root/.bashrc
pip install --upgrade pip


#data
gcloud compute scp /Users/pratik/data/timbre_A4 instance-gpu2:/home/pratik/data/timbre_A4 --zone us-east4-c --recurse --compress
gcloud compute scp /Users/pratik/data/half_notes_chromatic_data/wav/A4_half.wav instance-gpu2:/home/pratik/data/timbre_A4 --zone us-east4-c --recurse --compress
ssh-keygen
<add key to github>
git clone git@github.com:PratikStar/neural-waveshaping-synthesis.git
cd neural-waveshaping-synthesis
pip install .
#pip install -r requirements.txt # DONT RUN THIS ON GCP!!


<mv to root>
conda create --name nws python=3.9.16
conda activate nws
pip install pytorch-lightning wandb torchcrepe auraloss librosa black gin-config

wandb login
```

python scripts/create_dataset.py \
--gin-file gin/data/urmp_4second_crepe.gin \
--data-directory /root/data/timbre/monophonic-4secchunks \
--output-directory /root/data/nws/timbre-16k-f0_di_75 \
--device cuda:0

wandb agent auditory-grounding/nws/63ag5eev  >> ~/logs/sweep_63ag5eev_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75 \
--checkpoint-path /root/nws/timbre-16k-f0_di_75-static_dynamic_z_6_48 \
--checkpoint-file last-v1.ckpt \
--restore-checkpoint \
--load-data-to-memory  >> ~/logs/timbre-16k-f0_di_75-static_dynamic_z_6_48_$(date +%Y%m%d_%H%M%S).log 2>&1 &


python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75 \
--checkpoint-path /root/nws/timbre-16k-f0_di_75-concat_static_z_2_16 \
--checkpoint-file last.ckpt \
--restore-checkpoint \
--load-data-to-memory  >> ~/logs/timbre-16k-f0_di_75-concat_static_z_2_16_$(date +%Y%m%d_%H%M%S).log 2>&1 &


python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75 \
--checkpoint-path /root/nws/timbre-16k-f0_di_75-embed_static_z_2_16 \
--load-data-to-memory >> ~/logs/timbre-16k-f0_di_75-embed_static_z_2_16_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75 \
--checkpoint-path /root/nws/timbre-16k-f0_di_75-embed_static_z_4_32 \
--load-data-to-memory >> ~/logs/timbre-16k-f0_di_75-embed_static_z_4_32_$(date +%Y%m%d_%H%M%S).log 2>&1 &


python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75_batch \
--checkpoint-path  /root/nws/timbre-16k-f0_di_75-embed_static_roll_z_6_48 \
--load-data-to-memory  >> ~/logs/timbre-16k-f0_di_75-embed_static_roll_z_6_48_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75_batch \
--checkpoint-path  /root/nws/timbre-16k-f0_di_75-embed_static_roll_z_2_16 \
--load-data-to-memory  >> ~/logs/timbre-16k-f0_di_75-embed_static_roll_z_2_16_$(date +%Y%m%d_%H%M%S).log 2>&1 &


mv  train/audio/* all/audio/
mv  train/control/* all/control/
mv  val/audio/* all/audio/
mv  val/control/* all/control/

ls -la all/audio/ | wc -l
ls -la all/control/ | wc -l