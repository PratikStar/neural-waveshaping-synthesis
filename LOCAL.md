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
--checkpoint-path /root/nws/timbre-16k-f0_di_75-concat_static_z_2_32 \
--checkpoint-file last-v1.ckpt \
--restore-checkpoint \
--load-data-to-memory  >> ~/logs/timbre-16k-f0_di_75-concat_static_z_2_32_$(date +%Y%m%d_%H%M%S).log 2>&1 &


python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75 \
--checkpoint-path /root/nws/timbre-16k-f0_di_75-concat_static_z_6_48 \
--load-data-to-memory  >> ~/logs/timbre-16k-f0_di_75-concat_static_z_6_48_$(date +%Y%m%d_%H%M%S).log 2>&1 &


python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre-16k-f0_di_75 \
--checkpoint-path /root/tmp \
--load-data-to-memory



tensor([-0.6475, -0.8990], device='cuda:0', requires_grad=True), '16B': Parameter containing:
tensor([ 0.1055, -0.0342], device='cuda:0', requires_grad=True), '16C': Parameter containing:
tensor([ 0.1178, -0.8899], device='cuda:0', requires_grad=True), '16D': Parameter containing:
tensor([ 1.4827, -0.4723], device='cuda:0', requires_grad=True), '17A': Parameter containing:
tensor([2.9757, 0.3506], device='cuda:0', requires_grad=True), '17B': Parameter containing:
tensor([ 0.2337, -0.0874], device='cuda:0', requires_grad=True), '17C': Parameter containing:
tensor([0.0610, 0.6881], device='cuda:0', requires_grad=True), '17D': Parameter containing:
tensor([0.0809, 1.0047], device='cuda:0', requires_grad=True), '18A': Parameter containing:
tensor([ 0.9832, -0.4568], device='cuda:0', requires_grad=True), '18B': Parameter containing:
tensor([0.9831, 0.2739], device='cuda:0', requires_grad=True), '18C': Parameter containing:
tensor([-1.7805,  1.4514], device='cuda:0', requires_grad=True), '18D': Parameter containing:
tensor([-0.5405,  0.0884], device='cuda:0', requires_grad=True), '19A': Parameter containing:
tensor([ 0.6684, -1.1299], device='cuda:0', requires_grad=True), '19B': Parameter containing:
tensor([0.7544, 0.4326], device='cuda:0', requires_grad=True), '19C': Parameter containing:
tensor([-1.6034, -0.5674], device='cuda:0', requires_grad=True), '19D': Parameter containing:
tensor([ 0.7837, -1.0200], device='cuda:0', requires_grad=True), '20A': Parameter containing:
tensor([ 1.3652, -1.0433], device='cuda:0', requires_grad=True), '20B': Parameter containing:
tensor([-0.9113, -1.0770], device='cuda:0', requires_grad=True), '20C': Parameter containing:
tensor([0.6269, 1.2017], device='cuda:0', requires_grad=True), '20D': Parameter containing:
tensor([-0.5656,  0.7948], device='cuda:0', requires_grad=True)}



tensor([-0.6475, -0.8990], device='cuda:0', requires_grad=True), '16B': Parameter containing:
tensor([ 0.1055, -0.0342], device='cuda:0', requires_grad=True), '16C': Parameter containing:
tensor([ 0.1178, -0.8899], device='cuda:0', requires_grad=True), '16D': Parameter containing:
tensor([ 1.4827, -0.4723], device='cuda:0', requires_grad=True), '17A': Parameter containing:
tensor([2.9757, 0.3506], device='cuda:0', requires_grad=True), '17B': Parameter containing:
tensor([ 0.2337, -0.0874], device='cuda:0', requires_grad=True), '17C': Parameter containing:
tensor([0.0610, 0.6881], device='cuda:0', requires_grad=True), '17D': Parameter containing:
tensor([0.0809, 1.0047], device='cuda:0', requires_grad=True), '18A': Parameter containing:
tensor([ 0.9832, -0.4568], device='cuda:0', requires_grad=True), '18B': Parameter containing:
tensor([0.9831, 0.2739], device='cuda:0', requires_grad=True), '18C': Parameter containing:
tensor([-1.7805,  1.4514], device='cuda:0', requires_grad=True), '18D': Parameter containing:
tensor([-0.5405,  0.0884], device='cuda:0', requires_grad=True), '19A': Parameter containing:
tensor([ 0.6684, -1.1299], device='cuda:0', requires_grad=True), '19B': Parameter containing:
tensor([0.7544, 0.4326], device='cuda:0', requires_grad=True), '19C': Parameter containing:
tensor([-1.6034, -0.5674], device='cuda:0', requires_grad=True), '19D': Parameter containing:
tensor([ 0.7837, -1.0200], device='cuda:0', requires_grad=True), '20A': Parameter containing:
tensor([ 1.3652, -1.0433], device='cuda:0', requires_grad=True), '20B': Parameter containing:
tensor([-0.9113, -1.0770], device='cuda:0', requires_grad=True), '20C': Parameter containing:
tensor([0.6269, 1.2017], device='cuda:0', requires_grad=True), '20D': Parameter containing:
tensor([-0.5656,  0.7948], device='cuda:0', requires_grad=True)}