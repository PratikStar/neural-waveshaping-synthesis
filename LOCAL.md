
`
watch -n60 ./watch-acp.sh

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
gcloud compute scp ~/Downloads/monophonic-4secchunks-di_f0-20230318T164941Z-001.zip instance-3:/home/pratik --zone us-west1-b

gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-east4-c instance-gpu2

gcloud compute scp instance-gpu2:/home/pratik/nws/timbre_A4-16k-f0_hardcoded-static_dynamic_z/checkpoints/last-v2.ckpt /Users/pratik/nws/ --zone us-east4-c --recurse --compress



```shell
sudo su
apt-get install git wget
# install miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3<tab>
source /root/.bashrc
pip install --upgrade pip

gcloud compute scp instance-gpu2:/home/pratik/data/nws/timbre_A4-16k-f0_di /Users/pratik/data/nws/timbre_A4-16k-f0_di --zone us-east4-c --recurse --compress

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
```

python scripts/create_dataset.py \
--gin-file gin/data/urmp_4second_crepe.gin \
--data-directory /root/data/timbre_A4 \
--output-directory /root/nws/timbre_A4-16k-f0_di \
--device cuda:0

python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/timbre_A4-16k-f0_di \
--checkpoint-path /root/nws/timbre_A4-16k-f0_hardcoded-static_dynamic_z \
--restore-checkpoint \
--load-data-to-memory  >> ~/logs/timbre_A4-16k-f0_hardcoded-static_dynamic_z_$(date +%Y%m%d_%H%M%S).log 2>&1 &
