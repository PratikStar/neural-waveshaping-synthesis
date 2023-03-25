
`
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

# from wisteria
rsync -av w:/work/gk77/k77021/data/A_sharp_3 "/Users/pratik/Downloads"

rsync -av w:/work/gk77/k77021/nws/monophonic-4secchunks-di_f0-44032hz/checkpoints/last.ckpt "/Users/pratik/Downloads" 

# GCP
gcloud compute ssh --ssh-flag="-ServerAliveInterval=30" --zone us-west1-b instance-3
gcloud compute scp ~/Downloads/monophonic-4secchunks-di_f0-20230318T164941Z-001.zip instance-3:/home/pratik --zone us-west1-b


python scripts/train.py \
--gin-file gin/train/train_newt.gin \
--dataset-path /root/data/nws/monophonic-4secchunks-di_f0 \
--checkpoint-path /root/data/nws/monophonic-4secchunks-di_f0 \
--load-data-to-memory

