
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

# from wisteria
rsync -av w:/work/gk77/k77021/data/A_sharp_3 "/Users/pratik/Downloads"


python scripts/create_dataset.py \
--gin-file gin/data/urmp_4second_crepe.gin \
--data-directory /work/gk77/k77021/data/A_sharp_3 \
--output-directory /work/gk77/k77021/nws \
--device cuda:0