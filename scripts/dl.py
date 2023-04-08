from neural_waveshaping_synthesis.data.general import GeneralDataModule

dm = GeneralDataModule(
    "/root/data/nws/timbre-16k-f0_di_75_batch",
    load_to_memory=False,
    num_workers=1,
    shuffle=False,
    batch_size=3
)
dm.setup(stage="all")
dl = dm.train_dataloader()
print(len(dl))

it = iter(dl)
batch = next(it)
print(batch['name'])