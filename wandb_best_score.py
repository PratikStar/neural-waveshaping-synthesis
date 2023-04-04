import wandb
from tqdm import tqdm


WANDB_PROJECT = "nws"
TEST_CHANGES = True # Run once before changing

api = wandb.Api()
runs = [run for run in api.runs(WANDB_PROJECT)]
print(f"{len(runs)} runs found")

# Specify column name and comparison function (ideally min or max)
targs = (("val/loss", min), ("train/loss", min))
keys = [targ[0] for targ in targs]

changes = 0
change_made = False
for run in tqdm(runs):
    print(run)
    # continue
    try:
        previous = [run.summary[key] for key in keys]
    except KeyError: # Catches incomplete runs
        tqdm.write(f"{run.name} lacks the proper summary keys, skipping...")
        continue
    bests = [None] * len(targs)
    data = [row for row in run.scan_history(keys = keys)]
    print(data)
    print(previous)
    continue
    for i, (key, func) in enumerate(targs):
        bests[i] = func(row[key] for row in data)
        # Display incorrect "bests"
        if TEST_CHANGES:
            if (  (func == max and bests[i] < previous[i]) or
                    (func == min and bests[i] > previous[i])):
                tqdm.write(f"{run.name}\n{key} {previous[i]} -> {bests[i]}")
                changes += 1
        # Overwrite summary values when a better score was found
        else:
            if (  (func == max and bests[i] > previous[i]) or
                    (func == min and bests[i] < previous[i])):
                run.summary[key] = bests[i]
                changes += 1
                change_made = True
    # So update() is called per-run instead of per-change
    if change_made:
        run.summary.update()
        change_made = False

if TEST_CHANGES:
    print(f"{changes} bad changes caught")
else:
    print(f"{changes} changes applied")