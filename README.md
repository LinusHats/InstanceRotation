## weights and bisaes
### requirements
```
pip install wandb
```

```
wandb login
```


### in script
```
import wandb

wandb.init(
    project="ProjectName",

    config={
        "learning_rate": 0.0.00005,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 50,
    }
)


# training Loop
for epoch in epochs:


    wandb.log({"acc": acc, "loss": loss})

# in notebooks
wandb.finish()

```