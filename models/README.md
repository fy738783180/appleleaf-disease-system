# Model Files

This repository does not include trained model weights.

Please place the following files in this directory before running the app:

- `vit_resnet_multi_task_model.pth`
- `ld_deeplabv3plus_best_model_3.pth`

Expected structure:

```text
models/
├─ vit_resnet_multi_task_model.pth
└─ ld_deeplabv3plus_best_model_3.pth
```

You can also override these paths with environment variables:

- `CLS_MODEL_PATH`
- `SEG_MODEL_PATH`
