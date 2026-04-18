# 模型文件说明

运行项目需要以下两个模型文件:

- `vit_resnet_multi_task_model.pth`
- `ld_deeplabv3plus_best_model_3.pth`

你可以直接从 GitHub Releases 下载:

- 分类模型: [vit_resnet_multi_task_model.pth](https://github.com/fy738783180/appleleaf-disease-system/releases/latest/download/vit_resnet_multi_task_model.pth)
- 分割模型: [ld_deeplabv3plus_best_model_3.pth](https://github.com/fy738783180/appleleaf-disease-system/releases/latest/download/ld_deeplabv3plus_best_model_3.pth)

下载后放到当前目录，结构如下:

```text
models/
├─ vit_resnet_multi_task_model.pth
└─ ld_deeplabv3plus_best_model_3.pth
```

也支持通过环境变量指定:

- `CLS_MODEL_PATH`
- `SEG_MODEL_PATH`
