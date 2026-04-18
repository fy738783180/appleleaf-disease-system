# GitHub 发布信息

## 仓库名建议

`appleleaf-disease-system`

## 仓库标题

苹果叶病害识别与分割系统

## About / 简介

基于 Streamlit + PyTorch 的苹果叶病害识别与分割系统，支持分类、病斑分割、可视化展示与 Ubuntu 服务器部署。

## Topics

```text
streamlit
pytorch
computer-vision
image-segmentation
plant-disease
apple-leaf
deep-learning
ubuntu
```

## 首条发布文案

这是我整理并部署完成的苹果叶病害识别与分割系统。

项目基于 Streamlit 和 PyTorch，支持：

- 苹果叶病害分类
- 病斑分割与叠加显示
- Top-3 预测结果展示
- 像素占比统计
- Ubuntu 服务器部署

仓库中包含 Web 应用代码、依赖说明和 `systemd` 部署配置。

由于模型权重文件较大，本仓库未直接包含训练好的 `.pth` 文件，运行时可通过 `models/` 目录或环境变量指定模型路径。

欢迎交流和改进。
