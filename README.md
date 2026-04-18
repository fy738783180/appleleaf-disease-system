# 苹果叶病害识别与分割系统

这是一个基于 `Streamlit` 的苹果叶病害识别 Web 应用，支持叶片病害分类、病斑分割、结果可视化与像素占比统计，可直接部署到 Ubuntu 服务器上运行。

## 功能特点

- 9 类苹果叶病害分类
- 病斑区域分割
- Top-3 分类结果展示
- 彩色 Mask 与叠加可视化
- 病斑像素占比统计
- 支持 CPU 推理
- 支持 Ubuntu + `systemd` 部署

## 支持的分类类别

- Alternaria leaf spot
- Brown spot
- Frogeye leaf spot
- Grey spot
- Health
- Mosaic
- Powdery mildew
- Rust
- Scab

## 项目结构

```text
appleleaf-disease-system/
├─ app.py
├─ requirements.txt
├─ deployment/
│  ├─ appleleaf-streamlit.service
│  └─ DEPLOYMENT.md
└─ models/
   └─ README.md
```

## 运行环境

- Python 3.10 及以上
- Windows / Linux
- `torch`
- `torchvision`
- `streamlit`

## 本地启动

1. 创建虚拟环境

```bash
python -m venv venv
```

2. 激活虚拟环境

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

3. 安装依赖

```bash
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

4. 准备模型文件

将以下模型文件放到 `models/` 目录中：

- `vit_resnet_multi_task_model.pth`
- `ld_deeplabv3plus_best_model_3.pth`

也可以通过环境变量自定义模型路径：

```bash
export CLS_MODEL_PATH=/your/path/vit_resnet_multi_task_model.pth
export SEG_MODEL_PATH=/your/path/ld_deeplabv3plus_best_model_3.pth
```

5. 启动应用

```bash
streamlit run app.py
```

## 服务器部署

该项目已经完成 Ubuntu Server 24.04 的实际部署验证，适合用 `systemd` 方式长期运行。

部署步骤概览：

1. 上传项目到服务器
2. 创建 Python 虚拟环境
3. 安装依赖
4. 放置模型文件
5. 配置 `systemd` 服务
6. 放行应用端口
7. 使用公网 IP 访问

完整说明见：

[deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md)

## 模型说明

本仓库**不直接包含训练权重**。

原因：

- GitHub 对大文件有限制
- 模型文件体积较大
- 将仓库与模型分离更适合发布与协作

详见：

[models/README.md](models/README.md)

## 部署友好设计

为了适配服务器运行，项目做了这些处理：

- 支持相对路径读取模型
- 支持环境变量覆盖模型路径
- 启动时不强制下载预训练权重
- 可直接以独立端口运行，不影响已有 VPN / Nginx / HTTPS 服务

## 适合添加到 GitHub 的 Topics

`streamlit` `pytorch` `computer-vision` `image-segmentation` `plant-disease` `apple-leaf` `deep-learning`

## 作者

涛哥
