#!/bin/bash

# 1. 清除所有代理设置，避免网络干扰
unset all_proxy
unset http_proxy
unset ftp_proxy
unset rsync_proxy
unset https_proxy

# 2. 定义需要检查的目录
sudo apt update
# 3. 检查目录是否存在
all_exist=true
if [ ! -d "data/IRSTD-1K" ]; then
    echo "目录不存在: data/IRSTD-1K"
    all_exist=false
fi
if [ ! -d "data/NUDT-SIRST" ]; then
    echo "目录不存在: data/NUDT-SIRST"
    all_exist=false
fi
# 4. 根据检查结果输出信息
if [ "$all_exist" = true ]; then
    echo "Both directories exist. 所需目录已存在。"
else
    if [ -d "data.zip" ]; then
        echo "已经检测到数据data.zip"
        echo "开始解压data.zip"
        if ! command -v unzip &>/dev/null; then
            sudo apt install unzip 
        fi
        unzip data.zip
    else
        echo -e "\n=== 需要下载数据集 ==="
        echo "缺少上述数据集目录。"
        echo "请从以下百度网盘链接下载并解压到当前目录的 data/ 文件夹下："
        echo "链接: https://pan.baidu.com/s/19DOSJZTHC0KO-wKyGRSldQ?pwd=mxhe"
        echo "提取码: mxhe"
    fi
fi
python best_ckpt/test_best.py