#!/bin/bash

echo "========================================"
echo "项目Git设置脚本"
echo "========================================"
echo

echo "1. 检查Git状态..."
git status
echo

echo "2. 添加.gitignore文件..."
git add .gitignore
echo

echo "3. 添加README.md和说明文件..."
git add README.md 数据下载说明.md
echo

echo "4. 添加代码文件（排除大文件）..."
git add *.ipynb
git add 代码/
echo

echo "5. 提交更改..."
git commit -m "Add project files with .gitignore to handle large data files"
echo

echo "6. 推送到远程仓库..."
git push origin main
echo

echo "========================================"
echo "完成！"
echo "========================================"
echo
echo "注意：大型数据文件已被.gitignore排除"
echo "请查看 数据下载说明.md 了解如何获取数据文件"
echo
