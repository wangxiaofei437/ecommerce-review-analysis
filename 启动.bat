@echo off
chcp 65001 >nul
echo ========================================
echo 评论有用性分析系统 - 快速启动
echo ========================================
echo.

echo [1/3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)
python --version

echo.
echo [2/3] 检查依赖库...
python -c "import pandas, sklearn, xgboost, lightgbm" >nul 2>&1
if errorlevel 1 (
    echo 检测到缺少依赖库，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖库安装失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
) else (
    echo 依赖库检查通过
)

echo.
echo [3/3] 启动系统...
python main.py

if errorlevel 1 (
    echo.
    echo 启动失败，请查看错误信息
    pause
)
