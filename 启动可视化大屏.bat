@echo off
chcp 65001 >nul
echo ========================================
echo 评论有用性预测 · 可视化大屏 启动
echo ========================================
echo.

echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)
python --version

echo.
echo [2/3] 检查依赖库（pandas / sklearn / lightgbm / flask）...
python -c "import pandas, sklearn, lightgbm, flask" >nul 2>&1
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
echo [3/3] 启动 Flask 大屏服务（http://127.0.0.1:5000）...
echo 提示：首次启动需要加载模型并在测试集上重算指标，可能需要 30 秒~1 分钟。
start "" http://127.0.0.1:5000
python web\app.py

if errorlevel 1 (
    echo.
    echo 启动失败，请查看上方错误信息。
    pause
)
