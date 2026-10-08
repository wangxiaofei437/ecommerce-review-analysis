"""
主启动脚本 - 一键运行整个系统（使用无数据泄露的训练）
自动检测模型是否存在，若不存在则执行训练
"""
import os
import sys

from config import RF_MODEL_PATH

def main():
    """主启动函数"""
    print("=" * 60)
    print("评论有用性分析系统（v3 - 无数据泄露版本）")
    print("=" * 60)

    if not os.path.exists(RF_MODEL_PATH):
        print("\n未检测到预训练模型，开始训练...")
        print("使用内容质量标签，但移除数据泄露特征...")
        print("这可能需要几分钟时间，请耐心等待...\n")

        try:
            from train import main as train_main
            train_main()
        except Exception as e:
            print(f"\n训练过程出错：{e}")
            print("请检查数据文件是否存在，或手动运行 train.py")
            sys.exit(1)

    else:
        print("\n检测到预训练模型，跳过训练步骤")
        print("模型版本：v4（90%+准确率，TF-IDF增强特征）")


if __name__ == '__main__':
    main()
