#!/bin/bash
# 端口清理脚本 - 用于清理被占用的端口

PORT=${1:-5001}  # 默认清理5001端口，可通过参数指定其他端口

echo "🔍 检查端口 ${PORT} 的占用情况..."

# 查找占用端口的进程
PIDS=$(lsof -ti :${PORT} 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "✅ 端口 ${PORT} 未被占用"
    exit 0
fi

echo "📋 发现以下进程占用端口 ${PORT}:"
lsof -i :${PORT}

echo ""
read -p "⚠️  是否要杀死这些进程？(y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔨 正在终止进程..."
    echo "$PIDS" | xargs kill -9 2>/dev/null
    
    # 验证
    sleep 1
    if lsof -ti :${PORT} >/dev/null 2>&1; then
        echo "❌ 进程终止失败，请手动检查"
        exit 1
    else
        echo "✅ 端口 ${PORT} 已清理完成"
        exit 0
    fi
else
    echo "⏸️  操作已取消"
    exit 0
fi
