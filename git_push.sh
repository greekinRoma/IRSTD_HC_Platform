#!/bin/bash
# ============================================================================
#  GitHub 一键上传脚本
#  Usage:  bash git_push.sh  "commit message"
# ============================================================================

set -euo pipefail

echo ">>> 当前 Git 状态 <<<"
git status --short

echo ""
echo ">>> 添加所有修改 <<<"
git add -A

# 排除不需要上传的文件夹
echo ">>> 排除 __pycache__ 和 .vscode <<<"
git reset -- __pycache__/
git reset -- .vscode/
git reset -- 'model/**/__pycache__/'

echo ""
echo ">>> 即将提交的文件 <<<"
git status --short

# 使用传入的 commit message，或使用默认值
MSG="${1:-update $(date '+%Y-%m-%d %H:%M')}"
echo ""
echo ">>> 提交: $MSG <<<"
git commit -m "$MSG" -m "Co-Authored-By: Claude <noreply@anthropic.com>"

echo ""
echo ">>> 拉取远程最新代码 <<<"
git pull origin master --rebase || { echo "!! 有冲突，请手动解决后重新运行"; exit 1; }

echo ""
echo ">>> 推送到 origin/master <<<"
git push origin master

echo ""
echo "✅ 上传完成！"
