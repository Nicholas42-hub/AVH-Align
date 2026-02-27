#!/bin/bash
# Quick progress check for FakeAVCeleb full dataset processing

echo "============================================================"
echo "  FakeAVCeleb Full Dataset Processing Progress"
echo "  $(date)"
echo "============================================================"
echo ""

cd /data/projects/punim2637/nnliang/AVH-Align

TOTAL=21544
PREP=$(find data/favc_preprocessed -name "*.mp4" 2>/dev/null | wc -l)
FEAT=$(find data/favc_features -name "*.npz" 2>/dev/null | wc -l)

PREP_PCT=$(awk "BEGIN {printf \"%.1f\", $PREP/$TOTAL*100}")
FEAT_PCT=$(awk "BEGIN {printf \"%.1f\", $FEAT/$TOTAL*100}")

echo "📊 总视频数: $TOTAL"
echo ""
echo "1️⃣  预处理进度:"
echo "   已完成: $PREP / $TOTAL ($PREP_PCT%)"
echo "   待处理: $((TOTAL - PREP))"
echo ""
echo "2️⃣  特征提取进度:"
echo "   已完成: $FEAT / $TOTAL ($FEAT_PCT%)"
echo "   待处理: $((TOTAL - FEAT))"
echo ""

# Check running jobs
echo "🔄 当前任务状态:"
squeue -u nnliang --format="  %.10i %.12P %.20j %.8T %.10M %.6D %R"

echo ""
echo "============================================================"
echo ""
echo "💡 查看日志:"
echo "   tail -f logs/favc_full_preprocess_*.out"
echo "   tail -f logs/favc_full_features_*.out"
echo ""
