# 处理状态总结 (Processing Status Summary)

## 当前进度 (Current Progress)

### 任务状态 (Job Status)
✅ **AVDeepfake1M 预处理** - 正在运行 (Job ID: 22009014)
- 分区: cascade
- 运行时间: ~40分钟
- 状态: 正在处理57,340个视频

✅ **FakeAVCeleb 特征提取** - 正在运行 (Job ID: 22010184)  
- 分区: gpu-a100 (A100 GPU)
- 运行时间: ~2分钟
- 状态: 正在从4,089个预处理文件提取特征

⏳ **FakeAVCeleb 预处理** - 需要继续
- 已完成: 4,089 / 21,544 (19.0%)
- 需要: 继续处理剩余17,455个视频

## 修复的问题 (Fixed Issues)

### 🐛 Bug修复
修复了 `deepfake_preprocess.py` 中的数据集名称判断错误：
- **问题**: 脚本传递 `--dataset AVDeepfake1M`，但代码只接受 `AV1M`
- **修复**: 修改判断条件为 `elif args.dataset in ['AV1M', 'AVDeepfake1M']:`
- **文件**: `/data/projects/punim2637/nnliang/AVH-Align/av_hubert/avhubert/deepfake_preprocess.py`
- **结果**: AVDeepfake1M预处理现在正常工作

## 监控命令 (Monitoring Commands)

### 1. 查看任务队列
```bash
squeue -u nnliang
```

### 2. 实时监控日志
```bash
# AVDeepfake1M 预处理
tail -f logs/avd1m_scratch_preprocess_22009014.out

# FakeAVCeleb 特征提取  
tail -f logs/favc_scratch_features_22010184.out
```

### 3. 使用监控脚本
```bash
cd /data/projects/punim2637/nnliang/AVH-Align
./monitor_processing.sh
```

### 4. 快速检查数量（可能较慢）
```bash
# AVDeepfake1M 预处理进度
ls -R /data/scratch/projects/punim2637/nnliang/avd1m_preprocessed/val/*/*/*/ | grep "_roi.mp4" | wc -l

# FakeAVCeleb 特征数量
find /data/projects/punim2637/nnliang/AVH-Align/data/favc_features -name "*.npz" | wc -l
```

## 下一步操作 (Next Steps)

### 当前任务完成后:

1. **提交 AVDeepfake1M 特征提取**
   ```bash
   cd /data/projects/punim2637/nnliang/AVH-Align
   sbatch avd1m_scratch_features.slurm
   ```

2. **继续 FakeAVCeleb 预处理**（当前预处理任务完成后）
   ```bash
   cd /data/projects/punim2637/nnliang/AVH-Align
   sbatch favc_scratch_preprocess.slurm
   ```

3. **自动提交特征提取**（可选）
   ```bash
   # 当AVD1M预处理达到50000个文件时自动提交特征提取
   nohup ./auto_submit_features.sh avd1m 50000 > auto_avd1m.log 2>&1 &
   
   # 当FakeAVCeleb预处理达到20000个文件时自动提交特征提取
   nohup ./auto_submit_features.sh favc 20000 > auto_favc.log 2>&1 &
   ```

## 预计时间 (Estimated Time)

基于当前进度（AVD1M在40分钟处理了~2000个视频）:
- **AVDeepfake1M 预处理**: ~18-24小时（57,340个视频）
- **AVDeepfake1M 特征提取**: ~24-36小时（取决于GPU可用性）
- **FakeAVCeleb 预处理**: ~4-6小时（剩余17,455个视频）
- **FakeAVCeleb 特征提取**: 正在进行中（~2-4小时从4,089个文件）

## 存储信息 (Storage Info)

- **Scratch**: 39GB / 1TB (4%)
- **Projects**: 监控特征文件大小（.npz文件将复制到projects）
- **临时文件**: 在scratch中（预处理的视频和音频）
- **最终文件**: NPZ特征文件在projects中

## 文件位置 (File Locations)

### 输入数据 (Scratch)
- AVDeepfake1M: `/data/scratch/projects/punim2637/nnliang/Datasets/AVDeepfake1M/val`
- FakeAVCeleb: `/data/scratch/projects/punim2637/nnliang/Datasets/FakeAVCeleb_v1.2`

### 预处理输出 (Scratch)
- AVDeepfake1M: `/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed/val`
- FakeAVCeleb: `/data/scratch/projects/punim2637/nnliang/favc_preprocessed`

### 特征文件 (Projects)
- AVDeepfake1M: `/data/projects/punim2637/nnliang/AVH-Align/data/avh_features`
- FakeAVCeleb: `/data/projects/punim2637/nnliang/AVH-Align/data/favc_features`

## 注意事项 (Notes)

1. **GPU任务优先级**: FakeAVCeleb特征提取在GPU上运行，可能会被更高优先级任务抢占
2. **错误处理**: 一些视频文件可能无法加载（损坏或格式问题），这是正常的
3. **Scratch清理**: Scratch文件会在30-90天后自动删除，确保特征文件及时复制到projects
4. **分区建议**: 系统建议使用sapphire分区替代cascade（更新的资源）

## 问题排查 (Troubleshooting)

如果任务失败或停止:
```bash
# 查看错误日志
tail -100 logs/avd1m_scratch_preprocess_*.err
tail -100 logs/favc_scratch_features_*.err

# 重新提交任务
sbatch avd1m_scratch_preprocess.slurm
sbatch favc_scratch_features.slurm
```

---
**最后更新**: 2026-02-26 20:30
**状态**: ✅ 修复完成，任务运行中
