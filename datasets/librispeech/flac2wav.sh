#!/usr/bin/env bash

echo ============================================================================
echo "                              Librispeech                                 "
echo ============================================================================

DATASET_ROOT_PATH='~/corpus_en/LibriSpeech'

echo ============================================================================
echo "                        Convert from flac to wav                          "
echo ============================================================================

Nproc=15    # 可同时运行的最大作业数
echo $Nproc
function PushQue {    # 将PID压入队列
  Que="$Que $1"
  Nrun=$(($Nrun+1))
}

function GenQue {     # 更新队列
  OldQue=$Que
  Que=""; Nrun=0
  for PID in $OldQue; do
    if [[ -d /proc/$PID ]]; then
      PushQue $PID
    fi
  done
}

function ChkQue {     # 检查队列
  OldQue=$Que
  for PID in $OldQue; do
    if [[ ! -d /proc/$PID ]] ; then
      GenQue; break
    fi
  done
}

function CMD (){        # 测试命令, 随机等待几秒钟
  dir_path=$(dirname $1)
  file_name=$(basename $1)
  base=${file_name%.*}
  ext=${file_name##*.}
  wav_path=$dir_path"/"$base".wav"
  if [ $ext = "flac" ]; then
    sox $flac_path -t wav $wav_path
    echo "Converting from"$flac_path" to "$wav_path
    rm -f $flac_path
  else
    echo "Already converted: "$wav_path
  fi
}


flac_paths=$(find $DATASET_ROOT_PATH -iname '*.flac')
for flac_path in $flac_paths ; do
    CMD $flac_path &
    PID=$!
    PushQue $PID
    while [[ $Nrun -ge $Nproc ]]; do
        ChkQue
    done
done
wait
echo "Done~"
# https://jerkwin.github.io/2013/12/14/Bash%E8%84%9A%E6%9C%AC%E5%AE%9E%E7%8E%B0%E6%89%B9%E9%87%8F%E4%BD%9C%E4%B8%9A%E5%B9%B6%E8%A1%8C%E5%8C%96/
#flac_paths=$(find $DATASET_ROOT_PATH -iname '*.flac')
#for flac_path in $flac_paths ; do
#  dir_path=$(dirname $flac_path)
#  file_name=$(basename $flac_path)
#  base=${file_name%.*}
#  ext=${file_name##*.}
#  wav_path=$dir_path"/"$base".wav"
#  if [ $ext = "flac" ]; then
#    echo "Converting from"$flac_path" to "$wav_path
#    sox $flac_path -t wav $wav_path
#    rm -f $flac_path
#  else
#    echo "Already converted: "$wav_path
#  fi