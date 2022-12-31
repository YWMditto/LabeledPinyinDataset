#!/bin/bash


devices=
batch_size=
accumulation_steps=

# 一定要带上 -o；
ArgsOption=`getopt -o d:b:a: --long devices:,batch_size,accumulation_steps: -n 'wrong input, should be like "--devices" "0"' -- "$@"`

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$ArgsOption"

while true; do
  case "$1" in
    -d|--devices) devices=$2; shift 2;;
    -b|--batch_size) batch_size=$2; shift 2;;
    -a|--accumulation_steps) accumulation_steps=$2; shift 2;;
    --) shift; break ;;
    *) echo "$1, wrong input"; exit 1;;
  esac
done

if [ -z $devices ]; then
  echo "None devices, use '0' by default."
  devices='0'
fi

if [ -z $batch_size ]; then
  echo "None batch_size, use '128' by default."
  batch_size=128
fi

if [ -z $accumulation_steps ]; then
  echo "None accumulation_steps, use '1' by default."
  accumulation_steps=1
fi


if [ $# -ne 1 ]; then
  echo "Usage: $0 [options] <script-name>"
  echo "e.g. bash train_script.sh debug_run"
  exit 1;
fi

script_name=$1

if [ "$script_name" == "debug_run" ]; then
  python train.py \
    --run_name "debug_run" \
    --info "使用少量数据来测试环境是否正常，模型是否能够正常运行；" \
    --use_wandb False \
    --n_epochs 2 \
    --devices "$devices" \
    --theoretical_training False \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length 1000 \
    --save_model False \
    --pretrained_model "bert-base-chinese" \
    --adjust_loss_weight True \
    --cache_root_dir "./cache_dir/debug_run"
fi

if [ "$script_name" == "debug_run_theoretical" ]; then
  python train.py \
    --run_name "debug_run_theoretical" \
    --info "使用少量数据来测试环境是否正常，模型是否能够正常运行；" \
    --use_wandb False \
    --n_epochs 2 \
    --devices "$devices"  \
    --theoretical_training False \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length 1000 \
    --save_model False \
    --pretrained_model "bert-base-chinese" \
    --adjust_loss_weight True \
    --cache_root_dir "./cache_dir/debug_run_theoretical"
fi

if [ "$script_name" == "base_baseline_no_noised" ]; then
  python train.py \
    --run_name "base_baseline_no_noised" \
    --info "bert base baseline；bert+lstm；不在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0" \
    --overall_typo_ratio "None" \
    --typo_type_dist "None" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_baseline_no_noised" \
    --checkpoint_name "base_baseline_no_noised" \
    --use_lstm True
fi


if [ "$script_name" == "base_baseline_noised" ]; then
  python train.py \
    --run_name "base_baseline_noised" \
    --info "bert base baseline；bert+lstm；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_baseline_noised" \
    --checkpoint_name "base_baseline_noised" \
    --use_lstm True
fi


if [ "$script_name" == "large_baseline_no_noised" ]; then
  python train.py \
    --run_name "large_baseline_no_noised" \
    --info "bert large baseline；bert+lstm；不在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0" \
    --overall_typo_ratio "None" \
    --typo_type_dist "None" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "yechen/bert-large-chinese" \
    --cache_root_dir "./cache_dir/large_baseline_no_noised" \
    --checkpoint_name "large_baseline_no_noised" \
    --use_lstm True
fi

if [ "$script_name" == "large_baseline_noised" ]; then
  python train.py \
    --run_name "large_baseline_noised" \
    --info "bert large baseline；bert+lstm；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "yechen/bert-large-chinese" \
    --cache_root_dir "./cache_dir/large_baseline_noised" \
    --checkpoint_name "large_baseline_noised" \
    --use_lstm True
fi

if [ "$script_name" == "base_multitask_no_noised" ]; then
  python train.py \
    --run_name "base_multitask_no_noised" \
    --info "bert base multitask；bert+lstm+bl+cbl+awl；不在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0" \
    --overall_typo_ratio "None" \
    --typo_type_dist "None" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_multitask_no_noised" \
    --checkpoint_name "base_multitask_no_noised" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "base_multitask_noised" ]; then
  python train.py \
    --run_name "base_multitask_noised" \
    --info "bert base multitask；bert+lstm+bl+cbl+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_multitask_noised" \
    --checkpoint_name "base_multitask_noised" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "large_multitask_no_noised" ]; then
  python train.py \
    --run_name "large_multitask_no_noised" \
    --info "bert large multitask；bert+lstm+bl+cbl+awl；不在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0" \
    --overall_typo_ratio "None" \
    --typo_type_dist "None" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "yechen/bert-large-chinese" \
    --cache_root_dir "./cache_dir/large_multitask_no_noised" \
    --checkpoint_name "large_multitask_no_noised" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "large_multitask_noised" ]; then
  python train.py \
    --run_name "large_multitask_noised" \
    --info "bert large multitask；bert+lstm+bl+cbl+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "yechen/bert-large-chinese" \
    --cache_root_dir "./cache_dir/large_multitask_noised" \
    --checkpoint_name "large_multitask_noised" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "base_upbound_no_noised" ]; then
  python train.py \
    --run_name "base_upbound_no_noised" \
    --info "bert base upper bound；bert+lstm+multiple attention+awl；不在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training True \
    --wrong_sent_ratio "0" \
    --overall_typo_ratio "None" \
    --typo_type_dist "None" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_upbound_no_noised" \
    --checkpoint_name "base_upbound_no_noised" \
    --use_lstm True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "base_upbound_noised" ]; then
  python train.py \
    --run_name "base_upbound_noised" \
    --info "bert base upper bound；bert+lstm+multiple attention+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training True \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_upbound_noised" \
    --checkpoint_name "base_upbound_noised" \
    --use_lstm True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "large_upbound_no_noised" ]; then
  python train.py \
    --run_name "large_upbound_no_noised" \
    --info "bert large upper bound；bert+lstm+multiple attention+awl；不在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training True \
    --wrong_sent_ratio "0" \
    --overall_typo_ratio "None" \
    --typo_type_dist "None" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "yechen/bert-large-chinese" \
    --cache_root_dir "./cache_dir/large_upbound_no_noised" \
    --checkpoint_name "large_upbound_no_noised" \
    --use_lstm True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "large_upbound_noised" ]; then
  python train.py \
    --run_name "large_upbound_noised" \
    --info "bert large upper bound；bert+lstm+multiple attention+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training True \
    --wrong_sent_ratio "0.35" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "yechen/bert-large-chinese" \
    --cache_root_dir "./cache_dir/large_upbound_noised" \
    --checkpoint_name "large_upbound_noised" \
    --use_lstm True \
    --adjust_loss_weight True
fi


####### adjust the proportion of perturbed data #######
if [ "$script_name" == "base_multitask_noised_0.2" ]; then
  python train.py \
    --run_name "base_multitask_noised_0.2" \
    --info "bert base multitask；bert+lstm+bl+cbl+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0.2" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_multitask_noised_0.2" \
    --checkpoint_name "base_multitask_noised_0.2" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "base_multitask_noised_0.7" ]; then
  python train.py \
    --run_name "base_multitask_noised_0.7" \
    --info "bert base multitask；bert+lstm+bl+cbl+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "0.7" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_multitask_noised_0.7" \
    --checkpoint_name "base_multitask_noised_0.7" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi

if [ "$script_name" == "base_multitask_noised_1" ]; then
  python train.py \
    --run_name "base_multitask_noised_1" \
    --info "bert base multitask；bert+lstm+bl+cbl+awl；在训练数据中模拟实际的标注错误；" \
    --use_wandb True \
    --n_epochs 20 \
    --devices "$devices" \
    --batch_size $batch_size \
    --accumulation_steps $accumulation_steps \
    --theoretical_training False \
    --wrong_sent_ratio "1" \
    --overall_typo_ratio "(0.72, 0.2, 0.08)" \
    --typo_type_dist "(0.24, 0.16, 0.31, 0.1, 0.15, 0.04)" \
    --train_length -1 \
    --save_model True \
    --pretrained_model "bert-base-chinese" \
    --cache_root_dir "./cache_dir/base_multitask_noised_1" \
    --checkpoint_name "base_multitask_noised_1" \
    --use_lstm True \
    --boundary_predict True \
    --character_boundary_predict True \
    --adjust_loss_weight True
fi


















