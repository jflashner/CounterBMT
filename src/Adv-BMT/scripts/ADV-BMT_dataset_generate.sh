dir="/mnt/d/Projects/CounterBMT/data/scenarionet_waymo_training_500" 
save_dir="/mnt/d/Projects/CounterBMT/src/waymo_bmt_output"
num_scenario=1
TF_mode="all_TF_except_adv"
num_mode=1

mkdir -p $save_dir

python bmt/ADV-BMT_dataset_generate.py --dir $dir --save_dir $save_dir --TF_mode $TF_mode --num_scenario $num_scenario --num_mode $num_mode