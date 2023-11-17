echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
diversity_score_scale=4

for template_idx in 0 1 2 3
do
for method in direct
do
for direct_plus in 0 1
do
for ptm_name in gpt2-large
do

task=amazon_b \
template_idx=$template_idx \
method=$method \
progressive_p=3 \
initial_indication_set_size=4 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
ptm_name=$ptm_name \
final_candidate_size=500 \
label_balance=1 \
candidate_example_num_total=-1 \
candidate_example_num_every_label=4 \
direct_plus=$direct_plus \
diversity_score_scale=$diversity_score_scale \
bash commands/run_filter_and_search.sh

done
done
done
done