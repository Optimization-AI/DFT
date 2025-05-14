(
    CUDA_VISIBLE_DEVICES=0 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 0  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen0.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=0 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 1  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen1.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=0 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 2  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen2.log 2>&1
) &
(
    CUDA_VISIBLE_DEVICES=1 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 3  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen3.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=1 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 4  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen4.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=1 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 5  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen5.log 2>&1
) &
(
    CUDA_VISIBLE_DEVICES=2 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 6  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen6.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=2 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 7  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen7.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=2 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 8  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen8.log 2>&1
) &
(
    CUDA_VISIBLE_DEVICES=3 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 9  --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen9.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=3 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 10 --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen10.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=3 nohup python3 generator/gen.py --model mistralai/Mistral-7B-v0.1 --revision 7231864981174d9bee8c7687c24c8344414eae6b --input_dir meta-math/MetaMathQA --split train --world_size 1 --seed 11 --temp 0.7 --top_p 1.0 --top_k 50 --max_new_tokens 256 --output_prefix mistral_mmqa > gen11.log 2>&1
) &
wait