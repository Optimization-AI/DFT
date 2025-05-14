### Requirements
Install the required packages with the following command:
```bash
conda create -n eval_math python=3.10.9
conda activate eval_math
cd latex2sympy
pip install -e .
cd ..
pip install vllm==0.5.1 --no-build-isolation
pip install -r requirements.txt 
pip install transformers==4.45.0
```

### Evaluation
Evaluate MetaMath-Mistral-7B-DFT series model with the following command:
```bash
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH=ilgee/MetaMath-Mistral-7B-DFT
OUTPUT_DIR=MetaMath-Mistral-7B-DFT
PROMPT_TYPE="metamath"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
```

## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness) and [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason).
