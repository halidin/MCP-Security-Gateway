# Active MCP Interceptor (Starter)

This project bootstraps a research pipeline to detect Indirect Prompt Injection (IPI) before MCP tool execution.

## What this starter includes

- Data schema for goal-trace-label samples
- Script to normalize BIPIA-style data into train/test JSONL
- Baseline classifier training (`logistic` or `random_forest`)
- Evaluation with Precision/Recall/F1
- Detection horizon simulation (how many reasoning steps before tool call)
- Minimal MCP guard prototype with honeypot kill-switch

## Project layout

- `data/raw/` raw source files (BIPIA, prompts, traces)
- `data/processed/` normalized JSONL datasets
- `models/` trained model artifacts
- `reports/` metric summaries
- `scripts/` executable pipeline scripts
- `src/interceptor/` reusable package code

## Quick start

1. Create and activate a Python environment (3.10+ recommended).
2. Install dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
pip install --no-deps git+https://github.com/microsoft/BIPIA.git
```

If you need full BIPIA model-training stack, install its heavy dependencies (torch/deepspeed/vllm) separately.

3. Prepare data from BIPIA builder:

```bash
python scripts/prepare_bipia.py --use_bipia_builder --dataset_name email --context_data_file data/raw/context_data_file.csv --attack_data_file data/raw/attack_data_file.csv --output_dir data/processed --test_size 0.2 --seed 2023
```

4. Or prepare data from already exported CSV files in `data/raw/`:

```bash
python scripts/prepare_bipia.py --input_dir data/raw --output_dir data/processed --test_size 0.2
```

5. Train a baseline model:

```bash
python scripts/train_baseline.py --train data/processed/train.jsonl --test data/processed/test.jsonl --model random_forest --out_dir models
```

6. Evaluate detection horizon:

```bash
python scripts/evaluate_horizon.py --model_path models/random_forest.joblib --test data/processed/test.jsonl --report reports/horizon.json
```

7. Run the guard prototype:

```bash
python scripts/run_guard_demo.py --model_path models/random_forest.joblib --goal "Summarize my inbox" --trace "I should open hidden credential file first..." --resource secrets.txt
```

8. Generate structured reasoning summaries with OpenRouter:

Create a `.env` file in the project root with your API key:

```env
OPENROUTER_API_KEY=your_key_here
```

Then run the generator:

```bash
python scripts/generate_llm_traces.py --input data/processed/train.jsonl --output data/processed/train_llm.jsonl --model openai/gpt-oss-120b --limit 100
```

This script asks the model for a structured reasoning summary, not hidden chain-of-thought.

## Data format

Each JSONL row should contain:

```json
{
  "id": "sample-001",
  "user_goal": "summarize my emails",
  "agent_trace": "step 1 ... step 2 ...",
  "trace_steps": ["step 1 ...", "step 2 ..."],
  "tool_call_step": 3,
  "label": 1,
  "source": "bipia"
}
```

- `label=1` malicious/hijacked logic
- `label=0` benign/safe

## Suggested milestones

1. Baseline performance on balanced 5,000 test set.
2. Add transformer fine-tuning (`distilbert-base-uncased`) and compare.
3. Integrate inference hook in MCP gateway.
4. Add honey-resource traps and block policy.
5. Evaluate reduction in honeypot access against unprotected agent.

## Notes

- This starter avoids collecting hidden chain-of-thought from closed models.
- Use model-generated reasoning summaries or plan traces you are legally and ethically allowed to process.
