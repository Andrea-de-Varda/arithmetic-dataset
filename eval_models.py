#!/usr/bin/env python3
"""
Evaluate LLMs on arithmetic datasets

Usage:
    python eval_models.py --model mistralai/Mistral-7B-v0.1 --dataset all
    python eval_models.py --model mistralai/Magistral-Small-2509 --reasoning --dataset embedded
"""

import argparse
import json
import os
import re
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Dataset configurations
DATASETS = {
    'numeric': 'data/json/arith_dataset_numeric.json',
    'english': 'data/json/arith_dataset_english.json',
    'spanish': 'data/json/arith_dataset_spanish.json',
    'italian': 'data/json/arith_dataset_italian.json',
    'embedded': 'data/json/arith_dataset_embedded.json',
    'embedded_verbal': 'data/json/arith_dataset_embedded_verbal.json',
}

# Prompt suffixes for standard LLMs (to guide answer generation)
# These are appended to the prompt before generation
PROMPT_SUFFIXES = {
    'numeric': '',       # "12 + 41 =" already ends with =
    'english': '',       # "twelve plus forty-one equals" already ends appropriately
    'spanish': '',       # ends with "es igual a"
    'italian': '',       # ends with "fa"
    'embedded': ' The answer is',  # Word problems need guidance
    'embedded_verbal': ' The answer is',  # Word problems need guidance
}

# One-shot exemplars to steer standard models to return a bare answer.
ONE_SHOT_EXAMPLES = {
    'numeric': (
        "7 + 5 = 12\n"
        "{prompt}"
    ),
    'english': (
        "seven plus five equals twelve\n"
        "{prompt}"
    ),
    'spanish': (
        "siete más cinco es igual a doce\n"
        "{prompt}"
    ),
    'italian': (
        "sette più cinque fa dodici\n"
        "{prompt}"
    ),
    'embedded': (
        "Question: Alice has 3 marbles and finds 2 more. How many does she have now?\n"
        "Answer: 5\n"
        "{prompt}"
    ),
    'embedded_verbal': (
        "Question: Alice has 3 marbles and finds 2 more. How many does she have now?\n"
        "Answer: 5\n"
        "{prompt}"
    ),
    'default': (
        "2 + 2 = 4\n"
        "{prompt}"
    ),
}


def load_dataset(dataset_path: str, split: str = 'test') -> list:
    """Load dataset and filter by split"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item for item in data if item['split'] == split]


def normalize_answer(answer: str, dataset_type: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.strip()
    
    if dataset_type in ['numeric', 'embedded']:
        # Extract just the number
        match = re.search(r'-?\d+', answer)
        if match:
            return match.group()
        return answer
    else:
        # For verbal (english/spanish/italian/embedded_verbal), lowercase and clean
        # Remove punctuation, extra spaces
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s-]', '', answer)
        answer = ' '.join(answer.split())
        return answer


def build_standard_prompt(prompt: str, dataset_type: str, one_shot: bool) -> str:
    """Optionally prepend a one-shot example to steer concise answers."""
    if not one_shot:
        return prompt
    exemplar = ONE_SHOT_EXAMPLES.get(dataset_type, ONE_SHOT_EXAMPLES['default'])
    return exemplar.format(prompt=prompt)

def extract_answer_standard(text: str, dataset_type: str) -> str:
    """Extract the answer from standard model output"""
    text = text.strip()
    
    if dataset_type in ['numeric', 'embedded']:
        # For numeric, extract the first number
        match = re.search(r'-?\d+', text)
        if match:
            return match.group()
        return text
    else:
        # For verbal, take first meaningful response
        # Stop at newline, period, or common stop patterns
        text = text.split('\n')[0]
        text = re.split(r'[.!?]', text)[0]
        return text.strip()

def extract_answer_reasoning(text: str, dataset_type: str) -> str:
    """Extract final answer from reasoning model output"""
    # Look for explicit ANSWER: pattern (case insensitive)
    match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up any trailing punctuation
        answer = re.sub(r'[.!?]+$', '', answer)
        return normalize_answer(answer, dataset_type)
    
    # Look for "the answer is X" pattern
    match = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\n|$|\.)', text, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1).strip(), dataset_type)
    
    # Look for "= X" pattern
    match = re.search(r'=\s*(-?\d+)\s*(?:\n|$|\.)', text)
    if match:
        return match.group(1)
    
    # For numeric/embedded: get last number in text
    if dataset_type in ['numeric', 'embedded']:
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return numbers[-1]
    
    # Last resort: return last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return normalize_answer(lines[-1], dataset_type)
    
    return ""


def check_answer(predicted: str, expected: str, dataset_type: str) -> bool:
    """Check if predicted answer matches expected"""
    pred_norm = normalize_answer(predicted, dataset_type)
    exp_norm = normalize_answer(expected, dataset_type)
    
    if dataset_type in ['numeric', 'embedded']:
        # Exact numeric match
        return pred_norm == exp_norm
    else:
        # For verbal: check if prediction starts with or equals expected
        # This handles cases where model might generate extra words
        return pred_norm == exp_norm or pred_norm.startswith(exp_norm)


def evaluate_standard(model, tokenizer, dataset: list, dataset_type: str,
                      max_new_tokens: int = 8,
                      log_every: int = 100,
                      one_shot: bool = False) -> dict:
    """Evaluate standard (non-reasoning) LLM"""
    results = []
    correct = 0
    
    suffix = PROMPT_SUFFIXES.get(dataset_type, '')
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_type}")):
        prompt = item['prompt'] + suffix
        full_prompt = build_standard_prompt(prompt, dataset_type, one_shot)
        expected = item['answer']
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_len = inputs['input_ids'].shape[1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Extract and check answer
        predicted = extract_answer_standard(generated, dataset_type)
        is_correct = check_answer(predicted, expected, dataset_type)
        
        if is_correct:
            correct += 1
        
        results.append({
            'id': item['id'],
            'prompt': item['prompt'],
            'full_prompt': full_prompt,
            'expected': expected.strip(),
            'generated': generated,
            'predicted': predicted,
            'correct': is_correct,
        })

        # Periodic logging
        if (idx + 1) % log_every == 0:
            seen = idx + 1
            acc_so_far = correct / seen if seen else 0.0
            print(f"[progress] {dataset_type}: {seen}/{len(dataset)} processed, acc={acc_so_far*100:.2f}%", flush = True)
    
    accuracy = correct / len(dataset) if dataset else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(dataset),
        'results': results,
    }


def build_system_prompt(dataset_type: str) -> str:
    """Build a system prompt adapted to dataset type/language/format."""
    if dataset_type == 'numeric':
        return (
            "You are a helpful assistant that solves arithmetic expressions written in numbers. "
            "Think step by step, then provide ONLY the final numeric result. "
            "Your final answer MUST be on its own line in exactly this format:\n"
            "ANSWER: <number>"
        )
    if dataset_type == 'embedded':
        return (
            "You are a helpful assistant that solves short word problems in English about counts of objects. "
            "Think step by step, then provide ONLY the final numeric result. "
            "Your final answer MUST be on its own line in exactly this format:\n"
            "ANSWER: <number>"
        )
    if dataset_type == 'embedded_verbal':
        return (
            "You are a helpful assistant that solves short word problems in English about counts of objects. "
            "Think step by step, then provide ONLY final answer written in English words (no punctuation). "
            "Your final answer MUST be on its own line in exactly this format:\n"
            "ANSWER: <english words>"
        )
    if dataset_type == 'english':
        return (
            "You are a helpful assistant that solves arithmetic problems written in English words. "
            "Think step by step, then provide ONLY the final answer written in English words (no punctuation). "
            "Your final answer MUST be on its own line in exactly this format:\n"
            "ANSWER: <english words>"
        )
    if dataset_type == 'spanish':
        return (
            "Eres un asistente útil que resuelve problemas aritméticos escritos en palabras en español. "
            "Razona paso a paso y luego entrega SOLO la respuesta final escrita en español (sin puntuación). "
            "Tu respuesta final DEBE estar en su propia línea con exactamente este formato:\n"
            "ANSWER: <palabras en español>"
        )
    if dataset_type == 'italian':
        return (
            "Sei un assistente che risolve problemi aritmetici scritti in parole in italiano. "
            "Ragiona passo dopo passo e poi fornisci SOLO la risposta finale scritta in italiano (senza punteggiatura). "
            "La risposta finale DEVE essere su una riga separata in questo formato esatto:\n"
            "ANSWER: <parole in italiano>"
        )
    # Fallback
    return (
        "You are a helpful assistant that solves arithmetic problems. "
        "Think step by step, then provide ONLY the final answer on its own line:\n"
        "ANSWER: <answer>"
    )


def _build_reasoning_length_kwargs(max_new_tokens, model, tokenizer, input_len: int) -> dict:
    """Return generation length kwargs with no hard cap unless explicitly provided."""
    if max_new_tokens is not None:
        return {"max_new_tokens": max_new_tokens}

    # Prefer model context window; fallback to tokenizer hint; ignore absurd sentinels.
    candidates = []
    max_pos = getattr(model.config, "max_position_embeddings", None)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    for val in (max_pos, tok_limit):
        if val is None:
            continue
        if isinstance(val, float) and val == float("inf"):
            continue
        if val > 0 and val < 10_000_000:  # skip gigantic sentinel defaults
            candidates.append(val)

    if candidates:
        max_length = max(input_len + 1, min(candidates))
    else:
        # If we have no reliable context info, allow plenty of room.
        max_length = input_len + 4096

    return {"max_length": max_length}


def evaluate_reasoning(model, tokenizer, dataset: list, dataset_type: str,
                       max_new_tokens=None,
                       log_every: int = 100) -> dict:
    """Evaluate reasoning LLM with chat template"""
    results = []
    correct = 0
    
    # System prompt for reasoning models, adapted per dataset type
    system_prompt = build_system_prompt(dataset_type)

    pbar = tqdm(total=len(dataset), desc=f"Evaluating {dataset_type} (reasoning)", smoothing=0)

    for idx, item in enumerate(dataset):
        prompt = item['prompt']
        expected = item['answer']
        
        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        try:
            chat_input = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=True
            )
            inputs = {
                "input_ids": chat_input.to(model.device),
                "attention_mask": torch.ones_like(chat_input).to(model.device),
            }
            input_len = chat_input.shape[1]
        except Exception:
            # Fallback if chat template not available
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            inputs = tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs['input_ids'].shape[1]
        
        # Generate
        length_kwargs = _build_reasoning_length_kwargs(max_new_tokens, model, tokenizer, input_len)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **length_kwargs,
            )
        
        # Decode full response (includes reasoning)
        full_response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Extract just the answer
        predicted = extract_answer_reasoning(full_response, dataset_type)
        is_correct = check_answer(predicted, expected, dataset_type)
        
        if is_correct:
            correct += 1
        
        results.append({
            'id': item['id'],
            'prompt': item['prompt'],
            'expected': expected.strip(),
            'reasoning': full_response,  # Full reasoning chain
            'predicted': predicted,       # Extracted answer only
            'correct': is_correct,
        })

        # Running accuracy after every example (displayed in tqdm)
        seen = idx + 1
        acc_so_far = correct / seen if seen else 0.0
        pbar.set_postfix_str(f"acc={acc_so_far*100:.2f}%")
        pbar.update(1)

    pbar.close()
    
    accuracy = correct / len(dataset) if dataset else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(dataset),
        'results': results,
    }


def save_results(results: dict, model_name: str, dataset_type: str,
                 reasoning: bool, output_dir: str = 'results'):
    """Save evaluation results"""
    # Create output directory structure
    model_safe = model_name.replace('/', '__')
    mode = 'reasoning' if reasoning else 'standard'
    
    result_dir = Path(output_dir) / model_safe / mode
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    result_file = result_dir / f'{dataset_type}_results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary = {
        'model': model_name,
        'dataset': dataset_type,
        'mode': mode,
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_file = result_dir / f'{dataset_type}_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {result_dir}")
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on arithmetic datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'numeric', 'english', 'spanish', 'italian', 'embedded', 'embedded_verbal'],
                        help='Dataset to evaluate on')
    parser.add_argument('--reasoning', action='store_true',
                        help='Use reasoning mode with chat template (for reasoning LLMs)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate (for quick testing)')
    parser.add_argument('--max-new-tokens', type=int, default=8,
                        help='Max new tokens for standard mode')
    parser.add_argument('--max-new-tokens-reasoning', type=int, default=None,
                        help='Max new tokens for reasoning mode (default: use full context window)')
    parser.add_argument('--one-shot', action='store_true',
                        help='Prepend a one-shot example to standard prompts to reinforce answer format')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Log interim accuracy every N samples')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype for loading')
    
    args = parser.parse_args()
    
    # Determine datasets to evaluate
    if args.dataset == 'all':
        datasets_to_eval = list(DATASETS.keys())
    else:
        datasets_to_eval = [args.dataset]
    
    print("=" * 60)
    print("Arithmetic Dataset Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: {'reasoning' if args.reasoning else 'standard'}")
    print(f"Datasets: {', '.join(datasets_to_eval)}")
    print(f"Split: {args.split}")
    if args.one_shot and not args.reasoning:
        print("One-shot prompting: enabled for standard mode")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    dtype_map = {
        'auto': 'auto',
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully")
    print()
    
    # Evaluate each dataset
    all_summaries = []
    
    for dataset_type in datasets_to_eval:
        print("-" * 40)
        print(f"Evaluating: {dataset_type}")
        print("-" * 40)
        
        # Load dataset
        dataset_path = DATASETS[dataset_type]
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset not found at {dataset_path}, skipping...")
            continue
            
        dataset = load_dataset(dataset_path, args.split)
        
        if args.max_samples:
            dataset = dataset[:args.max_samples]
        
        print(f"Loaded {len(dataset)} samples from {args.split} split")
        
        # Evaluate
        if args.reasoning:
            results = evaluate_reasoning(
                model,
                tokenizer,
                dataset,
                dataset_type,
                max_new_tokens=args.max_new_tokens_reasoning,
                log_every=args.log_every,
            )
        else:
            results = evaluate_standard(
                model, tokenizer, dataset, dataset_type,
                max_new_tokens=args.max_new_tokens,
                log_every=args.log_every,
                one_shot=args.one_shot,
            )
        
        # Save results
        save_results(results, args.model, dataset_type, args.reasoning, args.output_dir)
        
        acc_pct = results['accuracy'] * 100
        print(f"Accuracy: {acc_pct:.2f}% ({results['correct']}/{results['total']})")
        print()
        
        all_summaries.append({
            'dataset': dataset_type,
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
        })
    
    # Print final summary
    print("=" * 60)
    print("Final Summary")
    print("=" * 60)
    for s in all_summaries:
        print(f"  {s['dataset']:12s}: {s['accuracy']*100:6.2f}% ({s['correct']}/{s['total']})")
    
    if all_summaries:
        avg_acc = sum(s['accuracy'] for s in all_summaries) / len(all_summaries)
        print(f"  {'Average':12s}: {avg_acc*100:6.2f}%")
    
    # Save overall summary
    model_safe = args.model.replace('/', '__')
    mode = 'reasoning' if args.reasoning else 'standard'
    summary_dir = Path(args.output_dir) / model_safe / mode
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    overall_summary = {
        'model': args.model,
        'mode': mode,
        'split': args.split,
        'timestamp': datetime.now().isoformat(),
        'results': all_summaries,
        'average_accuracy': avg_acc if all_summaries else 0,
    }
    
    with open(summary_dir / 'overall_summary.json', 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print()
    print(f"All results saved to: {summary_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
