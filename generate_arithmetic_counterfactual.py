#!/usr/bin/env python3
"""
Counterfactual Arithmetic Problem Dataset Generator

Generates arithmetic problems with corrupted versions (x, x') and their corresponding
correct answers (y, y'). Uses sign switching as the corruption method.

Output: JSON file with counterfactual pairs in numeric format
"""

import argparse
import json
import random
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Import the generator and formatter classes from the original file
from generate_arithmetic_data import (
    ArithmeticProblemGenerator,
    DatasetFormatter,
    split_dataset
)


def corrupt_problem_by_sign_switching(problem: Dict) -> Dict:
    """
    Create a corrupted version of a problem by switching all signs.
    
    Args:
        problem: Original problem dictionary with 'operands' and 'operators'
    
    Returns:
        Corrupted problem dictionary with switched operators
    """
    corrupted = problem.copy()
    
    # Switch all operators: + becomes -, - becomes +
    corrupted_operators = []
    for op in problem['operators']:
        if op == '+':
            corrupted_operators.append('-')
        elif op == '-':
            corrupted_operators.append('+')
        else:
            corrupted_operators.append(op)  # Fallback (shouldn't happen)
    
    corrupted['operators'] = corrupted_operators
    
    # Calculate the result for the corrupted problem
    operands = corrupted['operands']
    result = operands[0]
    
    for i, op in enumerate(corrupted_operators):
        if op == '+':
            result += operands[i + 1]
        elif op == '-':
            result -= operands[i + 1]
    
    corrupted['result'] = result
    
    return corrupted


def generate_counterfactual_dataset(problems: List[Dict]) -> List[Dict]:
    """
    Generate counterfactual dataset with x, x', y, y' for each problem.
    
    Args:
        problems: List of original problem dictionaries
    
    Returns:
        List of counterfactual entries with x, x', y, y'
    """
    formatter = DatasetFormatter()
    counterfactual_entries = []
    
    for problem in problems:
        # Format original problem (x, y)
        x_prompt, y_answer = formatter.format_numeric(problem)
        
        # Create corrupted problem
        corrupted_problem = corrupt_problem_by_sign_switching(problem)
        
        # Format corrupted problem (x', y')
        x_prime_prompt, y_prime_answer = formatter.format_numeric(corrupted_problem)
        
        # Create counterfactual entry
        entry = {
            'id': problem['id'],
            'x': x_prompt,  # Original problem
            'x_prime': x_prime_prompt,  # Corrupted problem
            'y': y_answer,  # Correct answer to x
            'y_prime': y_prime_answer,  # Correct answer to x' (incorrect for x)
            'ground_truth': problem['result'],  # Original correct answer
            'ground_truth_prime': corrupted_problem['result'],  # Corrupted correct answer
            'split': problem['split'],
            'has_carry': problem['has_carry'],
            'n_terms': problem['n_terms'],
            'n_digits': problem['n_digits']
        }
        
        counterfactual_entries.append(entry)
    
    return counterfactual_entries


def save_counterfactual_dataset(entries: List[Dict], output_filename: str):
    """
    Save counterfactual dataset to JSON file.
    
    Args:
        entries: List of counterfactual entries
        output_filename: Output filename
    """
    # Ensure target directory exists: data/json/
    base_dir = os.path.join('data', 'json')
    os.makedirs(base_dir, exist_ok=True)
    
    output_path = os.path.join(base_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {output_path} ({len(entries)} counterfactual entries)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate counterfactual arithmetic problem dataset with sign-switched corruptions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset size
    parser.add_argument('-n', '--num-problems', type=int, default=1000,
                        help='Number of problems to generate')
    
    # Difficulty parameters
    parser.add_argument('--terms', type=float, nargs=2, default=[0.5, 0.5],
                        metavar=('PROB_2', 'PROB_3'),
                        help='Probability distribution for 2-term and 3-term problems')
    parser.add_argument('--digits', type=float, nargs=2, default=[0.5, 0.5],
                        metavar=('PROB_2', 'PROB_3'),
                        help='Probability distribution for 2-digit and 3-digit numbers')
    parser.add_argument('--solution-digits', type=int, default=None,
                        help='Filter to keep only solutions with this many digits (None = no filter)')
    parser.add_argument('--carry-percentage', type=float, default=0.5,
                        help='Proportion of problems with carry/borrow operations')
    
    # Optional constraints
    parser.add_argument('--avoid-repeated-digits', action='store_true',
                        help='Avoid numbers with repeated digits (e.g., 121, 77, 88)')
    parser.add_argument('--avoid-clean-multiples', action='store_true',
                        help='Avoid clean multiples of 10 (e.g., 70, 80, 100)')
    parser.add_argument('--avoid-reverse-pairs', action='store_true',
                        help='Avoid reverse pairs for 2-term addition (if 34+21, skip 21+34)')
    
    # Output
    parser.add_argument('-o', '--output-filename', type=str, 
                        default='arith_dataset_numeric_counterfactual.json',
                        help='Output filename (will be saved in data/json/)')
    
    # Other
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate probability distributions
    if abs(sum(args.terms) - 1.0) > 1e-6:
        parser.error("--terms probabilities must sum to 1.0")
    if abs(sum(args.digits) - 1.0) > 1e-6:
        parser.error("--digits probabilities must sum to 1.0")
    
    print("=== Counterfactual Arithmetic Problem Dataset Generator ===\n")
    print(f"Generating {args.num_problems} problems with the following parameters:")
    print(f"  Terms distribution: 2-term={args.terms[0]:.1%}, 3-term={args.terms[1]:.1%}")
    print(f"  Digits distribution: 2-digit={args.digits[0]:.1%}, 3-digit={args.digits[1]:.1%}")
    print(f"  Solution digits filter: {args.solution_digits if args.solution_digits else 'None'}")
    print(f"  Carry/borrow percentage: {args.carry_percentage:.1%}")
    print(f"  Avoid repeated digits: {args.avoid_repeated_digits}")
    print(f"  Avoid clean multiples: {args.avoid_clean_multiples}")
    print(f"  Avoid reverse pairs: {args.avoid_reverse_pairs}")
    print(f"  Corruption method: Sign switching (all signs)")
    print(f"  Random seed: {args.seed if args.seed else 'None'}")
    print()
    
    # Initialize generator
    generator = ArithmeticProblemGenerator(
        terms_distribution=tuple(args.terms),
        digits_distribution=tuple(args.digits),
        solution_digits=args.solution_digits,
        carry_percentage=args.carry_percentage,
        avoid_repeated_digits=args.avoid_repeated_digits,
        avoid_clean_multiples=args.avoid_clean_multiples,
        avoid_reverse_pairs=args.avoid_reverse_pairs,
        seed=args.seed
    )
    
    # Generate problems
    print("Generating problems...")
    problems = generator.generate_problems(args.num_problems)
    
    if len(problems) == 0:
        print("Error: No problems generated. Try relaxing constraints.")
        return
    
    # Split into train/val/test
    print("\nSplitting into train/val/test...")
    problems = split_dataset(problems)
    
    # Assign unique IDs
    for i, problem in enumerate(problems):
        problem['id'] = f"prob_{i:06d}"
    
    # Generate counterfactual entries
    print("\nGenerating counterfactual pairs (x, x', y, y')...")
    counterfactual_entries = generate_counterfactual_dataset(problems)
    
    # Save dataset
    print("\nSaving counterfactual dataset...")
    save_counterfactual_dataset(counterfactual_entries, args.output_filename)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total counterfactual entries: {len(counterfactual_entries)}")
    
    # Count by split
    split_counts = defaultdict(int)
    for entry in counterfactual_entries:
        split_counts[entry['split']] += 1
    print(f"Train: {split_counts['train']} ({split_counts['train']/len(counterfactual_entries)*100:.1f}%)")
    print(f"Val: {split_counts['val']} ({split_counts['val']/len(counterfactual_entries)*100:.1f}%)")
    print(f"Test: {split_counts['test']} ({split_counts['test']/len(counterfactual_entries)*100:.1f}%)")
    
    # Count by characteristics
    terms_counts = defaultdict(int)
    digits_counts = defaultdict(int)
    carry_counts = defaultdict(int)
    
    for entry in counterfactual_entries:
        terms_counts[entry['n_terms']] += 1
        digits_counts[entry['n_digits']] += 1
        carry_counts[entry['has_carry']] += 1
    
    print(f"\n2-term problems: {terms_counts[2]} ({terms_counts[2]/len(counterfactual_entries)*100:.1f}%)")
    print(f"3-term problems: {terms_counts[3]} ({terms_counts[3]/len(counterfactual_entries)*100:.1f}%)")
    print(f"2-digit problems: {digits_counts[2]} ({digits_counts[2]/len(counterfactual_entries)*100:.1f}%)")
    print(f"3-digit problems: {digits_counts[3]} ({digits_counts[3]/len(counterfactual_entries)*100:.1f}%)")
    print(f"Problems with carry/borrow: {carry_counts[True]} ({carry_counts[True]/len(counterfactual_entries)*100:.1f}%)")
    print(f"Problems without carry/borrow: {carry_counts[False]} ({carry_counts[False]/len(counterfactual_entries)*100:.1f}%)")
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()

