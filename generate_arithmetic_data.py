#!/usr/bin/env python3
"""
Flexible Arithmetic Problem Dataset Generator

Generates arithmetic problems with control over:
- Difficulty (terms, digits, carry operations)
- Format (numeric, English, Spanish, Italian, embedded-in-context)
- Various constraints and filters

Output: 5 JSON files (one per format) with matched IDs and train/val/test splits
"""

import argparse
import json
import random
import hashlib
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from num2words import num2words


class ArithmeticProblemGenerator:
    """Generate arithmetic problems with flexible constraints"""
    
    # Language-specific arithmetic words
    ARITHMETIC_WORDS = {
        'en': {'plus': 'plus', 'minus': 'minus', 'equals': 'equals'},
        'es': {'plus': 'más', 'minus': 'menos', 'equals': 'es igual a'},
        'it': {'plus': 'più', 'minus': 'meno', 'equals': 'fa'},
    }
    
    def __init__(self, 
                 terms_distribution: Tuple[float, float] = (0.5, 0.5),
                 digits_distribution: Tuple[float, float] = (0.5, 0.5),
                 solution_digits: Optional[int] = None,
                 carry_percentage: float = 0.5,
                 avoid_repeated_digits: bool = False,
                 avoid_clean_multiples: bool = False,
                 avoid_reverse_pairs: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize the problem generator
        
        Args:
            terms_distribution: (prob_2_terms, prob_3_terms)
            digits_distribution: (prob_2_digits, prob_3_digits)
            solution_digits: Filter solutions to this many digits (None = no filter)
            carry_percentage: Proportion of problems with carry operations
            avoid_repeated_digits: Avoid numbers like 77, 88
            avoid_clean_multiples: Avoid numbers like 70, 80, 100
            avoid_reverse_pairs: Avoid reverse pairs (34+21 and 21+34)
            seed: Random seed for reproducibility
        """
        self.terms_distribution = terms_distribution
        self.digits_distribution = digits_distribution
        self.solution_digits = solution_digits
        self.carry_percentage = carry_percentage
        self.avoid_repeated_digits = avoid_repeated_digits
        self.avoid_clean_multiples = avoid_clean_multiples
        self.avoid_reverse_pairs = avoid_reverse_pairs
        
        if seed is not None:
            random.seed(seed)
    
    def _has_repeated_digits(self, num: int) -> bool:
        """Check if a number has repeated digits (e.g., 77, 88)"""
        s = str(abs(num))
        return len(set(s)) != len(s)
    
    def _is_clean_multiple(self, num: int) -> bool:
        """Check if a number is a clean multiple of 10"""
        return num % 10 == 0
    
    def _is_valid_number(self, num: int) -> bool:
        """Check if a number passes all constraint filters"""
        if self.avoid_repeated_digits and self._has_repeated_digits(num):
            return False
        if self.avoid_clean_multiples and self._is_clean_multiple(num):
            return False
        return True
    
    def _has_carry(self, num1: int, num2: int) -> bool:
        """Check if adding two numbers requires a carry operation"""
        # Check each position for carry
        s1 = str(abs(num1))[::-1]  # Reverse for easier position checking
        s2 = str(abs(num2))[::-1]
        
        carry = 0
        max_len = max(len(s1), len(s2))
        
        for i in range(max_len):
            d1 = int(s1[i]) if i < len(s1) else 0
            d2 = int(s2[i]) if i < len(s2) else 0
            total = d1 + d2 + carry
            
            if total >= 10:
                return True
            carry = total // 10
        
        return False
    
    def _has_borrow(self, num1: int, num2: int) -> bool:
        """Check if subtracting num2 from num1 requires borrowing"""
        # Check each position for borrow
        s1 = str(abs(num1))[::-1]  # Reverse for easier position checking
        s2 = str(abs(num2))[::-1]
        
        borrow = 0
        max_len = max(len(s1), len(s2))
        
        for i in range(max_len):
            d1 = int(s1[i]) if i < len(s1) else 0
            d2 = int(s2[i]) if i < len(s2) else 0
            
            # Apply previous borrow
            d1 -= borrow
            
            # Check if we need to borrow for this position
            if d1 < d2:
                borrow = 1
                return True
            else:
                borrow = 0
        
        return False
    
    def _generate_number(self, n_digits: int) -> int:
        """Generate a random number with n digits that passes constraints"""
        if n_digits == 2:
            min_val, max_val = 10, 99
        elif n_digits == 3:
            min_val, max_val = 100, 999
        else:
            raise ValueError(f"Unsupported digit count: {n_digits}")
        
        max_attempts = 10000
        for _ in range(max_attempts):
            num = random.randint(min_val, max_val)
            if self._is_valid_number(num):
                return num
        
        raise RuntimeError(f"Could not generate valid {n_digits}-digit number after {max_attempts} attempts")
    
    def _generate_two_term_problem(self, n_digits: int, require_carry: bool) -> Optional[Dict]:
        """Generate a 2-term addition or subtraction problem"""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            num1 = self._generate_number(n_digits)
            num2 = self._generate_number(n_digits)
            
            # Randomly choose operation (+ or -)
            op = random.choice(['+', '-'])
            
            if op == '+':
                # Check carry requirement for addition
                has_carry_or_borrow = self._has_carry(num1, num2)
                if has_carry_or_borrow != require_carry:
                    continue
                
                result = num1 + num2
            else:
                # For subtraction, ensure positive result
                if num1 <= num2:
                    continue
                
                # Check borrow requirement for subtraction
                has_carry_or_borrow = self._has_borrow(num1, num2)
                if has_carry_or_borrow != require_carry:
                    continue
                
                result = num1 - num2
            
            # Check result validity
            if not self._is_valid_number(result):
                continue
            
            # Check solution digit constraint
            if self.solution_digits is not None:
                result_digits = len(str(result))
                if result_digits != self.solution_digits:
                    continue
            
            return {
                'operands': [num1, num2],
                'operators': [op],
                'result': result,
                'has_carry': has_carry_or_borrow,
                'n_terms': 2,
                'n_digits': n_digits
            }
        
        return None
    
    def _generate_three_term_problem(self, n_digits: int, require_carry: bool) -> Optional[Dict]:
        """Generate a 3-term problem (addition/subtraction)"""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            num1 = self._generate_number(n_digits)
            num2 = self._generate_number(n_digits)
            num3 = self._generate_number(n_digits)
            
            # Randomly choose operators (+ or -)
            op1 = random.choice(['+', '-'])
            op2 = random.choice(['+', '-'])
            
            # Calculate result
            result = num1
            if op1 == '+':
                result += num2
            else:
                result -= num2
            
            if op2 == '+':
                result += num3
            else:
                result -= num3
            
            # Skip negative results
            if result <= 0:
                continue
            
            # Check carry/borrow requirement (check each operation)
            if op1 == '+':
                has_carry_borrow_1 = self._has_carry(num1, num2)
            else:
                has_carry_borrow_1 = self._has_borrow(num1, num2)
            
            intermediate = num1 + num2 if op1 == '+' else num1 - num2
            
            if op2 == '+':
                has_carry_borrow_2 = self._has_carry(intermediate, num3)
            else:
                has_carry_borrow_2 = self._has_borrow(intermediate, num3) if intermediate > num3 else False
            
            has_carry_or_borrow = has_carry_borrow_1 or has_carry_borrow_2
            
            if has_carry_or_borrow != require_carry:
                continue
            
            # Check result validity
            if not self._is_valid_number(result):
                continue
            
            # Check solution digit constraint
            if self.solution_digits is not None:
                result_digits = len(str(result))
                if result_digits != self.solution_digits:
                    continue
            
            return {
                'operands': [num1, num2, num3],
                'operators': [op1, op2],
                'result': result,
                'has_carry': has_carry_or_borrow,
                'n_terms': 3,
                'n_digits': n_digits
            }
        
        return None
    
    def generate_problem(self) -> Optional[Dict]:
        """Generate a single problem based on configured distributions"""
        # Decide number of terms
        if random.random() < self.terms_distribution[0]:
            n_terms = 2
        else:
            n_terms = 3
        
        # Decide number of digits
        if random.random() < self.digits_distribution[0]:
            n_digits = 2
        else:
            n_digits = 3
        
        # Decide carry requirement
        require_carry = random.random() < self.carry_percentage
        
        # Generate the problem
        if n_terms == 2:
            return self._generate_two_term_problem(n_digits, require_carry)
        else:
            return self._generate_three_term_problem(n_digits, require_carry)
    
    def generate_problems(self, n_problems: int, max_attempts: int = None) -> List[Dict]:
        """
        Generate multiple unique problems
        
        Args:
            n_problems: Target number of problems to generate
            max_attempts: Maximum attempts (default: n_problems * 10)
        
        Returns:
            List of problem dictionaries
        """
        if max_attempts is None:
            max_attempts = n_problems * 10
        
        problems = []
        seen_problems = set()  # To track uniqueness
        
        attempts = 0
        while len(problems) < n_problems and attempts < max_attempts:
            attempts += 1
            problem = self.generate_problem()
            
            if problem is None:
                continue
            
            # Create a signature for uniqueness checking
            signature = self._create_problem_signature(problem)
            
            if signature in seen_problems:
                continue
            
            seen_problems.add(signature)
            problems.append(problem)
            
            # Progress indicator
            if len(problems) % 100 == 0:
                print(f"Generated {len(problems)}/{n_problems} problems...")
        
        if len(problems) < n_problems:
            print(f"Warning: Only generated {len(problems)} problems out of {n_problems} requested")
        
        return problems
    
    def _create_problem_signature(self, problem: Dict) -> str:
        """Create a unique signature for a problem"""
        operands = problem['operands']
        operators = problem['operators']
        
        # For reverse pair detection (only applies to 2-term addition)
        if self.avoid_reverse_pairs and len(operators) == 1 and operators[0] == '+':
            # Sort operands to normalize order for commutative operations
            sig_parts = sorted([str(op) for op in operands])
            sig_parts.append('_'.join(operators))
            return '_'.join(sig_parts)
        else:
            # Keep original order for all other cases
            sig_parts = [str(op) for op in operands]
            sig_parts.extend(operators)
            return '_'.join(sig_parts)


class DatasetFormatter:
    """Format problems into different representations"""
    
    @staticmethod
    def format_numeric(problem: Dict) -> Tuple[str, str]:
        """Format as numeric expression"""
        operands = problem['operands']
        operators = problem['operators']
        result = problem['result']
        
        # Build expression
        expr_parts = [str(operands[0])]
        for i, op in enumerate(operators):
            expr_parts.append(f" {op} {operands[i+1]}")
        
        prompt = ''.join(expr_parts) + ' ='
        answer = f' {result}'
        
        return prompt, answer
    
    @staticmethod
    def format_verbal(problem: Dict, language: str = 'en') -> Tuple[str, str]:
        """Format as verbal expression in specified language"""
        operands = problem['operands']
        operators = problem['operators']
        result = problem['result']
        
        words = ArithmeticProblemGenerator.ARITHMETIC_WORDS[language]
        
        # Convert numbers to words
        verbal_operands = [num2words(num, lang=language) for num in operands]
        verbal_result = num2words(result, lang=language)
        
        # Build expression
        expr_parts = [verbal_operands[0]]
        for i, op in enumerate(operators):
            op_word = words['plus'] if op == '+' else words['minus']
            expr_parts.append(f" {op_word} {verbal_operands[i+1]}")
        
        prompt = ''.join(expr_parts) + f" {words['equals']}"
        answer = f' {verbal_result}'
        
        return prompt, answer
    
    @staticmethod
    def format_embedded_context(problem: Dict, verbal: bool = False) -> Tuple[str, str]:
        """Format as embedded-in-context (short story word problem)"""
        # Define possible characters, objects, and templates
        characters = ["Alex", "Emma", "Noah", "Olivia", "Liam", "Sophia", "Jackson", "Ava",
                     "Lucas", "Isabella", "Mason", "Mia", "Ethan", "Charlotte", "Logan",
                     "Amelia", "Aiden", "Harper", "James", "Evelyn", "Lily", "Benjamin",
                     "Michael", "Sofia", "Jacob", "Abigail", "Daniel", "Emily", "Henry",
                     "Ella", "Matthew", "Madison", "Samuel", "Scarlett", "David", "Victoria"]
        
        objects = [
            {"name": "apple", "plural": "apples", "context": "fruit"},
            {"name": "orange", "plural": "oranges", "context": "fruit"},
            {"name": "book", "plural": "books", "context": "school"},
            {"name": "pencil", "plural": "pencils", "context": "school"},
            {"name": "cookie", "plural": "cookies", "context": "food"},
            {"name": "sticker", "plural": "stickers", "context": "collection"},
            {"name": "marble", "plural": "marbles", "context": "game"},
            {"name": "toy car", "plural": "toy cars", "context": "toy"},
            {"name": "doll", "plural": "dolls", "context": "toy"},
            {"name": "stamp", "plural": "stamps", "context": "collection"},
            {"name": "coin", "plural": "coins", "context": "money"},
            {"name": "flower", "plural": "flowers", "context": "garden"},
            {"name": "balloon", "plural": "balloons", "context": "party"},
            {"name": "chocolate", "plural": "chocolates", "context": "food"},
            {"name": "card", "plural": "cards", "context": "game"},
        ]
        
        # Templates for addition problems
        addition_templates = [
            "{char1} has {num1} {obj_plural}. {char2} has {num2} {obj_plural}. How many {obj_plural} do they have together?",
            "{char1} found {num1} {obj_plural}. Then {char1} found {num2} more {obj_plural}. How many {obj_plural} did {char1} find in total?",
            "There are {num1} {obj_plural} in the basket. {char1} puts {num2} more {obj_plural} in the basket. How many {obj_plural} are in the basket now?",
            "{char1} baked {num1} {obj_plural}. {char2} baked {num2} {obj_plural}. How many {obj_plural} did they bake altogether?",
            "{char1} collected {num1} {obj_plural} on Monday and {num2} {obj_plural} on Tuesday. How many {obj_plural} did {char1} collect in these two days?",
        ]
        
        # Templates for subtraction problems
        subtraction_templates = [
            ("{char1} had {num_total} {obj_plural}. {char1} gave {num2} {obj_plural} to {char2}. How many {obj_plural} does {char1} have left?", 'subtract'),
            ("There were {num_total} {obj_plural} on the table. {char1} took {num2} {obj_plural}. How many {obj_plural} remain on the table?", 'subtract'),
            ("{char1} had {num_total} {obj_plural}. After giving some to {char2}, {char1} had {num1} {obj_plural} left. How many {obj_plural} did {char1} give to {char2}?", 'difference'),  # Answer is different
            ("{char1} had some {obj_plural}. After {char1} got {num2} more {obj_plural}, {char1} had {num_total} {obj_plural}. How many {obj_plural} did {char1} have at first?", 'difference'),  # Answer is different
            ("{char1} had {num_total} {obj_plural}. {num2} of them were eaten. How many {obj_plural} does {char1} have now?", 'subtract'),
        ]
        
        # Get problem details
        operands = problem['operands']
        operators = problem['operators']
        result = problem['result']
        
        # Helper to format numbers
        def fmt_num(n):
            return num2words(n, lang='en') if verbal else n
        
        # Select random characters and object using a stable, deterministic seed
        # Seed is derived from problem id (if available), operands and operators
        seed_source = f"{problem.get('id', '')}|{operands}|{operators}"
        seed_int = int.from_bytes(hashlib.sha256(seed_source.encode('utf-8')).digest()[:8], 'big')
        temp_rng = random.Random(seed_int)
        
        char1 = temp_rng.choice(characters)
        char2 = temp_rng.choice(characters)
        while char2 == char1:
            char2 = temp_rng.choice(characters)
        
        obj = temp_rng.choice(objects)
        obj_plural = obj['plural']
        
        # Handle different operation types
        if len(operators) == 1 and operators[0] == '+':
            # Simple addition (2 terms)
            num1, num2 = fmt_num(operands[0]), fmt_num(operands[1])
            template = temp_rng.choice(addition_templates)
            prompt = template.format(char1=char1, char2=char2, num1=num1, num2=num2, obj_plural=obj_plural)
            answer = f' {fmt_num(result)}'
            
        elif len(operators) == 2 and operators[0] == '+' and operators[1] == '+':
            # Three-term addition
            num1, num2, num3 = fmt_num(operands[0]), fmt_num(operands[1]), fmt_num(operands[2])
            char3 = temp_rng.choice(characters)
            while char3 == char1 or char3 == char2:
                char3 = temp_rng.choice(characters)
            prompt = f"{char1} has {num1} {obj_plural}, {char2} has {num2} {obj_plural}, and {char3} has {num3} {obj_plural}. How many {obj_plural} do they have in total?"
            answer = f' {fmt_num(result)}'
            
        elif len(operators) == 1 and operators[0] == '-':
            # Simple subtraction (2 terms)
            num1, num2 = operands[0], operands[1]
            num_total = num1
            template_tuple = temp_rng.choice(subtraction_templates)
            template, answer_type = template_tuple
            
            if answer_type == 'difference':
                # Two different "difference" templates with different answers
                if 'got' in template:
                    # Template B: "After got {num2} more, had {num_total}. How many at first?"
                    # Answer: initial amount = num_total - num2 = result
                    actual_answer = result
                    num2_value = num_total - result  # This is the amount they got
                    prompt = template.format(char1=char1, char2=char2, num1=fmt_num(result), num2=fmt_num(num2_value), num_total=fmt_num(num_total), obj_plural=obj_plural)
                    answer = f' {fmt_num(actual_answer)}'
                else:
                    # Template A: "After giving some, had {num1} left. How many did give?"
                    # Answer: amount given = num_total - result = num2
                    actual_answer = num_total - result  # This is num2
                    prompt = template.format(char1=char1, char2=char2, num1=fmt_num(result), num2=fmt_num(actual_answer), num_total=fmt_num(num_total), obj_plural=obj_plural)
                    answer = f' {fmt_num(actual_answer)}'
            else:
                # Standard subtraction
                prompt = template.format(char1=char1, char2=char2, num1=fmt_num(num1), num2=fmt_num(num2), num_total=fmt_num(num_total), obj_plural=obj_plural)
                answer = f' {fmt_num(result)}'
                
        else:
            # Three-term mixed operations
            if operators[0] == '+' and operators[1] == '-':
                num1, num2, num3 = fmt_num(operands[0]), fmt_num(operands[1]), fmt_num(operands[2])
                prompt = f"{char1} had {num1} {obj_plural}. {char2} gave {char1} {num2} more {obj_plural}. Then {char1} gave {num3} {obj_plural} away. How many {obj_plural} does {char1} have now?"
                answer = f' {fmt_num(result)}'
            elif operators[0] == '-' and operators[1] == '+':
                num1, num2, num3 = fmt_num(operands[0]), fmt_num(operands[1]), fmt_num(operands[2])
                prompt = f"{char1} had {num1} {obj_plural}. {char1} gave {num2} {obj_plural} to {char2}. Then {char1} found {num3} more {obj_plural}. How many {obj_plural} does {char1} have now?"
                answer = f' {fmt_num(result)}'
            elif operators[0] == '-' and operators[1] == '-':
                num1, num2, num3 = fmt_num(operands[0]), fmt_num(operands[1]), fmt_num(operands[2])
                prompt = f"{char1} had {num1} {obj_plural}. {char1} gave {num2} {obj_plural} to {char2} and {num3} {obj_plural} to another friend. How many {obj_plural} does {char1} have left?"
                answer = f' {fmt_num(result)}'
            else:
                # Fallback
                prompt = f"{char1} started with {fmt_num(operands[0])} {obj_plural}. After some transactions, {char1} ended up with how many {obj_plural}?"
                answer = f' {fmt_num(result)}'
        
        return prompt, answer


def split_dataset(problems: List[Dict], 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.1, 
                  test_ratio: float = 0.1) -> List[Dict]:
    """
    Split dataset into train/val/test while preserving statistics
    
    Strategy: Stratify by (n_terms, n_digits, has_carry) to preserve distributions
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Group problems by characteristics
    strata = defaultdict(list)
    for problem in problems:
        key = (problem['n_terms'], problem['n_digits'], problem['has_carry'])
        strata[key].append(problem)
    
    # Assign splits within each stratum
    for key, stratum_problems in strata.items():
        n = len(stratum_problems)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # n_test gets the remainder to ensure all problems are assigned
        
        # Shuffle within stratum
        random.shuffle(stratum_problems)
        
        # Assign splits
        for i, problem in enumerate(stratum_problems):
            if i < n_train:
                problem['split'] = 'train'
            elif i < n_train + n_val:
                problem['split'] = 'val'
            else:
                problem['split'] = 'test'
    
    return problems


def save_dataset_helm(problems: List[Dict], output_prefix: str, target_formats: List[str] = None):
    """
    Save dataset in HELM (Holistic Evaluation of Language Models) format

    HELM format uses JSONL (JSON Lines) where each line is a JSON object
    Creates 5 files: numeric, English, Spanish, Italian, embedded-in-context
    """
    formatter = DatasetFormatter()

    # Assign unique IDs
    for i, problem in enumerate(problems):
        problem['id'] = f"prob_{i:06d}"

    # Ensure target directory exists: data/HELM/
    base_dir = os.path.join('data', 'HELM')
    os.makedirs(base_dir, exist_ok=True)
    

    # Add 'HELM' to filenames to clearly indicate the format
    filenames = {
        'numeric': os.path.join(base_dir, f'{output_prefix}_HELM_numeric.jsonl'),
        'en': os.path.join(base_dir, f'{output_prefix}_HELM_english.jsonl'),
        'es': os.path.join(base_dir, f'{output_prefix}_HELM_spanish.jsonl'),
        'it': os.path.join(base_dir, f'{output_prefix}_HELM_italian.jsonl'),
        'embedded': os.path.join(base_dir, f'{output_prefix}_HELM_embedded.jsonl'),
        'embedded_verbal': os.path.join(base_dir, f'{output_prefix}_HELM_embedded_verbal.jsonl')
    }

    # Language names for metadata
    lang_names = {
        'numeric': 'numeric',
        'en': 'english',
        'es': 'spanish',
        'it': 'italian',
        'embedded': 'embedded-in-context'
    }

    # Create HELM instances for each format
    for format_key, filename in filenames.items():
        if target_formats and 'all' not in target_formats and format_key not in target_formats:
            continue

        with open(filename, 'w', encoding='utf-8') as f:
            for problem in problems:
                # Get prompt and answer based on format
                if format_key == 'numeric':
                    prompt, answer = formatter.format_numeric(problem)
                elif format_key == 'embedded':
                    prompt, answer = formatter.format_embedded_context(problem)
                elif format_key == 'embedded_verbal':
                    prompt, answer = formatter.format_embedded_context(problem, verbal=True)
                else:
                    prompt, answer = formatter.format_verbal(problem, language=format_key)

                # Create HELM instance with nested structure
                helm_instance = {
                    'input': {
                        'text': prompt
                    },
                    'references': [
                        {
                            'output': {
                                'text': answer
                            },
                            'tags': ['correct']
                        }
                    ],
                    'split': problem['split'],
                    'id': problem['id']
                }

                # Write as JSON line
                f.write(json.dumps(helm_instance, ensure_ascii=False) + '\n')

        print(f"Saved {filename} ({len(problems)} problems)")

    # Print statistics
    _print_dataset_statistics(problems)


def save_dataset(problems: List[Dict], output_prefix: str, target_formats: List[str] = None):
    """
    Save dataset in 4 formats: numeric, English, Spanish, Italian
    
    Each file contains the same problems with matched IDs
    """
    formatter = DatasetFormatter()
    
    # Assign unique IDs
    for i, problem in enumerate(problems):
        problem['id'] = f"prob_{i:06d}"
    
    # Create datasets for each format
    datasets = {
        'numeric': [],
        'en': [],
        'es': [],
        'it': [],
        'embedded': [],
        'embedded_verbal': []
    }
    
    for problem in problems:
        # Numeric format
        if not target_formats or 'all' in target_formats or 'numeric' in target_formats:
            numeric_prompt, numeric_answer = formatter.format_numeric(problem)
            datasets['numeric'].append({
                'id': problem['id'],
                'prompt': numeric_prompt,
                'answer': numeric_answer,
                'ground_truth': problem['result'],
                'split': problem['split'],
                'has_carry': problem['has_carry'],
                'n_terms': problem['n_terms'],
                'n_digits': problem['n_digits']
            })
        
        # Verbal formats (English, Spanish, Italian)
        for lang in ['en', 'es', 'it']:
            if target_formats and 'all' not in target_formats and lang not in target_formats:
                continue
            verbal_prompt, verbal_answer = formatter.format_verbal(problem, language=lang)
            datasets[lang].append({
                'id': problem['id'],
                'prompt': verbal_prompt,
                'answer': verbal_answer,
                'ground_truth': problem['result'],
                'split': problem['split'],
                'has_carry': problem['has_carry'],
                'n_terms': problem['n_terms'],
                'n_digits': problem['n_digits']
            })
        
        # Embedded-in-context format
        if not target_formats or 'all' in target_formats or 'embedded' in target_formats:
            embedded_prompt, embedded_answer = formatter.format_embedded_context(problem)
            datasets['embedded'].append({
                'id': problem['id'],
                'prompt': embedded_prompt,
                'answer': embedded_answer,
                'ground_truth': problem['result'],
                'split': problem['split'],
                'has_carry': problem['has_carry'],
                'n_terms': problem['n_terms'],
                'n_digits': problem['n_digits']
            })

        # Embedded-in-context verbal format
        if not target_formats or 'all' in target_formats or 'embedded_verbal' in target_formats:
            embedded_verbal_prompt, embedded_verbal_answer = formatter.format_embedded_context(problem, verbal=True)
            datasets['embedded_verbal'].append({
                'id': problem['id'],
                'prompt': embedded_verbal_prompt,
                'answer': embedded_verbal_answer,
                'ground_truth': problem['result'],
                'split': problem['split'],
                'has_carry': problem['has_carry'],
                'n_terms': problem['n_terms'],
                'n_digits': problem['n_digits']
            })
    
    # Ensure target directory exists: data/json/
    base_dir = os.path.join('data', 'json')
    os.makedirs(base_dir, exist_ok=True)
    
    # Save each dataset
    filenames = {
        'numeric': os.path.join(base_dir, f'{output_prefix}_numeric.json'),
        'en': os.path.join(base_dir, f'{output_prefix}_english.json'),
        'es': os.path.join(base_dir, f'{output_prefix}_spanish.json'),
        'it': os.path.join(base_dir, f'{output_prefix}_italian.json'),
        'embedded': os.path.join(base_dir, f'{output_prefix}_embedded.json'),
        'embedded_verbal': os.path.join(base_dir, f'{output_prefix}_embedded_verbal.json')
    }
    
    for format_name, dataset in datasets.items():
        if target_formats and 'all' not in target_formats and format_name not in target_formats:
            continue
        filename = filenames[format_name]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved {filename} ({len(dataset)} problems)")
    
    # Print statistics
    _print_dataset_statistics(problems)


def _print_dataset_statistics(problems: List[Dict]):
    """Print comprehensive dataset statistics"""
    print("\n=== Dataset Statistics ===")
    print(f"Total problems: {len(problems)}")
    
    # Count by split
    split_counts = defaultdict(int)
    for p in problems:
        split_counts[p['split']] += 1
    print(f"Train: {split_counts['train']} ({split_counts['train']/len(problems)*100:.1f}%)")
    print(f"Val: {split_counts['val']} ({split_counts['val']/len(problems)*100:.1f}%)")
    print(f"Test: {split_counts['test']} ({split_counts['test']/len(problems)*100:.1f}%)")
    
    # Count by characteristics
    terms_counts = defaultdict(int)
    digits_counts = defaultdict(int)
    carry_counts = defaultdict(int)
    
    for p in problems:
        terms_counts[p['n_terms']] += 1
        digits_counts[p['n_digits']] += 1
        carry_counts[p['has_carry']] += 1
    
    print(f"\n2-term problems: {terms_counts[2]} ({terms_counts[2]/len(problems)*100:.1f}%)")
    print(f"3-term problems: {terms_counts[3]} ({terms_counts[3]/len(problems)*100:.1f}%)")
    print(f"2-digit problems: {digits_counts[2]} ({digits_counts[2]/len(problems)*100:.1f}%)")
    print(f"3-digit problems: {digits_counts[3]} ({digits_counts[3]/len(problems)*100:.1f}%)")
    print(f"Problems with carry/borrow: {carry_counts[True]} ({carry_counts[True]/len(problems)*100:.1f}%)")
    print(f"Problems without carry/borrow: {carry_counts[False]} ({carry_counts[False]/len(problems)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate arithmetic problem datasets with flexible constraints',
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
                        help='Proportion of problems with carry/borrow operations (applies to both addition and subtraction)')
    
    # Optional constraints
    parser.add_argument('--avoid-repeated-digits', action='store_true',
                        help='Avoid numbers with repeated digits (e.g., 121, 77, 88)')
    parser.add_argument('--avoid-clean-multiples', action='store_true',
                        help='Avoid clean multiples of 10 (e.g., 70, 80, 100)')
    parser.add_argument('--avoid-reverse-pairs', action='store_true',
                        help='Avoid reverse pairs for 2-term addition (if 34+21, skip 21+34)')
    
    # Output
    parser.add_argument('-o', '--output-prefix', type=str, default='arith_dataset',
                        help='Output filename prefix')
    parser.add_argument('--output-format', type=str, default='json', 
                        choices=['json', 'helm'],
                        help='Output format: json (standard JSON arrays) or helm (JSONL format)')
    
    # New argument to select specific formats
    parser.add_argument('--formats', type=str, nargs='+', default=['all'],
                        choices=['all', 'numeric', 'en', 'es', 'it', 'embedded', 'embedded_verbal'],
                        help='Specific problem formats to generate (default: all)')
    
    # Other
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate probability distributions
    if abs(sum(args.terms) - 1.0) > 1e-6:
        parser.error("--terms probabilities must sum to 1.0")
    if abs(sum(args.digits) - 1.0) > 1e-6:
        parser.error("--digits probabilities must sum to 1.0")
    
    print("=== Arithmetic Problem Dataset Generator ===\n")
    print(f"Generating {args.num_problems} problems with the following parameters:")
    print(f"  Terms distribution: 2-term={args.terms[0]:.1%}, 3-term={args.terms[1]:.1%}")
    print(f"  Digits distribution: 2-digit={args.digits[0]:.1%}, 3-digit={args.digits[1]:.1%}")
    print(f"  Solution digits filter: {args.solution_digits if args.solution_digits else 'None'}")
    print(f"  Carry/borrow percentage: {args.carry_percentage:.1%}")
    print(f"  Avoid repeated digits: {args.avoid_repeated_digits}")
    print(f"  Avoid clean multiples: {args.avoid_clean_multiples}")
    print(f"  Avoid reverse pairs: {args.avoid_reverse_pairs}")
    print(f"  Output format: {args.output_format.upper()}")
    print(f"  Target formats: {', '.join(args.formats)}")
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
    
    # Save datasets in the appropriate format
    print("\nSaving datasets...")
    if args.output_format == 'helm':
        save_dataset_helm(problems, args.output_prefix, args.formats)
    else:
        save_dataset(problems, args.output_prefix, args.formats)
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()

