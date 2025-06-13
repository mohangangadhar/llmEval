import ast
import os
import sqlite3

import pandas as pd
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import time
import threading

# Set the environment variable for Transformers cache location
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_models"

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print("Model loaded")


# Function to generate code from a prompt
# def generate_code(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         outputs = model.generate(inputs["input_ids"], max_length=256)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Function to validate code against assertion tests
def test_code(code_str, assertions):
    local_scope = {}
    try:
        print(code_str)
        exec(code_str, {}, local_scope)
        print(assertions)
        exec(assertions, {}, local_scope)
        return True, None
    except Exception as e:
        return False, traceback.format_exc()


def test_code_with_timeout(code, assertions, timeout=5):
    result_list = []
    stop_event = threading.Event()

    def run_code():
        local_env = {}
        try:
            exec(code, {}, local_env)  # Execute generated code
            for idx, assertion in enumerate(assertions):
                if stop_event.is_set():  # Check if stop signal is triggered
                    return
                start_time = time.time()
                try:
                    exec(assertion, {}, local_env)  # Execute assertion
                    result = "pass"  # If no error, assertion passed
                except Exception as e:
                    result = "fail"  # Assertion failed
                runtime = (time.time() - start_time) * 1e9
                result_list.append((idx, assertion, result, True, runtime))
        except Exception as e:
            for idx, assertion in enumerate(assertions):
                result_list.append((idx, assertion, "fail", False, None))

    thread = threading.Thread(target=run_code)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        stop_event.set()  # Signal the thread to stop
        return [(idx, assertion, "fail", False, "timeout") for idx, assertion in enumerate(assertions)]

    return result_list


def extract_python_code(text):
    lines = text.split('\n')
    code_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("import "):
            in_code_block = True
        if in_code_block:
            if stripped and (stripped.startswith("#") is False):
                code_lines.append(line)

    return '\n'.join(code_lines)


def clean_code(code_text):
    """Extract only the valid Python code from a text block."""
    code_markers = ['def ', 'class ', 'import ', 'from ', '@']
    lines = code_text.split('\n')

    # Find the first line that starts with a Python construct
    start_idx = None
    for idx, line in enumerate(lines):
        if any(line.strip().startswith(marker) for marker in code_markers):
            start_idx = idx
            break

    if start_idx is None:
        return ""  # No code found

    # Extract all indented or empty lines after code start
    code_lines = []
    for line in lines[start_idx:]:
        if line.strip() == "" or line.startswith(' ') or any(
                line.strip().startswith(marker) for marker in code_markers):
            code_lines.append(line)
        else:
            break  # Stop when non-code content resumes

    return '\n'.join(code_lines)


def clean_code_old(code_text):
    """Clean and validate generated code."""
    print("code text")
    print(code_text)
    # Remove any non-code text before the first Python construct
    code_markers = ['def ', 'class ', 'import ', 'from ', '@', '#']
    lines = code_text.split('\n')
    start_idx = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped.startswith(marker) for marker in code_markers):
            start_idx = idx
            break

    code_only = '\n'.join(lines[start_idx:])
    # print(code_only)
    # Remove any trailing non-code text
    code_lines = []
    in_code_block = False

    for line in code_only.split('\n'):
        stripped = line.strip()
        if any(stripped.startswith(marker) for marker in code_markers):
            in_code_block = True
        if in_code_block and (stripped or line.startswith('    ')):
            code_lines.append(line)

    return '\n'.join(code_lines)


def validate_code(code_text):
    """Validate Python code syntax."""
    try:
        ast.parse(code_text)
        return True
    except SyntaxError:
        return False


def generate_code(prompt, max_length=512, temperature=0.7, top_p=0.95, num_return_sequences=1):
    # Tokenize with attention to maximum length
    inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length).to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"]
            )

        generated_codes = []
        for output in outputs:
            uncleaned_code = tokenizer.decode(output, skip_special_tokens=True)
            cleaned_code = clean_code(uncleaned_code)
            cleaned_text = cleaned_code.replace("<|file_separator|>", "")
            generated_codes.append(cleaned_code)
            # if self.validate_code(cleaned_code):
            #     generated_codes.append(cleaned_code)
            #     print(cleaned_code)
        return generated_codes[0] if generated_codes else "# No valid Python code generated"

    except Exception as e:
        return f"# Generation error: {str(e)}"


def extract_and_clean_python_code(text):
    import re
    code_pattern = r"(def |import )[\s\S]+"
    match = re.search(code_pattern, text)
    if match:
        code = match.group()
        code_without_asserts = re.sub(r"^\s*assert .*$", "", code, flags=re.MULTILINE)
        code_without_comments = re.sub(r"#.*", "", code_without_asserts)
        cleaned_code_print = re.sub(r"^\s*print\(.+?\)\s*$", "", code_without_comments, flags=re.MULTILINE)
        cleaned_code = re.sub(r"\n\s*\n", "\n", cleaned_code_print)
        return cleaned_code.strip()
    return None


# Retry-based code generation and validation
def generate_valid_code(problem_description, assertions, test_case = False):
    # Formulate the prompt
    if test_case:
        prompt = f""" 
You are a Python code developer. Write a clean and correct Python function based on the following problem description. 

Problem:
{problem_description}

Follow these rules:
                # 1. Generate only one valid Python code with proper indentation
                # 2. Do not include any explanations or comments or examples
                # 3. Start with function definition or imports
                # 4. Ensure the code is complete and functional
                # 5. Do not add test cases or driver method
                # 6. Do not repeat the prompt given

Your function must pass the following assertions:
{assertions}
"""
    else:
        prompt = f"""
You are a Python code developer. Write a clean and correct Python function based on the following problem description. 

Problem:
{problem_description}

Follow these rules:
                # 1. Generate only one valid Python code with proper indentation
                # 2. Do not include any explanations or comments or examples
                # 3. Start with function definition or imports
                # 4. Ensure the code is complete and functional
                # 5. Do not add test cases or driver method
                # 6. Do not repeat the prompt given

"""

    # Generate code based on the prompt
    cleaned_code = generate_code(prompt)
    return cleaned_code
    # new_clean_code = extract_and_clean_python_code(cleaned_code)
    # print(new_clean_code)
    # Test the code
    # result = test_code_with_timeout(cleaned_code, assertions)
    # print(result)


conn = sqlite3.connect('output-phi-4-mini.db')
cursor = conn.cursor()

# Load dataset
df = pd.read_csv("mbpp_with_problem_type.csv")

# Add new columns to store results
df["generated_code"] = ""
df["status"] = ""
record_count = 0

# Process each row
for index, row in df.iterrows():
    print(f"\nProcessing Task ID {row['task_id']}")
    problem = row["text"]
    assertions = row["test_list"]
    result_data = {
        'task_id': row['task_id'],
        'model_name' : model_name,
        'problem': problem,
        'test_list': str(assertions)
    }

    try:
        for i in range(1,6):
            print(f"Problem {index}, Iteration {i}")
            result_data[f'generated_code_{i}'] = generate_valid_code(problem, assertions, False)
            # result = generate_valid_code(problem, assertions, False)
            # df.at[index, f"generated_code_{i}"] = result
        for i in range(1,6):
            print(f"Problem {index}, Iteration {i} - With Test Case")
            result_data[f'generated_code_test_case_{i}'] = generate_valid_code(problem, assertions, True)
            # result = generate_valid_code(problem, assertions, True)
            # df.at[index, f"generated_code_test_case_{i}"] = result

        placeholders = ', '.join(['?'] * len(result_data))
        columns = ', '.join(result_data.keys())
        values = list(result_data.values())

        cursor.execute(f'''
            INSERT OR REPLACE INTO generated_results ({columns}) VALUES ({placeholders})
        ''', values)

        record_count += 1
        if record_count % 100 == 0:
            conn.commit()
            print("Committed after 100 records")

        print("====================================")
        # df.at[index, "generated_code"] = code
        # df.at[index, "status"] = "success" if success else "failed"
    except Exception as e:
        print(e)
    #     df.at[index, "generated_code"] = str(e)
    #     df.at[index, "status"] = "error"

# Save output
# df.to_csv("mbpp_with_generated_code.csv", index=False)
conn.commit()
print("\nCode generation and validation complete. Output saved to 'mbpp_with_generated_code.csv'")
