import ast
import gc
import os
import re
import sqlite3
import time
from multiprocessing import cpu_count, Pool
from threading import Thread


def execute_in_process(code, assertions, return_dict):
    local_env = {}
    result_list = []
    try:
        exec(code, {}, local_env)  # Execute generated code
        for idx, assertion in enumerate(assertions):
            try:
                exec(assertion, {}, local_env)
                result_list.append((idx, assertion, "pass", True))
            except Exception as e:
                result_list.append((idx, assertion, "fail", False))
    except Exception as e:
        for idx, assertion in enumerate(assertions):
            result_list.append((idx, assertion, "fail", False))
    finally:
        return_dict['result'] = result_list


def execute_with_timeout(code, assertions, timeout=5):
    result_container = {}

    def target():
        execute_in_process(code, assertions, result_container)

    thread = Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return [(idx, assertion, "fail", False, "timeout") for idx, assertion in enumerate(assertions)]
    return result_container.get('result', [])


def evaluate_code(code, assertions):
    if isinstance(assertions, str):
        try:
            asserts = ast.literal_eval(assertions)
            if not isinstance(asserts, list):
                raise ValueError("Parsed assertions is not a list.")
        except Exception as e:
            raise ValueError(f"Failed to parse assertions string: {e}")
    elif isinstance(assertions, list):
        asserts = assertions
    else:
        raise TypeError("assertions must be a list or a stringified list")

    try:
        updated_code = replace_function_name_with_asserts(code, asserts)
    except Exception as e:
        # Raise error clearly so we can catch it in `process_row`
        raise ValueError(str(e)) from e

    return execute_with_timeout(updated_code, asserts, timeout=5)


def is_code_complete(code: str) -> bool:
    code = code.strip()

    # Rule 1: Must not be empty or just imports
    if not code:
        return False
    if all(line.startswith("import") or line.startswith("from") for line in code.splitlines()):
        return False

    # Rule 2: Avoid placeholders or unfinished code
    if "..." in code or code.endswith("..."):
        return False

    # Rule 3: Syntax should be valid
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Rule 4: At least one executable body (not just import or docstring)
    has_exec = any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)) or
                   (isinstance(node, ast.Expr) and not isinstance(node.value, ast.Str))
                   for node in tree.body)

    return has_exec


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def replace_function_name_with_asserts(code: str, asserts) -> str:
    # Extract function name from code
    def_match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code)
    if not def_match:
        raise ValueError("No function definition found in the code.")

    old_name = def_match.group(1)

    # Extract desired function name from first assertion
    assert_str = asserts[0]
    assert_match = re.search(r'assert\s*\(?\s*([a-zA-Z_]\w*)\s*\(', assert_str)
    if not assert_match:
        raise ValueError(f"No function call found in assertion: {assert_str}")

    new_name = assert_match.group(1)

    # Replace only the function name in the code
    updated_code = re.sub(rf'\bdef\s+{old_name}\b', f'def {new_name}', code, count=1)
    return updated_code


def flush_batch(batch_results, output_cur, output_db, evaluation_counter):
    if batch_results:
        output_cur.executemany("""
            INSERT INTO EvaluationResults (
                task_id, model_name, problem, problem_type,
                generated_code, updated_code, generated_col_name,
                assertion_id, assertion, passed, error, comment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_results)
        output_db.commit()
        batch_results.clear()
        gc.collect()
        print(f"✅ Saved at iteration {evaluation_counter} with memory cleared.")


def process_row(args):
    row, col_index, i = args
    print(
        f"[PID {os.getpid()}] Processing task_id={row[col_index['task_id']]} model={row[col_index['model_name']]} code_{i}")
    try:
        task_id = row[col_index["task_id"]]
        model_name = row[col_index["model_name"]]
        raw_test_list = row[col_index["test_list"]]
        full_assertions = ast.literal_eval(raw_test_list)
        if not isinstance(full_assertions, list):
            raise ValueError("test_list is not a list of assertions")
    except Exception as e:
        # If test_list is invalid, skip this row
        return []

    code = row[col_index.get(f"generated_code_{i}", "")]

    # === Else Case: Handle Missing/Invalid Code ===
    if not code or not is_code_complete(code) or not is_syntax_valid(code):
        problem = row[col_index["problem"]] if "problem" in col_index else "MISSING"
        problem_type = row[col_index["problem_type"]] if "problem_type" in col_index else "MISSING"

        return [
            (
                task_id, model_name, problem, problem_type,
                code, "", f"generated_code_{i}",
                f"assertion_{idx}", assertion,
                "fail", "No Code generated", ""
            )
            for idx, assertion in enumerate(full_assertions)
        ]

    # === Try Evaluate Valid Code ===
    try:
        updated_code = replace_function_name_with_asserts(code, full_assertions)
        results = evaluate_code(code, full_assertions)

        output_records = []
        problem = row[col_index["problem"]] if "problem" in col_index else "MISSING"
        problem_type = row[col_index["problem_type"]] if "problem_type" in col_index else "MISSING"

        for idx, assertion, pass_fail, executed in results:
            error = "" if pass_fail == "pass" else "Failed assertion"
            comment_text = generate_comment(pass_fail, error, code)

            output_records.append((
                task_id, model_name, problem, problem_type,
                code, updated_code, f"generated_code_{i}",
                f"assertion_{idx}", assertion,
                pass_fail, error, comment_text
            ))
        return output_records

    # === Handle ValueErrors (e.g., no function name found in assert) ===
    except ValueError as e:
        problem = row[col_index["problem"]] if "problem" in col_index else "MISSING"
        problem_type = row[col_index["problem_type"]] if "problem_type" in col_index else "MISSING"
        error_msg = str(e)
        return [
            (
                task_id, model_name, problem, problem_type,
                code, "", f"generated_code_{i}",
                f"assertion_{idx}", assertion,
                "fail", error_msg, ""
            )
            for idx, assertion in enumerate(full_assertions)
        ]

    # === Catch-all for other exceptions ===
    except Exception as e:
        problem = row[col_index["problem"]] if "problem" in col_index else "MISSING"
        problem_type = row[col_index["problem_type"]] if "problem_type" in col_index else "MISSING"
        error_msg = str(e)
        return [
            (
                task_id, model_name, problem, problem_type,
                code, "", f"generated_code_{i}",
                f"assertion_{idx}", assertion,
                "fail", error_msg, ""
            )
            for idx, assertion in enumerate(full_assertions)
        ]


def generate_comment(pass_fail, error, code):
    if pass_fail == "pass":
        return "Test case passed"
    elif not code or code.strip() == "":
        return "No code submitted"
    elif not is_code_complete(code):
        return "Incomplete code detected"
    elif not is_syntax_valid(code):
        return "Syntax error in code"
    else:
        error_lower = error.lower() if error else ""
        if "no function call found" in error_lower:
            return "Function name mismatch"
        elif "timeout" in error_lower:
            return "Execution timeout"
        elif "assertionerror" in error_lower:
            return "Assertion failed"
        elif "nameerror" in error_lower:
            return "Name error - undefined variable or function"
        elif "typeerror" in error_lower:
            return "Type error - invalid operation or argument"
        else:
            return f"Runtime error: {error.strip().splitlines()[0]}"


# === Step 3: Evaluate each generated_code_# with test cases ===
evaluation_counter = 0
batch_results = []
start_time = time.time()
row_args = []

if __name__ == '__main__':
    # === Step 1: Read from merged.db1 ===
    source_db = sqlite3.connect("db/output-bytedance.db")
    source_cur = source_db.cursor()
    source_cur.execute("SELECT * FROM results;")
    rows = source_cur.fetchall()
    columns = [desc[0] for desc in source_cur.description]
    col_index = {name: idx for idx, name in enumerate(columns)}

    # === Step 2: Setup output DB ===
    output_db = sqlite3.connect("db/evaluation_results.db")
    output_cur = output_db.cursor()
    output_cur.execute("""
    CREATE TABLE IF NOT EXISTS EvaluationResults (
        task_id INTEGER,
        model_name TEXT,
        problem TEXT,
        problem_type TEXT,
        generated_code TEXT,
        updated_code TEXT,
        generated_col_name TEXT,
        assertion_id TEXT,
        assertion TEXT, 
        passed TEXT,
        error TEXT,
        comment TEXT
    );
    """)

    # Prepare arguments for multiprocessing
    # Prepare arguments for threading
    row_args = []
    # target_task_id = 11
    # target_model_name = "ByteDance-Seed/Seed-Coder-8B-Base"

    for row in rows:
        # task_id = row[col_index["task_id"]]
        # model_name = row[col_index["model_name"]]
        # if task_id == target_task_id and model_name == target_model_name:
        for i in range(1, 6):  # generated_code_1 to _5
            row_args.append((row, col_index, i))

    # Start threaded evaluation
    with Pool(cpu_count()) as pool:
        for i, result in enumerate(pool.imap(process_row, row_args), 1):
            batch_results.extend(result)
            evaluation_counter += 1
            if evaluation_counter % 100 == 0:
                flush_batch(batch_results, output_cur, output_db, evaluation_counter)

    # Final flush if anything remains
    if batch_results:
        flush_batch(batch_results, output_cur, output_db, evaluation_counter)

    # Close connections
    output_db.close()
    source_db.close()

