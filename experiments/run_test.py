#!/usr/bin/env python3
import os
import signal
import sys
import time
import subprocess
import asyncio
import pandas as pd
import yaml
from collections import defaultdict
from pathlib import Path
from multiprocessing import Process, Queue
import queue

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = str(SCRIPT_DIR.parent)
CONFIG_PATH = SCRIPT_DIR / "run_config.yaml"

sys.path.insert(0, PROJECT_DIR)

def load_run_config(config_path=None):
    """Load run_config.yaml; return (tasks, workers, model, preset, full_cfg, excel_file)."""
    path = Path(config_path or CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = (cfg.get("model") or "remote").strip().lower()
    if model not in ("local", "remote", "text"):
        model = "remote"
    preset = cfg.get(model, {})
    if not preset:
        raise ValueError(f"Missing '{model}' section in run_config.yaml")
    tasks = int(cfg.get("tasks", 5))
    workers = int(cfg.get("workers", 5))
    serial_per_url = bool(cfg.get("serial_per_url", False))
    excel_file = cfg.get("excel_file", "")
    true_label_column = cfg.get("true_label_column", "A_score")
    return tasks, workers, model, preset, cfg, excel_file, true_label_column, serial_per_url


def apply_llm_env(preset, for_local=False):
    """Set metagpt environment variables from preset['llm'] (must be called before importing metagpt)."""
    llm = preset.get("llm") or {}
    os.environ["llm__api_type"] = str(llm.get("api_type", "openai"))
    os.environ["llm__model"] = str(llm.get("model", ""))
    os.environ["llm__base_url"] = str(llm.get("base_url", ""))
    os.environ["llm__api_key"] = str(llm.get("api_key", ""))
    os.environ["llm__stream"] = str(llm.get("stream", "false"))
    if for_local:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""


def start_xvfb(display_num):
    xvfb = subprocess.Popen(
        ["Xvfb", f":{display_num}", "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return xvfb


def _start_dbus_and_atspi(display_num, worker_id):
    """Start D-Bus session bus and AT-SPI services for accessibility tree support.

    Returns (dbus_proc, atspi_launcher, atspi_registryd) or Nones on failure.
    """
    dbus_proc = atspi_launcher = atspi_registryd = None
    env = os.environ.copy()
    env["DISPLAY"] = f":{display_num}"

    try:
        result = subprocess.run(
            ["dbus-launch", "--sh-syntax"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().rstrip(";").strip("'\"")
                    os.environ[key] = val
            bus_pid = os.environ.get("DBUS_SESSION_BUS_PID")
            print(f"[Worker {worker_id}] D-Bus started (PID: {bus_pid})")
        else:
            print(f"[Worker {worker_id}] D-Bus launch failed: {result.stderr}")
            return None, None, None

        atspi_launcher_path = "/usr/libexec/at-spi-bus-launcher"
        if os.path.exists(atspi_launcher_path):
            atspi_launcher = subprocess.Popen(
                [atspi_launcher_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
            time.sleep(2)
            print(f"[Worker {worker_id}] AT-SPI bus launcher started (PID: {atspi_launcher.pid})")

        atspi_registryd_path = "/usr/libexec/at-spi2-registryd"
        if os.path.exists(atspi_registryd_path):
            atspi_registryd = subprocess.Popen(
                [atspi_registryd_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
            time.sleep(1)
            print(f"[Worker {worker_id}] AT-SPI registryd started (PID: {atspi_registryd.pid})")

    except Exception as e:
        print(f"[Worker {worker_id}] AT-SPI setup failed: {e}")

    return dbus_proc, atspi_launcher, atspi_registryd


def _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id):
    """Stop AT-SPI services."""
    for name, proc in [("registryd", atspi_registryd), ("bus-launcher", atspi_launcher)]:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    dbus_pid = os.environ.get("DBUS_SESSION_BUS_PID")
    if dbus_pid:
        try:
            os.kill(int(dbus_pid), signal.SIGTERM)
        except Exception:
            pass


SHARED_MODEL_CACHE = os.path.join(PROJECT_DIR, ".cache", "modelscope")

def worker_process(task_queue, result_queue, worker_id, preset, excel_file):
    """Worker process: run tasks using local or remote model based on preset."""
    import shutil

    agent_class = preset.get("agent_class", "osagent")
    is_text_agent = agent_class == "text_agent"

    if is_text_agent:
        print(f"[Worker {worker_id}] TextAgent mode: skipping OCR model and GPU allocation")
    else:
        worker_cache = f"/tmp/modelscope_worker_{worker_id}"
        os.environ["MODELSCOPE_CACHE"] = worker_cache
        if os.path.exists(SHARED_MODEL_CACHE) and not os.path.exists(os.path.join(worker_cache, "hub")):
            shutil.copytree(SHARED_MODEL_CACHE, worker_cache, dirs_exist_ok=True)
            print(f"[Worker {worker_id}] Copied shared model cache -> {worker_cache}")
        os.makedirs(worker_cache, exist_ok=True)

        # Round-robin GPU assignment
        cuda_devices = preset.get("cuda_devices", [0])
        assigned_gpu = cuda_devices[worker_id % len(cuda_devices)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        print(f"[Worker {worker_id}] Assigned GPU {assigned_gpu}")

    apply_llm_env(preset, for_local=(preset.get("config_file", "").find("local") >= 0))

    base_display = int(preset.get("base_display", 300))
    base_chrome_port = int(preset.get("base_chrome_port", 9500))
    config_file = preset.get("config_file", "configs/config.yaml")
    log_dir_prefix = preset.get("log_dir_prefix", "test2")

    display_num = base_display + worker_id
    port = base_chrome_port + worker_id
    user_data_dir = f"/tmp/chrome_test2_{log_dir_prefix}_{worker_id}"

    if os.path.exists(user_data_dir):
        shutil.rmtree(user_data_dir, ignore_errors=True)
    os.makedirs(user_data_dir, exist_ok=True)

    time.sleep(worker_id * 0.3)
    xvfb = start_xvfb(display_num)
    os.environ["DISPLAY"] = f":{display_num}"

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # AT-SPI: text_agent (cdp mode) does not need D-Bus/AT-SPI; osagent (atspi mode) does
    dbus_proc = atspi_launcher = atspi_registryd = None
    if not is_text_agent or preset.get("a11y_mode", "cdp") == "atspi":
        os.environ["GTK_MODULES"] = "gail:atk-bridge"
        os.environ["GNOME_ACCESSIBILITY"] = "1"
        os.environ["NO_AT_BRIDGE"] = "0"
        dbus_proc, atspi_launcher, atspi_registryd = _start_dbus_and_atspi(display_num, worker_id)
    else:
        print(f"[Worker {worker_id}] TextAgent (cdp mode): skipping D-Bus/AT-SPI startup")

    os.chdir(PROJECT_DIR)
    from appeval.roles.eval_runner import AppEvalRole

    print(f"[Worker {worker_id}] Config: {config_file}")

    tasks_since_recycle = 0
    RECYCLE_EVERY = 20

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break
        if item is None:
            break
        tasks_to_do = item if isinstance(item, list) else [item]

        for task in tasks_to_do:
            idx = task["idx"]
            case_name = task["case_name"]
            url = task["url"]
            test_point = task["test_point"]

            print(f"[Worker {worker_id}] #{idx}: {case_name}")
            print(f"[Worker {worker_id}] a11y_mode: {preset.get('a11y_mode', 'atspi')}, preset: {preset}")
            try:
                test_cases = {
                    "0": {
                        "case_desc": test_point,
                        "result": "",
                        "evidence": "",
                    }
                }
                agent_class = preset.get("agent_class", "osagent")
                appeval = AppEvalRole(
                    config_file=config_file,
                    remote_debugging_port=port,
                    user_data_dir=user_data_dir,
                    use_chrome_debugger=False,
                    a11y_mode=preset.get("a11y_mode", "atspi"),
                    max_iters=preset.get("max_iters", 15),
                    use_ocr=preset.get("use_ocr", agent_class != "text_agent"),
                    post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                    agent_class=agent_class,
                    debug_screenshots=preset.get("debug_screenshots", True),
                )
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                t0 = time.perf_counter()
                result, _ = loop.run_until_complete(
                    appeval.run_api(
                        task_name=f"{case_name}_{idx}",
                        test_cases=test_cases,
                        start_func=url,
                        log_dir=f"{log_dir_prefix}/{case_name}_{idx}",
                    )
                )
                elapsed_sec = time.perf_counter() - t0
                loop.close()

                prompt_tokens = completion_tokens = 0
                try:
                    if hasattr(appeval.test_generator, "llm") and hasattr(appeval.test_generator.llm, "get_costs"):
                        c = appeval.test_generator.llm.get_costs()
                        prompt_tokens += getattr(c, "total_prompt_tokens", 0) or 0
                        completion_tokens += getattr(c, "total_completion_tokens", 0) or 0
                    if hasattr(appeval, "osagent") and appeval.osagent and hasattr(appeval.osagent, "llm") and hasattr(appeval.osagent.llm, "get_costs"):
                        c = appeval.osagent.llm.get_costs()
                        prompt_tokens += getattr(c, "total_prompt_tokens", 0) or 0
                        completion_tokens += getattr(c, "total_completion_tokens", 0) or 0
                except Exception:
                    pass

                score = 0
                evidence = ""
                if result and "0" in result:
                    res = result["0"]
                    if isinstance(res, dict):
                        result_value = res.get("result", "Fail")
                        evidence = res.get("evidence", "")
                    else:
                        result_value = str(res)
                        evidence = str(res)
                    score = 1 if result_value.lower().strip() in ("pass", "true", "1") else 0
                else:
                    evidence = "No result returned"

                # Black screen detection: if evidence contains black-screen keywords and Fail, retry once
                black_kws = ["black screen", "blank screen", "black page", "blank page"]
                if score == 0 and any(k in evidence.lower() for k in black_kws):
                    print(f"[Worker {worker_id}] #{idx} Black screen detected! Restarting Chrome for retry...")
                    try:
                        from appeval.utils.window_utils import kill_windows
                        loop2 = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop2)
                        loop2.run_until_complete(kill_windows(user_data_dir=user_data_dir))
                        loop2.close()
                        time.sleep(2)
                        import shutil as _shutil
                        if os.path.exists(user_data_dir):
                            _shutil.rmtree(user_data_dir, ignore_errors=True)
                            os.makedirs(user_data_dir, exist_ok=True)
                        appeval2 = AppEvalRole(
                            config_file=config_file,
                            remote_debugging_port=port,
                            user_data_dir=user_data_dir,
                            use_chrome_debugger=False,
                            a11y_mode=preset.get("a11y_mode", "atspi"),
                            max_iters=preset.get("max_iters", 15),
                            use_ocr=preset.get("use_ocr", agent_class != "text_agent"),
                            post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                            agent_class=agent_class,
                            debug_screenshots=preset.get("debug_screenshots", True),
                        )
                        loop3 = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop3)
                        t1 = time.perf_counter()
                        result2, _ = loop3.run_until_complete(
                            appeval2.run_api(
                                task_name=f"{case_name}_{idx}",
                                test_cases=test_cases,
                                start_func=url,
                                log_dir=f"{log_dir_prefix}/{case_name}_{idx}",
                            )
                        )
                        elapsed_sec += time.perf_counter() - t1
                        loop3.close()
                        if result2 and "0" in result2:
                            res2 = result2["0"]
                            evidence = res2.get("evidence", evidence)
                            score = 1 if res2.get("result", "Fail").lower() == "pass" else 0
                            print(f"[Worker {worker_id}] #{idx} Retry result: {score}")
                    except Exception as retry_err:
                        print(f"[Worker {worker_id}] #{idx} Retry failed: {retry_err}")

                print(f"[Worker {worker_id}] Done #{idx}: {case_name} -> {score} ({elapsed_sec:.0f}s, {prompt_tokens + completion_tokens} tokens)")
                result_queue.put({
                    "idx": idx, "case_name": case_name, "score": score, "evidence": evidence,
                    "elapsed_sec": elapsed_sec, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                })
            except Exception as e:
                print(f"[Worker {worker_id}] Error #{idx}: {e}")
                result_queue.put({
                    "idx": idx, "case_name": case_name, "score": 0, "evidence": f"Error: {str(e)}",
                    "elapsed_sec": 0, "prompt_tokens": 0, "completion_tokens": 0,
                })

            # Periodically recycle Chrome to prevent memory leaks
            tasks_since_recycle += 1
            if tasks_since_recycle >= RECYCLE_EVERY:
                try:
                    import subprocess as _sp
                    _sp.run(f"pkill -f 'user-data-dir={user_data_dir}'", shell=True, capture_output=True)
                    time.sleep(1)
                    print(f"[Worker {worker_id}] Chrome recycled (every {RECYCLE_EVERY} tasks)")
                    tasks_since_recycle = 0
                except Exception:
                    pass

    _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id)
    xvfb.terminate()
    xvfb.wait()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DiagEval evaluation runner (see run_config.yaml)")
    parser.add_argument("--config", type=str, default=None, help="Path to run_config.yaml")
    parser.add_argument("--tasks", type=int, default=None, help="Override number of tasks")
    parser.add_argument("--workers", type=int, default=None, help="Override number of parallel workers")
    parser.add_argument("--model", type=str, choices=["local", "remote", "text"], default=None, help="Override model mode: local | remote | text")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume: skip tasks with existing scores (default: enabled)")
    parser.add_argument("--rerun-failed", action="store_true", default=False,
                        help="Only rerun incorrectly judged cases (requires result file and ground-truth column)")
    parser.add_argument("--no-cleanup", action="store_true", default=True, help="Skip stale process cleanup on startup")

    args = parser.parse_args()

    if not args.no_cleanup:
        cleanup_script = SCRIPT_DIR / "cleanup.sh"
        if cleanup_script.exists():
            print("Cleaning up stale processes before run...")
            subprocess.run(["bash", str(cleanup_script)], timeout=30)
        else:
            print(f"Cleanup script not found: {cleanup_script}, skipping")

    try:
        subprocess.run(
            ["sysctl", "-w", "fs.inotify.max_user_instances=8192"],
            capture_output=True, timeout=5
        )
        print("inotify.max_user_instances set to 8192")
    except Exception:
        print("Cannot set inotify limit (requires root); Chrome may fail with many workers")

    config_path = args.config or str(CONFIG_PATH)
    tasks, workers, model, preset, full_cfg, excel_file, true_label_column, serial_per_url = load_run_config(config_path)

    if args.tasks is not None:
        tasks = args.tasks
    if args.workers is not None:
        workers = args.workers
    if args.model is not None:
        model = args.model
        preset = full_cfg.get(model, preset)

    preset["cuda_devices"] = full_cfg.get("cuda_devices", [0])

    apply_llm_env(preset, for_local=(model == "local"))

    excel_path = SCRIPT_DIR / (excel_file or full_cfg.get("excel_file", "RealDevBench_MGX_20260130.xlsx"))
    if not excel_path.exists():
        raise FileNotFoundError(f"Data file not found: {excel_path}")

    result_excel = preset.get("result_excel", "test2_results.xlsx")
    score_col = preset.get("score_column", "os_agent_score")
    evidence_col = preset.get("evidence_column", "evidence")
    out_path = SCRIPT_DIR / result_excel

    def _get_url_col(dataframe):
        if "prod_id" in dataframe.columns:
            return "prod_id"
        elif "prod_url" in dataframe.columns:
            return "prod_url"
        else:
            raise ValueError(f"No URL column found (need prod_id or prod_url), available: {list(dataframe.columns)}")

    if args.rerun_failed and out_path.exists():
        df = pd.read_excel(out_path)
        url_col = _get_url_col(df)
        valid_df = df[
            df[url_col].notna() & df[true_label_column].notna() & df["test_point"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)
        if score_col in valid_df.columns:
            has_both = valid_df[valid_df[score_col].notna() & valid_df[true_label_column].notna()]
            wrong_mask = has_both[score_col].astype(int) != has_both[true_label_column].astype(int)
            valid_df = has_both[wrong_mask].copy()
            valid_df[score_col] = None
            valid_df[evidence_col] = None
            for idx in valid_df.index:
                df.at[idx, score_col] = None
                if evidence_col in df.columns:
                    df.at[idx, evidence_col] = None
            df.to_excel(out_path, index=False)
        else:
            valid_df = valid_df.head(0)
        print(f"[Rerun failed] Result file: {out_path}, incorrect cases: {len(valid_df)}")
    elif args.resume and out_path.exists():
        df = pd.read_excel(out_path)
        url_col = _get_url_col(df)
        valid_df = df[
            df[url_col].notna() & df[true_label_column].notna() & df["test_point"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)
        if score_col in valid_df.columns:
            valid_df = valid_df[valid_df[score_col].isna()]
        else:
            valid_df = valid_df.head(0)
        print(f"[Resume] Result file exists: {out_path}, pending: {len(valid_df)}")
    else:
        df = pd.read_excel(excel_path)
        url_col = _get_url_col(df)
        valid_df = df[
            df[url_col].notna() & df[true_label_column].notna() & df["test_point"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)

    agent_class = preset.get("agent_class", "osagent")
    print(f"Model: {model} | Agent: {agent_class} | Tasks: {tasks} | Workers: {workers} | URL col: {url_col}")
    print(f"Config: {preset.get('config_file')} | Results: {result_excel}")
    if agent_class == "text_agent":
        print(f"TextAgent mode: text-only a11y tree, a11y_mode={preset.get('a11y_mode', 'cdp')}, no OCR/GPU")

    if len(valid_df) == 0:
        print("No valid tasks (or all completed in resume mode), exiting")
        return

    task_list = [
        {
            "idx": idx,
            "case_name": row["case_name"],
            "url": row[url_col],
            "test_point": row["test_point"],
        }
        for idx, row in valid_df.iterrows()
    ]

    task_queue = Queue()
    result_queue = Queue()
    if serial_per_url:
        by_url = defaultdict(list)
        for t in task_list:
            by_url[t["url"]].append(t)
        for _url, group in by_url.items():
            task_queue.put(group)
        print(f"Serial per URL, parallel across URLs: {len(by_url)} URLs, {len(task_list)} tasks")
    else:
        for t in task_list:
            task_queue.put(t)
    for _ in range(workers):
        task_queue.put(None)

    total_start = time.perf_counter()
    processes = []
    for i in range(workers):
        p = Process(
            target=worker_process,
            args=(task_queue, result_queue, i, preset, str(excel_path)),
        )
        p.start()
        processes.append(p)

    results = {}
    completed = 0
    last_save_time = time.perf_counter()
    last_result_time = time.perf_counter()
    save_interval = 120
    stall_timeout = 1200

    while completed < len(task_list):
        try:
            r = result_queue.get(timeout=30)
            results[r["idx"]] = r
            completed += 1
            last_result_time = time.perf_counter()
            print(f"Progress: {completed}/{len(task_list)} ({100 * completed / len(task_list):.1f}%)")

            now = time.perf_counter()
            if now - last_save_time > save_interval:
                for col in (score_col, evidence_col):
                    if col not in df.columns:
                        df[col] = None
                    df[col] = df[col].astype(object)
                for idx, res in results.items():
                    df.at[idx, score_col] = res["score"]
                    df.at[idx, evidence_col] = str(res["evidence"]) if res["evidence"] else ""
                df.to_excel(out_path, index=False)
                print(f"  Incremental save ({completed} rows)")
                last_save_time = now

        except queue.Empty:
            alive = [p for p in processes if p.is_alive()]
            if not alive:
                print(f"All workers exited, completed {completed}/{len(task_list)}")
                break
            stall_sec = time.perf_counter() - last_result_time
            if stall_sec > stall_timeout:
                print(f"No new results for {stall_sec/60:.0f} min, {len(alive)} workers may be stuck, force exit")
                break
            continue
        except Exception as e:
            print(f"Result collection error: {e}")
            break

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            print(f"  Force terminating worker PID {p.pid}")
            p.terminate()
            p.join(timeout=5)

    total_elapsed = time.perf_counter() - total_start
    total_prompt = sum(r.get("prompt_tokens", 0) for r in results.values())
    total_completion = sum(r.get("completion_tokens", 0) for r in results.values())
    total_tokens = total_prompt + total_completion

    for col in (score_col, evidence_col):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)
    for idx, res in results.items():
        df.at[idx, score_col] = res["score"]
        df.at[idx, evidence_col] = str(res["evidence"]) if res["evidence"] else ""
        true_label = df.at[idx, true_label_column]
        if pd.notna(true_label):
            df.at[idx, "match"] = 1 if res["score"] == int(true_label) else 0

    df.to_excel(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    tested = df[df[score_col].notna()]
    if len(tested) > 0:
        valid = tested[tested[true_label_column].notna()]
        if len(valid) > 0:
            consistent = (valid[score_col] == valid[true_label_column]).sum()
            accuracy = consistent / len(valid) * 100
            n = len(task_list)
            avg_sec = (sum(r.get("elapsed_sec", 0) for r in results.values()) / n) if n else 0
            print(f"\n=== Summary ===")
            print(f"Tested: {len(valid)} | Accuracy: {accuracy:.2f}%")
            print(f"Total time: {total_elapsed:.1f}s | Avg per task: {avg_sec:.1f}s")
            print(f"Tokens: prompt {total_prompt} + completion {total_completion} = total {total_tokens}")


if __name__ == "__main__":
    main()
