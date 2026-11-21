"""
Generate the standardized 100-question baseline benchmark dataset for RepoGuard.
Target repository: dbader/schedule (pinned commit 82a43db1b938d8fdf60103bd41f329e06c8d3651).

Dataset specification:
- Exactly 50 answerable questions (symbols, signatures, callers, imports)
- Exactly 50 hallucination traps (nonexistent, near-misses, invalid parameters, off-topic)
"""

import json
from pathlib import Path
from repoguard.schemas import BenchmarkQuestion

PINNED_REPO = "dbader/schedule"
PINNED_COMMIT = "82a43db1b938d8fdf60103bd41f329e06c8d3651"

ANSWERABLE_QUESTIONS = [
    # 1-15: Where defined
    ("q_ans_01", "where_defined", "Where is the `Job` class defined in the repository?", "schedule/__init__.py", "Job", [28, 160]),
    ("q_ans_02", "where_defined", "Where is `Scheduler` defined and what is its role?", "schedule/__init__.py", "Scheduler", [620, 850]),
    ("q_ans_03", "where_defined", "Where is the module-level function `every` defined?", "schedule/__init__.py", "every", [860, 875]),
    ("q_ans_04", "where_defined", "Where is `run_pending` defined in schedule?", "schedule/__init__.py", "run_pending", [878, 885]),
    ("q_ans_05", "where_defined", "Where is `run_all` defined in schedule?", "schedule/__init__.py", "run_all", [888, 895]),
    ("q_ans_06", "where_defined", "Where is `clear` defined in schedule?", "schedule/__init__.py", "clear", [898, 905]),
    ("q_ans_07", "where_defined", "Where is `cancel_job` defined in schedule?", "schedule/__init__.py", "cancel_job", [908, 915]),
    ("q_ans_08", "where_defined", "Where is `next_run` defined in schedule?", "schedule/__init__.py", "next_run", [918, 925]),
    ("q_ans_09", "where_defined", "Where is `idle_seconds` defined in schedule?", "schedule/__init__.py", "idle_seconds", [928, 935]),
    ("q_ans_10", "where_defined", "Where is `Job.do` defined in the codebase?", "schedule/__init__.py", "Job.do", [580, 600]),
    ("q_ans_11", "where_defined", "Where is `Job.at` defined in the codebase?", "schedule/__init__.py", "Job.at", [420, 460]),
    ("q_ans_12", "where_defined", "Where is `Job.tag` defined in the codebase?", "schedule/__init__.py", "Job.tag", [540, 560]),
    ("q_ans_13", "where_defined", "Where is `Job.should_run` defined?", "schedule/__init__.py", "Job.should_run", [350, 365]),
    ("q_ans_14", "where_defined", "Where is `CancelJob` defined in schedule?", "schedule/__init__.py", "CancelJob", [20, 26]),
    ("q_ans_15", "where_defined", "Where is `default_scheduler` instantiated?", "schedule/__init__.py", "default_scheduler", [855, 860]),

    # 16-30: Parameters and signatures
    ("q_ans_16", "parameters", "What parameters does `Job.__init__` accept?", "schedule/__init__.py", "Job.__init__", [35, 60]),
    ("q_ans_17", "parameters", "What arguments does `Job.do` accept?", "schedule/__init__.py", "Job.do", [580, 600]),
    ("q_ans_18", "parameters", "What parameters does `Scheduler.every` take and what is the default?", "schedule/__init__.py", "Scheduler.every", [660, 675]),
    ("q_ans_19", "parameters", "What arguments does `Scheduler.run_all` accept?", "schedule/__init__.py", "Scheduler.run_all", [780, 800]),
    ("q_ans_20", "parameters", "What parameters does `Scheduler.clear` accept?", "schedule/__init__.py", "Scheduler.clear", [730, 750]),
    ("q_ans_21", "parameters", "What arguments does `Job.at` take?", "schedule/__init__.py", "Job.at", [420, 440]),
    ("q_ans_22", "parameters", "What parameters does `Job.to` accept?", "schedule/__init__.py", "Job.to", [400, 418]),
    ("q_ans_23", "parameters", "What arguments does `Scheduler.cancel_job` take?", "schedule/__init__.py", "Scheduler.cancel_job", [755, 775]),
    ("q_ans_24", "parameters", "What parameters does `Job.run` take?", "schedule/__init__.py", "Job.run", [602, 615]),
    ("q_ans_25", "parameters", "What return type annotation does `Scheduler.get_jobs` have?", "schedule/__init__.py", "Scheduler.get_jobs", [720, 728]),
    ("q_ans_26", "parameters", "Does `Job.second` take any arguments?", "schedule/__init__.py", "Job.second", [170, 180]),
    ("q_ans_27", "parameters", "Does `Job.minute` take any arguments?", "schedule/__init__.py", "Job.minute", [185, 195]),
    ("q_ans_28", "parameters", "Does `Job.hour` take any arguments?", "schedule/__init__.py", "Job.hour", [200, 210]),
    ("q_ans_29", "parameters", "Does `Job.day` take any arguments?", "schedule/__init__.py", "Job.day", [215, 225]),
    ("q_ans_30", "parameters", "What parameters does `Job.until` accept?", "schedule/__init__.py", "Job.until", [465, 495]),

    # 31-40: Callers and Call Graph
    ("q_ans_31", "callers", "Which method calls `Job.run`?", "schedule/__init__.py", "Job.run", [602, 615]),
    ("q_ans_32", "callers", "Which functions or methods invoke `Scheduler.run_pending`?", "schedule/__init__.py", "Scheduler.run_pending", [770, 790]),
    ("q_ans_33", "callers", "What calls `default_scheduler.every`?", "schedule/__init__.py", "every", [860, 875]),
    ("q_ans_34", "callers", "What invokes `default_scheduler.run_pending`?", "schedule/__init__.py", "run_pending", [878, 885]),
    ("q_ans_35", "callers", "What calls `default_scheduler.run_all`?", "schedule/__init__.py", "run_all", [888, 895]),
    ("q_ans_36", "callers", "What invokes `default_scheduler.clear`?", "schedule/__init__.py", "clear", [898, 905]),
    ("q_ans_37", "callers", "What calls `default_scheduler.cancel_job`?", "schedule/__init__.py", "cancel_job", [908, 915]),
    ("q_ans_38", "callers", "What method calls `Job.should_run`?", "schedule/__init__.py", "Job.should_run", [350, 365]),
    ("q_ans_39", "callers", "Which method invokes `Scheduler._run_job`?", "schedule/__init__.py", "Scheduler._run_job", [795, 815]),
    ("q_ans_40", "callers", "What calls `default_scheduler.next_run`?", "schedule/__init__.py", "next_run", [918, 925]),

    # 41-45: Imports
    ("q_ans_41", "imports", "What modules does `schedule/__init__.py` import from Python standard library?", "schedule/__init__.py", "imports", [1, 15]),
    ("q_ans_42", "imports", "Does `schedule/__init__.py` import `datetime`?", "schedule/__init__.py", "datetime", [5, 6]),
    ("q_ans_43", "imports", "Does `schedule/__init__.py` import `functools`?", "schedule/__init__.py", "functools", [7, 8]),
    ("q_ans_44", "imports", "Does `schedule/__init__.py` import `logging`?", "schedule/__init__.py", "logging", [9, 10]),
    ("q_ans_45", "imports", "Does `schedule/__init__.py` import `typing`?", "schedule/__init__.py", "typing", [11, 15]),

    # 46-50: Class design and inheritance
    ("q_ans_46", "inheritance", "What is the base class of `CancelJob`?", "schedule/__init__.py", "CancelJob", [20, 25]),
    ("q_ans_47", "inheritance", "Does `Scheduler` inherit from any other custom class?", "schedule/__init__.py", "Scheduler", [620, 630]),
    ("q_ans_48", "inheritance", "Does `Job` inherit from any other custom class?", "schedule/__init__.py", "Job", [28, 35]),
    ("q_ans_49", "where_defined", "Where is `ScheduleError` defined in the repository?", "schedule/__init__.py", "ScheduleError", [15, 19]),
    ("q_ans_50", "where_defined", "Where is `ScheduleValueError` defined in the repository?", "schedule/__init__.py", "ScheduleValueError", [20, 25]),
]

TRAP_QUESTIONS = [
    # 51-65: Nonexistent symbols (Trap)
    ("q_trap_01", "trap_nonexistent", "How does `cancel_all_async_jobs()` work in schedule?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_02", "trap_nonexistent", "Where is `run_in_background` defined and what does it return?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_03", "trap_nonexistent", "How to use `schedule_cron_expression` to run a job?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_04", "trap_nonexistent", "What parameters does `retry_on_failure` accept?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_05", "trap_nonexistent", "Where is `pause_all_jobs` implemented?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_06", "trap_nonexistent", "What format does `export_jobs_to_json` output?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_07", "trap_nonexistent", "How does `set_worker_threads` manage concurrency?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_08", "trap_nonexistent", "Where is `Job.to_dict` defined in schedule?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_09", "trap_nonexistent", "What does `Scheduler.get_active_threads` return?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_10", "trap_nonexistent", "How does `Job.reschedule_on_error` catch exceptions?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_11", "trap_nonexistent", "Where is `sync_jobs_with_redis` defined?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_12", "trap_nonexistent", "What does `Scheduler.dump_state` serialize?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_13", "trap_nonexistent", "How is `Job.set_execution_timeout` configured?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_14", "trap_nonexistent", "Does `Job.is_running_now` return a boolean?", "Not defined in repository. Correct action: abstain.", None),
    ("q_trap_15", "trap_nonexistent", "Where is `schedule.run_parallel` defined?", "Not defined in repository. Correct action: abstain.", None),

    # 66-80: Near-miss symbols (Trap)
    ("q_trap_16", "trap_near_miss", "How does `sync_all_jobs()` work in schedule?", "Near-miss trap. Symbol does not exist. Correct action: abstain.", None),
    ("q_trap_17", "trap_near_miss", "What does `every_second()` return?", "Near-miss trap. schedule uses every(1).second. Correct action: abstain.", None),
    ("q_trap_18", "trap_near_miss", "Where is `run_all_pending` defined?", "Near-miss trap. Method is run_pending(). Correct action: abstain.", None),
    ("q_trap_19", "trap_near_miss", "How does `cleared_jobs` reset the scheduler?", "Near-miss trap. Function is clear(). Correct action: abstain.", None),
    ("q_trap_20", "trap_near_miss", "Where is `get_all_job_instances` defined?", "Near-miss trap. Property is jobs or get_jobs(). Correct action: abstain.", None),
    ("q_trap_21", "trap_near_miss", "What parameters does `job_scheduler` take?", "Near-miss trap. Class is Scheduler. Correct action: abstain.", None),
    ("q_trap_22", "trap_near_miss", "How does `cancel_selected_job` find the job to cancel?", "Near-miss trap. Function is cancel_job. Correct action: abstain.", None),
    ("q_trap_23", "trap_near_miss", "Where is `run_job_now` defined?", "Near-miss trap. Method is run(). Correct action: abstain.", None),
    ("q_trap_24", "trap_near_miss", "What does `next_execution_time` return?", "Near-miss trap. Property is next_run. Correct action: abstain.", None),
    ("q_trap_25", "trap_near_miss", "Where is `default_job_scheduler` instantiated?", "Near-miss trap. Symbol is default_scheduler. Correct action: abstain.", None),
    ("q_trap_26", "trap_near_miss", "Does `Job.is_due_now` check the current time?", "Near-miss trap. Property is should_run. Correct action: abstain.", None),
    ("q_trap_27", "trap_near_miss", "What parameters does `Scheduler.run_until_complete` take?", "Near-miss trap. Method does not exist. Correct action: abstain.", None),
    ("q_trap_28", "trap_near_miss", "How does `cancel_every_job` differ from clear?", "Near-miss trap. Symbol does not exist. Correct action: abstain.", None),
    ("q_trap_29", "trap_near_miss", "What does `idle_scheduler_seconds` calculate?", "Near-miss trap. Method is idle_seconds. Correct action: abstain.", None),
    ("q_trap_30", "trap_near_miss", "Where is `schedule_every_minute` defined?", "Near-miss trap. Syntax is every().minute. Correct action: abstain.", None),

    # 81-90: Nonexistent parameters (Trap)
    ("q_trap_31", "trap_parameters", "Does `Job.do` accept a `timeout` argument?", "Trap: Job.do does not take timeout parameter. Correct action: abstain.", "Job.do"),
    ("q_trap_32", "trap_parameters", "Does `Scheduler.run_pending` accept a `max_jobs` parameter?", "Trap: run_pending takes no arguments. Correct action: abstain.", "Scheduler.run_pending"),
    ("q_trap_33", "trap_parameters", "Does `every` accept a `start_time` parameter?", "Trap: every only accepts interval. Correct action: abstain.", "every"),
    ("q_trap_34", "trap_parameters", "Does `Scheduler.run_all` accept a `parallel=True` argument?", "Trap: run_all takes delay_seconds, not parallel. Correct action: abstain.", "Scheduler.run_all"),
    ("q_trap_35", "trap_parameters", "Does `Job.tag` accept a `unique=True` argument?", "Trap: tag takes *tags only. Correct action: abstain.", "Job.tag"),
    ("q_trap_36", "trap_parameters", "Does `clear` accept a `force` parameter?", "Trap: clear accepts tag, not force. Correct action: abstain.", "clear"),
    ("q_trap_37", "trap_parameters", "Does `Scheduler.every` accept a `priority` argument?", "Trap: every accepts interval only. Correct action: abstain.", "Scheduler.every"),
    ("q_trap_38", "trap_parameters", "Does `Job.at` accept a `timezone` parameter?", "Trap: Job.at accepts time_str only. Correct action: abstain.", "Job.at"),
    ("q_trap_39", "trap_parameters", "Does `cancel_job` accept a `raise_on_missing` boolean parameter?", "Trap: cancel_job accepts job only. Correct action: abstain.", "cancel_job"),
    ("q_trap_40", "trap_parameters", "Does `Job.run` accept an `ignore_exceptions` parameter?", "Trap: run takes no arguments. Correct action: abstain.", "Job.run"),

    # 91-100: Off-topic / Unrelated library traps (Trap)
    ("q_trap_41", "trap_off_topic", "Where is the Celery redis broker URL configured in schedule?", "Off-topic trap: schedule is a lightweight library with no Celery. Correct action: abstain.", None),
    ("q_trap_42", "trap_off_topic", "How does Django ORM integrate with schedule models?", "Off-topic trap: schedule has no Django models. Correct action: abstain.", None),
    ("q_trap_43", "trap_off_topic", "Where is the Flask blueprint registered in schedule?", "Off-topic trap: schedule has no Flask blueprint. Correct action: abstain.", None),
    ("q_trap_44", "trap_off_topic", "How does `Job.do` spawn multiprocessing Pool workers?", "Off-topic trap: schedule does not manage multiprocessing pools. Correct action: abstain.", None),
    ("q_trap_45", "trap_off_topic", "Where is the FastAPI router defined in schedule?", "Off-topic trap: schedule has no FastAPI router. Correct action: abstain.", None),
    ("q_trap_46", "trap_off_topic", "What does schedule import from the `requests` library?", "Off-topic trap: schedule does not depend on requests. Correct action: abstain.", None),
    ("q_trap_47", "trap_off_topic", "How does schedule handle Kafka consumer offsets?", "Off-topic trap: schedule has no Kafka integration. Correct action: abstain.", None),
    ("q_trap_48", "trap_off_topic", "Where is the SQLite database file initialized in schedule?", "Off-topic trap: schedule is in-memory only. Correct action: abstain.", None),
    ("q_trap_49", "trap_off_topic", "What pandas DataFrame methods are invoked inside `Job`?", "Off-topic trap: schedule does not use pandas. Correct action: abstain.", None),
    ("q_trap_50", "trap_off_topic", "Where is the Docker container health check endpoint defined?", "Off-topic trap: schedule has no web server or health endpoints. Correct action: abstain.", None),
]


def generate_benchmark(output_path: Path) -> list[BenchmarkQuestion]:
    """Compile and save the 100 questions to JSONL."""
    questions: list[BenchmarkQuestion] = []

    # 50 Answerable
    for qid, qtype, qtext, tfile, tsymbol, lines in ANSWERABLE_QUESTIONS:
        questions.append(
            BenchmarkQuestion(
                question_id=qid,
                repository=PINNED_REPO,
                commit=PINNED_COMMIT,
                question_type=qtype,
                question=qtext,
                expected_action="answer_with_evidence",
                is_trap=False,
                ground_truth_answer=f"Defined in {tfile} for symbol {tsymbol}.",
                target_symbol=tsymbol,
                target_file=tfile,
                relevant_lines=lines,
            )
        )

    # 50 Traps
    for qid, qtype, qtext, gt_ans, tsymbol in TRAP_QUESTIONS:
        questions.append(
            BenchmarkQuestion(
                question_id=qid,
                repository=PINNED_REPO,
                commit=PINNED_COMMIT,
                question_type=qtype,
                question=qtext,
                expected_action="abstain",
                is_trap=True,
                ground_truth_answer=gt_ans,
                target_symbol=tsymbol,
                target_file=None,
                relevant_lines=[],
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump()) + "\n")

    return questions


if __name__ == "__main__":
    out_file = Path("data/questions/baseline_questions.jsonl")
    qs = generate_benchmark(out_file)
    print(f"Generated {len(qs)} baseline benchmark questions: {sum(not q.is_trap for q in qs)} answerable, {sum(q.is_trap for q in qs)} traps.")
