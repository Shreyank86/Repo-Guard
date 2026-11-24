from repoguard.verification import Claim, ClaimType, extract_claims


def test_extracts_the_class_and_the_claimed_file_together() -> None:
    text = "The `Job` class is defined in the file `job.py`. It manages scheduled jobs."

    assert extract_claims(text) == [Claim(symbol="Job", file="job.py")]


def test_supports_explicit_definitions_files_and_imports_in_order() -> None:
    text = (
        "File setup.py exists. The function `every` is declared in `schedule/__init__.py`. "
        "`schedule/__init__.py` imports `datetime`."
    )

    assert extract_claims(text) == [
        Claim(type=ClaimType.FILE_EXISTS, file="setup.py"),
        Claim(type=ClaimType.FUNCTION_EXISTS, symbol="every", file="schedule/__init__.py"),
        Claim(type=ClaimType.IMPORTS, file="schedule/__init__.py", module="datetime"),
    ]


def test_extracts_a_parameter_list_and_deduplicates_a_repeated_claim() -> None:
    text = (
        "The `Job.do` method accepts the following parameters:\n\n"
        "- `job_func`: The callback.\n"
        "- `**kwargs`: Keyword arguments.\n\n"
        "`Job.do` takes a `job_func` parameter.\n"
    )

    assert extract_claims(text) == [
        Claim(type=ClaimType.PARAMETER_EXISTS, symbol="Job.do", parameter="job_func"),
        Claim(type=ClaimType.PARAMETER_EXISTS, symbol="Job.do", parameter="kwargs"),
    ]


def test_preserves_call_direction_in_active_and_passive_sentences() -> None:
    expected = [Claim(type=ClaimType.CALLS, caller="Scheduler._run_job", callee="Job.run")]
    assert extract_claims("The method `Scheduler._run_job` calls `Job.run`.") == expected
    assert extract_claims("`Job.run` is called by `Scheduler._run_job`.") == expected
    assert extract_claims("The method that calls `Job.run` is `Scheduler._run_job`.") == expected


def test_does_not_promote_negations_questions_fences_or_copied_context() -> None:
    prose = (
        "The Job class is not defined in job.py.\n"
        "The Scheduler class is defined in scheduler.py?\n"
        "> The Job class is defined in job.py.\n"
        "```text\nThe Job class is defined in job.py.\n```\n"
        "Repository Context: The Job class is defined in job.py."
    )

    assert extract_claims(prose) == []


def test_extracts_markdown_bullet_and_compound_roles() -> None:
    text = (
        "- Function `every` is declared in `schedule/__init__.py`.\n"
        "* Class `Job` is defined in `schedule/__init__.py`.\n"
        "1. The function or class `run_pending` is defined in `schedule/__init__.py`.\n"
    )
    assert extract_claims(text) == [
        Claim(type=ClaimType.FUNCTION_EXISTS, symbol="every", file="schedule/__init__.py"),
        Claim(type=ClaimType.CLASS_EXISTS, symbol="Job", file="schedule/__init__.py"),
        Claim(type=ClaimType.SYMBOL_EXISTS, symbol="run_pending", file="schedule/__init__.py"),
    ]
