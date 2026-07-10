import asyncio

_tasks: dict[str, asyncio.Task] = {}


def register_task(job_id: str, task: asyncio.Task) -> None:
    _tasks[job_id] = task


def unregister_task(job_id: str) -> None:
    _tasks.pop(job_id, None)


def cancel_task(job_id: str) -> bool:
    task = _tasks.get(job_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


def cancel_all_tasks() -> int:
    cancelled = 0
    for task in list(_tasks.values()):
        if task and not task.done():
            task.cancel()
            cancelled += 1
    return cancelled
