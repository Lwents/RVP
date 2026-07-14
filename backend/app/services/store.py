from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.core import settings
from app.models.job import DubbingRequest, JobProgress, JobStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobProgress] = {}
        self._lock = Lock()
        self._index_file = Path(settings.storage_dir) / "jobs_index.json"
        self._load()

    def create(self, request: DubbingRequest) -> JobProgress:
        now = datetime.now(UTC)
        job = JobProgress(
            job_id=str(uuid4()),
            status=JobStatus.queued,
            progress=0,
            stage="Đã nhận cấu hình",
            request=request,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._save_unlocked()
        return job

    def get(self, job_id: str) -> JobProgress | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[JobProgress]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda item: item.completed_at or item.updated_at,
                reverse=True,
            )

    def clear(self) -> list[str]:
        with self._lock:
            warnings = self._trash_job_artifacts_unlocked()
            self._jobs.clear()
            self._save_unlocked()
            return warnings

    def update(self, job_id: str, **changes: object) -> JobProgress:
        with self._lock:
            job = self._jobs[job_id]
            data = job.model_dump()
            data.update(changes)
            data["updated_at"] = datetime.now(UTC)
            updated = JobProgress.model_validate(data)
            self._jobs[job_id] = updated
            self._save_unlocked()
            return updated

    def _load(self) -> None:
        if not self._index_file.exists():
            return
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            self._jobs = {
                item["job_id"]: JobProgress.model_validate(
                    {
                        **item,
                        "completed_at": item.get("completed_at") or (
                            item.get("updated_at") if item.get("status") == JobStatus.completed.value else None
                        ),
                    }
                )
                for item in data
                if isinstance(item, dict) and item.get("job_id")
            }
            self._mark_interrupted_jobs()
        except Exception:
            self._jobs = {}

    def _mark_interrupted_jobs(self) -> None:
        changed = False
        for job_id, job in list(self._jobs.items()):
            if job.status not in {JobStatus.queued, JobStatus.processing}:
                continue
            data = job.model_dump()
            data.update(
                {
                    "status": JobStatus.failed,
                    "stage": "Job bị gián đoạn khi backend khởi động lại",
                    "error": "Backend đã restart trước khi job hoàn tất. Hãy chạy lại job này.",
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = JobProgress.model_validate(data)
            changed = True
        if changed:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        self._index_file.parent.mkdir(parents=True, exist_ok=True)
        data = [job.model_dump(mode="json") for job in self._jobs.values()]
        self._index_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _trash_job_artifacts_unlocked(self) -> list[str]:
        paths = self._artifact_paths_unlocked()
        if not paths:
            return []
        try:
            from send2trash import send2trash
        except ImportError as exc:
            raise RuntimeError("Thiếu Send2Trash. Chạy pip install -r requirements.txt trong backend.") from exc

        warnings: list[str] = []
        for path in paths:
            try:
                send2trash(str(path))
            except Exception as exc:
                warnings.append(f"{path}: {exc}")
        return warnings[:10]

    def _artifact_paths_unlocked(self) -> list[Path]:
        storage_dir = Path(settings.storage_dir).resolve()
        jobs_dir = storage_dir / "jobs"
        paths: list[Path] = []

        if jobs_dir.exists():
            paths.extend(path.resolve() for path in jobs_dir.iterdir() if path.is_dir())

        for job in self._jobs.values():
            if not job.output_file_path:
                continue
            output_file = Path(job.output_file_path).resolve()
            if not output_file.exists():
                continue
            if _is_relative_to(output_file, jobs_dir):
                continue
            if _is_relative_to(output_file, storage_dir):
                paths.append(output_file)

        return _dedupe_existing_paths(paths)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _dedupe_existing_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.exists():
            continue
        if any(_is_relative_to(path, parent) for parent in unique if parent.is_dir()):
            continue
        unique.append(path)
        seen.add(path)
    return unique


job_store = JobStore()
