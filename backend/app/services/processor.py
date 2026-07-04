from app.pipeline.dubbing import process_dubbing_job


async def process_job(job_id: str) -> None:
    await process_dubbing_job(job_id)
