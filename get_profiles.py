import asyncio
import json
from src.evaluation.user_profiler import UserProfiler
from pathlib import Path

async def main():
    p = UserProfiler(Path('data/chat_logs/sessions.db'))
    profs = await p.build_all_profiles()
    print(json.dumps(profs, indent=2))

asyncio.run(main())
