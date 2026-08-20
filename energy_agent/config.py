from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    database: Path
    artifacts_dir: Path
    google_api_key: str
    default_model: str
    memory_turn_limit: int
    summary_char_limit: int
    max_parallel_agents: int

    @classmethod
    def load(cls, root: Path) -> "Settings":
        data_dir = Path(os.getenv("ENERGY_AGENT_DATA_DIR", root / "runtime")).resolve()
        return cls(
            root=root,
            data_dir=data_dir,
            database=Path(os.getenv("ENERGY_AGENT_DB", data_dir / "energy_agent.db")).resolve(),
            artifacts_dir=Path(os.getenv("ENERGY_AGENT_ARTIFACTS", data_dir / "artifacts")).resolve(),
            google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            default_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip(),
            memory_turn_limit=max(4, int(os.getenv("MEMORY_RECENT_TURNS", "12"))),
            summary_char_limit=max(2000, int(os.getenv("MEMORY_SUMMARY_CHARS", "10000"))),
            max_parallel_agents=max(1, int(os.getenv("MAX_PARALLEL_AGENTS", "4"))),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
