from pathlib import Path
import sys
import os


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import app  # noqa: E402


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        app_dir=".",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,
    )


if __name__ == "__main__":
    main()
