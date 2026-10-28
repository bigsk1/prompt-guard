"""Process entrypoint. One process loads one model."""

import logging
import sys

import uvicorn
from pydantic import ValidationError

from prompt_guard.app import create_app
from prompt_guard.config import load_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        settings = load_settings()
    except (ValidationError, FileNotFoundError) as exc:
        logging.getLogger("prompt_guard").error("invalid configuration: %s", exc)
        sys.exit(1)
    logging.getLogger().setLevel(settings.log_level.upper())
    logging.getLogger("transformers").setLevel(logging.ERROR)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level,
        access_log=True,
        proxy_headers=False,
        server_header=False,
        ssl_certfile=settings.tls_certfile,
        ssl_keyfile=settings.tls_keyfile,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
