#! encoding:utf-8
import uvicorn
from api.endpoint.web_app import app

if __name__ == "__main__":
    log_config = {
        "version": 1,
        "loggers": {
            "uvicorn": {"level": "INFO", "propagate": True},
            "uvicorn.error": {"level": "INFO", "propagate": True},
            "uvicorn.access": {"level": "INFO", "propagate": True}
        }
    }
    uvicorn.run(app,
                host="0.0.0.0",
                port=7755,
                log_level="info",
                log_config=log_config)
