#! encoding: utf-8

import logging
import logging.config

# 定义配置字典
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s - %(module)s:%(funcName)s:%(lineno)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'app.log',
            'mode': 'w',  # 可以改为 'a' 以追加模式写入
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'my_module': {  # 可以根据需要配置其他 logger
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

# 配置 logging
logging.config.dictConfig(LOGGING_CONFIG)

# 创建 logger
logger = logging.getLogger()
