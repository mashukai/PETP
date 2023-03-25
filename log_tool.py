import logging
from datetime import datetime
import pytz

def logger_config(log_name, log_path, add_console=True, filelog_level=logging.DEBUG, consolelog_level=logging.ERROR):
    
    logger = logging.getLogger(log_name)# 获取logger对象,取名
    logger.setLevel(level=logging.DEBUG)# 输出level及以上级别的信息，针对所有输出的第一层过滤

    # set two handlers, consoleHandler相当于控制台输出，fileHandler文件输出,
    fileHandler = logging.FileHandler(log_path, encoding='UTF-8')
    fileHandler.setLevel(filelog_level)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(consolelog_level)

    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)  # 为logger对象添加句柄
    if add_console:
        logger.addHandler(consoleHandler)  # 为logger对象添加句柄

    return logger

# sg = pytz.timezone('Asia/Singapore')
# timeSG = datetime.now(sg).strftime("%Y-%M-%d_%H%M")

# mylog = logger_config('test', 'testlog'+timeSG+'.log')
# mylog.info('...')
# mylog.debug('???')
# mylog.error('!!!')
# mylog.warning(':::')
