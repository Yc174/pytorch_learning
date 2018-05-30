#!/usr/local/bin/python
# -*- coding:utf-8 -*-
import logging

logger = logging.getLogger('global.log')
print('logging... ....')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
format_str = '%(asctime)s-rk{}-%(filename)s#%(lineno)d:%(message)s'.format(0)
formatter = logging.Formatter(format_str)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info('dist:{}'.format('ewdweddwedw'))



# # 通过下面的方式进行简单配置输出方式与日志级别
# logging.basicConfig(filename='logger.log', level=logging.INFO)
#
# logging.debug('debug message')
# logging.info('info message')
# logging.warn('warn message')
# logging.error('error message')
# logging.critical('critical message')
