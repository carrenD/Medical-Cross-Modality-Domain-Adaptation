"""notifications.py

convenience functions for status notifications
"""
import logging

# initialize module logger
logger = logging.getLogger(__name__)

# disable pushbullet library logging
pblogger = logging.getLogger('pushbullet')
pblogger.addHandler(logging.NullHandler())

try:
    from config import _PB_API_KEY_
    from pushbullet import Pushbullet

    # pushbullet config
    __pb__ = Pushbullet(_PB_API_KEY_)
    __pb_channel_research__ = __pb__.channels[0]

    def pushNotification(title, body):
        __pb_channel_research__.push_note(title, body)
except:
    logger.debug('Notifications have been disabled because no valid pushbullet api key was defined in config.py')
    def pushNotification(title, body):
        pass
