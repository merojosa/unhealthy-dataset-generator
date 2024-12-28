from typing import Literal
from datetime import time, datetime, date


# This will be fancier in the future I guess?
def get_times(video_start_time: time, start_time: time, end_time: time):
    dt_video_start_time = datetime.combine(date.today(), video_start_time)
    dt_start_time = datetime.combine(date.today(), start_time)
    dt_end_time = datetime.combine(date.today(), end_time)

    start_time_seconds = (dt_start_time - dt_video_start_time).total_seconds()
    end_time_seconds = (dt_end_time - dt_video_start_time).total_seconds()

    return (start_time_seconds, end_time_seconds)
