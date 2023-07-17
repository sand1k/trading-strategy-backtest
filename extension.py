import sys
from os import environ
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
from zipline.utils.paths import zipline_root
from zipline.data.bundles import register
from zipline.utils.calendar_utils import get_calendar

sys.path.append(zipline_root())
from sharadar_ingest import sharadar_to_bundle

end_date = environ.get('SHARADAR_END_DATE', None)
if end_date is None:
    end_date = date.today()
    
start_date = environ.get('SHARADAR_START_DATE', None)
if start_date is None:
    start_date = end_date - timedelta(days=365*10)

calendar_name = 'XNYS'
calendar = get_calendar(calendar_name)
sessions = calendar.sessions_in_range(start_date, end_date)

register(
    'sharadar',
    sharadar_to_bundle(),
    calendar_name = calendar_name,
    start_session=sessions[0],
    end_session=sessions[-1]
)
