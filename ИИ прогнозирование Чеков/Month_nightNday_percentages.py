import pandas as pd

from constants import SRC_PATH, HORIZON, PATH_TO_DAY_NIGHT_FILE
from current_forecast import proportion_of_day_night, periods_maker

df = pd.read_csv(SRC_PATH, sep='|')

def persantages_of_predicted_month(value_day, value_night):
    pers_day = value_day/(value_day+value_night)
    return pers_day, 1-pers_day

