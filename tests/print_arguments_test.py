import os
import rootpath
import test_settings
from utils.beautiful import print_argument_texts

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

argument_ids = [
    "fbe6ad2-2019-04-18T11:12:36Z-00003-000",
    "fbe6ad2-2019-04-18T11:12:36Z-00004-000",
    "fbe6ad2-2019-04-18T11:12:36Z-00005-000",
    "fbe6ad2-2019-04-18T11:12:36Z-00006-000",
]

print_argument_texts(argument_ids, CSV_PATH)
