import os
import site
from pathlib import Path

from ..CyberarkClient import CyberarkClient

ROOT = Path(__file__).resolve().parent.parent.parent

DEBUG = True

for directory in os.listdir(ROOT):
    site.addsitedir(directory)

CYBERBARK_PARAMS = {
    'safe': 'AIM_PAML_SMS',
    'password_key': 'pamlsmsopr',
    'folder': 'root'
}
