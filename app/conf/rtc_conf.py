# Agora RTC App configurations
# No.1
# APPID = "275656ac7baf43e0b57d672aba291020"
# APP_CERTIFICATE = "3bc37bad6a704e84be3331cb9985a976"
# No.2
# APPID = "0bbeaf7d7f88416d81cc4e6037e92bb6"
# APP_CERTIFICATE = "f2db6b70fdac4ae89f6c0b048816c682"
# No.3
APPID = "caef6d355a154b39b214e76844da4121"
APP_CERTIFICATE = "9941236ebfb44e13bb93cb3bea58f86c"

import os
from util.app_util import get_unique_id_from_keyword
SERVER_UID = get_unique_id_from_keyword(os.getenv('HOSTNAME', default='SERVER'), 100)
