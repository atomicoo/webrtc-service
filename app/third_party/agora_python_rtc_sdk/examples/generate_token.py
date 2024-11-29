from agora_dynamic_key.src.RtcTokenBuilder2 import *

APPID = "275656ac7baf43e0b57d672aba291020"
APP_CERTIFICATE = "3bc37bad6a704e84be3331cb9985a976"

def get_token(channel_name: str, user_id: int):
    token_expiration_in_seconds = 3600
    privilege_expiration_in_seconds = 3600
    token = RtcTokenBuilder.build_token_with_uid(APPID, APP_CERTIFICATE, channel_name, user_id, Role_Subscriber,
                                                 token_expiration_in_seconds, privilege_expiration_in_seconds)
    return token

token = get_token("test", 7777)
print(f"TOKEN: {token}")
# CHANNEL: test; USERID: 1111; TOKEN: 007eJxTYNj8SOxRG8/WhBI915PLhGScfiX2fk/+2NUxaTrfRCEf5r0KDEbmpmamZonJ5kmJaSbGqQZJpuYpZuZGiUmJRpaGBkYGXj/upQnwMTDI2bmzMDJAIIjPwlCSWlzCwmAIBAB+8x3h
# CHANNEL: test; USERID: 3333; TOKEN: 007eJxTYJj2ye5Du1Rd8kPWmX0dXB6exv5vfxzIzW5U065WebPbp06Bwcjc1MzULDHZPCkxzcQ41SDJ1DzFzNwoMSnRyNLQwMhg6Y97aQJ8DAwLP51nZGSAQBCfhaEktbiEhcEYCADDFSAk
# CHANNEL: test; USERID: 7777; TOKEN: 007eJxTYEgovrF5xZ61c9hkOy+s6pwRfV0qY938DwsiAjzWik5bybpTgcHI3NTM1Cwx2TwpMc3EONUgydQ8xczcKDEp0cjS0MDIIOLrvTQBPgYG+/BEFkYGCATxWRhKUotLWBjMgQAABp0fxQ==
