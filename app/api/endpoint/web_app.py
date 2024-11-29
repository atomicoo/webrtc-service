import time
import threading
import uuid
from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from conf.rtc_conf import APPID, SERVER_UID
from model.api.endpoint.web_app import *
from util.logger import logger
from util.app_util import get_unique_id_from_keyword
from rtcn import connection_cache, user_login_cache
from rtcn import get_token, connect


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info('** server start **')
    yield
    logger.info('** server stop **')


app = FastAPI(lifespan=lifespan)

origins = ["*"]  # allow all IPs
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

router = APIRouter(tags=["rtc"], prefix="/v1/rtc")


@router.get("/health")
def health() -> str:
    return 'ok'


@router.get("/config")
def config(client_user_id: str = Query(..., description="the id of user")) -> ConfigResponse:
    # client_user_id = int(time.time())
    logger.info(f'config: {client_user_id}')
    client_user_id = get_unique_id_from_keyword(client_user_id, 10000)
    # channel_name = uuid.uuid1().hex
    channel_name = uuid.uuid3(uuid.NAMESPACE_OID, str(client_user_id)).hex
    token = get_token(channel_name, client_user_id)
    return ConfigResponse(code=0, message='success',
                          data=ConfigResponseData(app_id=APPID,
                                                  token=token,
                                                  channel_id=channel_name,
                                                  client_user_id=client_user_id))


@router.post("/join_channel")
def join_channel(request: JoinChannelRequest) -> JoinChannelResponse:
    logger.info(f'join_channel: {request}')

    if connection_cache.get(str(request.client_user_id), None):
        logger.error(f'user already in channel: {request.client_user_id}')
        return JoinChannelResponse(code=1, message='user already in channel')

    user_login_cache.set(str(request.client_user_id), int(time.time()))
    thread = threading.Thread(target=connect, args=(
        request.channel_id, request.client_user_id))
    thread.daemon = True
    thread.start()
    return JoinChannelResponse(code=0, message='success',
                               data=JoinChannelResponseData(channel_id=request.channel_id,
                                                            server_user_id=SERVER_UID,
                                                            index=request.index))

@router.post("/quit_channel")
def quit_channel(request: QuitChannelRequest) -> QuitChannelResponse:
    logger.info(f'quit_channel: {request.client_user_id}')
    user_login_cache.remove(str(request.client_user_id))
    return QuitChannelResponse(code=0, message='success')


app.include_router(router)
