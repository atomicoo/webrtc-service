from pydantic import BaseModel, Field

class ConfigResponseData(BaseModel):
    app_id: str = Field(..., description="APP ID")
    token: str = Field(..., description="Token")
    channel_id: str = Field(..., description="频道ID")
    client_user_id: int = Field(..., description="客户端用户ID")

class ConfigResponse(BaseModel):
    code: int = Field(..., description="错误码")
    message: str = Field(..., description="错误信息")
    data: ConfigResponseData | None = Field(..., description="数据")


class JoinChannelRequest(BaseModel):
    channel_id: str = Field(..., description="频道ID")
    client_user_id: int = Field(..., description="客户端用户ID")
    index: int = Field(..., description="序号, 可以是时间戳")

class JoinChannelResponseData(BaseModel):
    channel_id: str = Field(..., description="频道ID")
    server_user_id: int = Field(..., description="服务器用户ID")
    index: int = Field(..., description="序号，同请求中的序号")

class JoinChannelResponse(BaseModel):
    code: int = Field(..., description="错误码")
    message: str = Field(..., description="错误信息")
    data: JoinChannelResponseData | None = Field(..., description="数据")


class QuitChannelRequest(JoinChannelRequest):
    pass

class QuitChannelResponse(BaseModel):
    code: int = Field(..., description="错误码")
    message: str = Field(..., description="错误信息")
