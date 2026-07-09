"""Direct desktop WebSocket gateway (audio ingress + feedback egress).

Bypasses the NestJS backend on the realtime critical path:
- desktop-app streams PCM s16le binary frames straight into the python
  pipeline (same JWT access token used on the backend egress WS);
- feedback events are pushed back to the desktop as JSON frames on the
  same server, while the gRPC publish to the backend continues in
  parallel for persistence/dashboard only.
"""

from .feedback_hub import FeedbackHub
from .gateway_server import DesktopWsGateway
from .jwt_auth import DesktopWsAuthenticator, WsAuthError

__all__ = [
    'DesktopWsGateway',
    'FeedbackHub',
    'DesktopWsAuthenticator',
    'WsAuthError',
]
