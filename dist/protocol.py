#------------------------------------------------------------------------
# Heartbeat
#------------------------------------------------------------------------

PING = 'ping'
PONG = 'pong'

#------------------------------------------------------------------------
# Server Instructions
#------------------------------------------------------------------------

MAP       = 'map'
REDUCE    = 'reduce'
DONE      = 'done'
BYTECODE  = 'bytecode'
HEARTBEAT = 'heartbeat'

#------------------------------------------------------------------------
# Worker
#------------------------------------------------------------------------

MAPATOM     = 'mapdone'
MAPCHUNK    = 'mapkeydone'
REDUCEATOM  = 'reducedone'

CONNECT_WORKER = 'connect_worker'
CONNECT_STORE  = 'connect_store'

#------------------------------------------------------------------------
# Storage
#------------------------------------------------------------------------

LIST   = 'list'
SET    = 'set'
RENAME = 'rename'
GET    = 'get'
DELETE = 'delete'

ACK_LIST   = 'acklist'
ACK_SET    = 'ackset'
ACK_RENAME = 'ackrename'
ACK_GET    = 'ackget'
ACK_DELETE = 'ackdelete'
