import datetime as dt
def is_expired(expires_iso: str) -> bool:
    try:
        return dt.datetime.utcnow() > dt.datetime.fromisoformat(expires_iso.replace("Z",""))
    except Exception:
        return False
