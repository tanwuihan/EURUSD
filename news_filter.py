from datetime import datetime, time

HIGH_IMPACT_NEWS = [
    {"date": "2026-01-17", "time": time(13, 30)},
]

def is_news_time(now):
    for e in HIGH_IMPACT_NEWS:
        d = datetime.strptime(e["date"], "%Y-%m-%d").date()
        if now.date() == d:
            mins = abs(
                (now.hour*60 + now.minute) -
                (e["time"].hour*60 + e["time"].minute)
            )
            if mins <= 60:
                return True
    return False
