# discord_notifier.py
import requests
import datetime
from typing import Optional, Dict, Any

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1380139392869339228/wUt9IJkN6GBsWf08Rtt85-TiEP0wzptddTuBxr2S5JzdWisLIC9BcrU8XdFUbBSWLW9d"

def send_experiment_complete(scene_name: str, message: str):
    """실험 완료 알림을 보냅니다."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    embed_data = {
        "title": "🎉 실험이 완료되었습니다!",
        "color": 5814783,  # 파란색
        "fields": [
            {"name": "Scene", "value": scene_name, "inline": True},
            {"name": "Message", "value": message, "inline": True},
            {"name": "완료 시간", "value": current_time, "inline": True}
        ]
    }


    data = {
        "content": "@here 실험이 완료되었습니다!",
        "username": "실험 알리미",
        "embeds": [embed_data]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
        print("디스코드 알림 전송 성공!")
    except Exception as e:
        print(f"디스코드 알림 전송 실패: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python experiment_utils.py <scene_name> <message>")
        sys.exit(1)
    
    scene_name = sys.argv[1]
    message = sys.argv[2]
    send_experiment_complete(scene_name, message)