import json
from nse import NSE
from datetime import datetime

# Initialize NSE API
nse = NSE("downloads")

def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_corporate_actions(symbol):
    """Fetch forthcoming corporate actions for a given NSE stock symbol."""
    try:
        actions = nse.actions(segment='equities', symbol=symbol)
        filtered_actions = [{"exDate": a["exDate"], "subject": a["subject"]} for a in actions]

        return filtered_actions if filtered_actions else "No corporate actions found."
    except Exception as e:
        return f"Error fetching corporate actions: {e}"

def get_board_meetings(symbol):
    """Fetch forthcoming board meetings for a given NSE stock symbol."""
    try:
        meetings = nse.boardMeetings(index='equities', symbol=symbol)
        filtered_meetings = [{"bm_date": m["bm_date"], "bm_desc": m["bm_desc"]} for m in meetings]
        return filtered_meetings if filtered_meetings else "No board meetings found."
    except Exception as e:
        return f"Error fetching board meetings: {e}"


# example usage
if __name__ == "__main__":
    symbols = ["RELIANCE", "TATASTEEL", "HDFC"]
    
    for symbol in symbols:
        print("\n📌 Fetching Corporate Actions...")
        corp_actions = get_corporate_actions(symbol)
        save_to_json(corp_actions, f"market/{symbol}corp_actions.json")

        print("\n📌 Fetching Board Meetings...")
        board_meetings = get_board_meetings(symbol)
        save_to_json(board_meetings, f"market/{symbol}board_meetings.json")

