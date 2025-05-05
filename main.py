import sys
from event_processor import process_events

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py eval.csv")
    else:
        process_events(sys.argv[1], "match_clip_01.mp4", "submission.csv") 