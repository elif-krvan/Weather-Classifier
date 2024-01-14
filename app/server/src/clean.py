import schedule
import time
from datetime import datetime, timedelta
import os

def delete_old_img(folder_path, hours_threshold=1):
    now = datetime.now()
    try:
        for filename in os.listdir(folder_path):
            
                file_path = os.path.join(folder_path, filename)

                if os.path.isfile(file_path):
                    creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    time_difference = now - creation_time

                    if time_difference.total_seconds() > hours_threshold * 3600:
                        os.remove(file_path)
    except Exception as e:
        # Handle other exceptions
        print(f"An unexpected error occurred: {e}", flush=True)

def job():
    print("Running scheduled task at ", datetime.now(), flush=True)
    folder_path = "user_uploads"
    delete_old_img(folder_path)

# schedule for every 4 hours
schedule.every(4).hours.do(job)

files = os.listdir(".")
print("current dir:", files, flush=True)
for filename in files:
    if os.path.isdir(filename):
        print("current dir: ", filename, " content: ", os.listdir(filename), flush=True)
        
# run the task initially
job()

while True:
    schedule.run_pending()
    time.sleep(1)
