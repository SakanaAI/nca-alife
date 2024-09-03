import sys

import socket
import time
from datetime import datetime

def main():
    print(sys.executable)
    # Get the host name
    hostname = socket.gethostname()
    
    # Print the host name
    print(f"Host Name: {hostname}")
    
    # Print the current time
    print(f"Current Time: {datetime.now()}")

    # Wait for 1 minute (60 seconds)
    for i in range(60*10):
        print(i)
        time.sleep(1)
        sys.stdout.flush()
    # time.sleep(60)
    
    # Print the current time again
    print(f"Time After 1 Minute: {datetime.now()}")

if __name__ == "__main__":
    main()






