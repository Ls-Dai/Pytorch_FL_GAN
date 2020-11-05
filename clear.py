import shutil
import os


def clear_records(if_clients=True, if_servers=True, if_logs=True):

    if if_clients:
        if os.path.exists("clients"):
            shutil.rmtree("clients")
            print("The folder of clients has been removed!")
        else:
            print("No folder of clients exists!")
    
    if if_servers:
        if os.path.exists("servers"):
            shutil.rmtree("servers")
            print("The folder of servers has been removed!")
        else:
            print("No folder of servers exists")

    if if_logs:
        if os.path.exists("logs.txt"):
            os.remove("logs.txt")
            print("The file of logs has been removed!")
        else:
            print("No file of logs exists!")

        print("Now all records have been removed, to begin the program, firstly init the clients and servers!")

if __name__ == "__main__":
    clear_records(if_clients=True, if_servers=True, if_logs=True)