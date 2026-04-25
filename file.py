import os

def create_dataset(gesture_name, num_samples=60, base_path="dataset"):
    
    # Create main dataset folder if not exists
    os.makedirs(base_path, exist_ok=True)
    
    # Create gesture folder
    gesture_path = os.path.join(base_path, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)
    
    # CSV header
    header = "ax,ay,az,gx,gy,gz,label\n"
    
    for i in range(1, num_samples + 1):
        file_name = f"{gesture_name}_{i:02d}.csv"
        file_path = os.path.join(gesture_path, file_name)
        
        with open(file_path, "w") as f:
            f.write(header)
    
    print(f"Created {num_samples} CSV files in '{gesture_path}'")


gesture_name = "m"

create_dataset(gesture_name)