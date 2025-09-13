import multiprocessing
from ultralytics import YOLO
import os
import shutil
import glob


PROJECT_NAME = "Tumor_Detection_YOLOV11"
DATA_PATH = f"{PROJECT_NAME}/dataset_tumor/data.yaml"
TEST_IMAGES_PATH = f"{PROJECT_NAME}/dataset_tumor/test/images"

TRAINING_ARGS = {
    "data": DATA_PATH,
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "project": "Tumor_Detection_YOLOV11/runs",
    "name": "detect/train"
}

if not os.path.exists(TEST_IMAGES_PATH):
    print(
        f"Warning: Test images path not found at {TEST_IMAGES_PATH}. Prediction step will be skipped.")
    TEST_IMAGES_PATH = None


def train_model_process(model_path, training_args):

    print("Training process started...")
    model = YOLO(model_path)
    model.train(**training_args)
    print("Training process finished.")
    return


def results_save(test_source):
   # Define the base directory where YOLO stores its runs
    runs_dir = 'Tumor_Detection_YOLOV11/runs/detect'

    # Get a list of all directories in 'runs/detect'
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))

    # If no training directories are found, return None
    if not train_dirs:
        print("No YOLO training runs found in 'runs/detect'.")
        return None

    # Find the latest directory based on the creation time
    latest_dir = max(train_dirs, key=os.path.getctime)
    print(f"Found latest training directory: {latest_dir}")

    # Construct the full path to the best.pt file
    best_model_path = os.path.join(latest_dir, 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}.")
        return
    print("Loading the best-trained model for prediction...")
    model = YOLO(best_model_path)

    print("Running prediction on test data...")
    model.predict(source=test_source, save=True, save_txt=True, project='Tumor_Detection_YOLOV11',
                  name='Results', exist_ok=True)
    print("Prediction finished and results saved.")
    destination_path = "Tumor_Detection_YOLOV11/tumordetector.pt"
    shutil.copy(best_model_path, destination_path)


if __name__ == "__main__":
    # The if __name__ == "__main__": block is essential for multiprocessing.
    # It prevents the child process from re-running the parent script.

    # Define the parameters for training
    model_path = "yolo11n.pt"

    print("Starting the model training in a new process...")

    # Create the multiprocessing.Process with the function and its arguments
    train_results_process = multiprocessing.Process(
        target=train_model_process,
        args=(model_path, TRAINING_ARGS)
    )

    # Start the process
    train_results_process.start()

    # Wait for the process to complete
    train_results_process.join()

    print("Training process has completed. Proceeding with prediction.")

    if TEST_IMAGES_PATH:
        # Run the prediction and saving logic using the newly trained model
        results_save(TEST_IMAGES_PATH)
    else:
        print("Skipping prediction due to missing test image path.")

    print("Finished Testing, Results Saved")

