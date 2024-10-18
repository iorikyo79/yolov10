import os
from ultralytics import YOLO
from ultralytics import settings
import mlflow
import mlflow.pytorch

def on_train_epoch_end(trainer):
    """Custom callback to log metrics to MLflow at the end of each training epoch."""
    metrics = trainer.metrics
    epoch = trainer.epoch
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            modified_key = key.replace('(B)', '').replace('/', '_')
            mlflow.log_metric(modified_key, value, step=epoch)
            #print(f"Logged metric: {modified_key} = {value} at epoch {epoch}")

def main():
    settings['mlflow'] = False
    
    mlflow.set_tracking_uri('http://10.10.40.132:8080')
    mlflow.set_experiment('Yolov10')

    model = YOLO("/mnt/Disk1/source/yolov10/weights/yolov10l.pt")

    # Add the custom callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    params = {
        'data': '/mnt/Disk1/source/yolov10/data/data.yaml',
        'name': 'Ex1-R1-BaseLine',
        'single_cls': True,
        'epochs': 150,
        'optimizer': 'AdamW',
        'lr0': 0.0002,
        'lrf': 0.0000002,
        'cos_lr': True,
        'save_txt': True,
        'save_conf': True,
        'fraction': 1
    }
    
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        
        try:
            results = model.train(**params)
            
            # Log final metrics
            if hasattr(results, 'results_dict'):
                for key, value in results.results_dict.items():
                    if isinstance(value, (int, float)):
                        modified_key = f"final_{key.replace('(B)', '').replace('/', '_')}"
                        mlflow.log_metric(modified_key, value)
                        print(f"Logged final metric: {modified_key} = {value}")
            
            # Save final model
            model_path = model.export()
            mlflow.log_artifact(model_path, "final_model")
            print(f"Final model saved at: {model_path}")
            
            mlflow.end_run(status='FINISHED')   
            
        except Exception as e:
            print(f"Error : {e}")
            mlflow.end_run(status='FAILED')

    print("Training completed. Check MLflow for results.")

if __name__ == '__main__':
    main()
