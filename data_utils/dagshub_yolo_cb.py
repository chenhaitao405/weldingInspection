import mlflow
import os
import re
import ultralytics


class CustomYOLO(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = ultralytics.YOLO(context.artifacts['path'], task='segment')

    def predict(self, context, img):
        preds = self.model(img)
        
        return preds

def custom_on_pretrain_routine_end(trainer):
    if mlflow:
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
        mlflow.log_params(vars(trainer.model.args))

def custom_on_fit_epoch_end(trainer):
    if mlflow:
        metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
        mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def generate_on_train_end(model_name):
    def custom_on_train_end(trainer):
        """Called at end of train loop to log model artifact info."""
        if mlflow:
            experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
            mlflow.log_artifact(trainer.save_dir)

            #（ log_model还有bug，当前直接将训练权重保存在experiment中）
            # mlflow.pyfunc.log_model(artifact_path=experiment_name,
            #                         python_model=CustomYOLO(),
            #                         artifacts={'path': str(trainer.best)},
            #                         registered_model_name=model_name)

    return custom_on_train_end

def generate_callbacks_fn(model_name):
    def custom_callbacks_fn(instance):
        from ultralytics.utils.callbacks.mlflow import callbacks as mlflow_cb
        mlflow_cb['on_pretrain_routine_end'] = custom_on_pretrain_routine_end
        mlflow_cb['on_fit_epoch_end'] = custom_on_fit_epoch_end
        mlflow_cb['on_train_end'] = generate_on_train_end(model_name)
        for k, v in mlflow_cb.items():
            if v not in instance.callbacks[k]:  # prevent duplicate callbacks addition
                instance.callbacks[k].append(v)

    return custom_callbacks_fn
    