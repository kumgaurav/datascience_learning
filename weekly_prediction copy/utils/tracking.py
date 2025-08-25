import os
from typing import Any, Dict, Optional


class _NullTracker:
    def __init__(self):
        self.enabled = False

    def start_run(self, run_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        return self

    def log_params(self, params: Dict[str, Any]):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def set_tags(self, tags: Dict[str, Any]):
        pass

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        pass

    def end_run(self):
        pass


class _MLflowTracker:
    def __init__(self, experiment_name: Optional[str] = None, tracking_uri: Optional[str] = None):
        import mlflow  # type: ignore
        self.mlflow = mlflow
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            self.mlflow.set_experiment(experiment_name)
        self.enabled = True
        self._active = False

    def start_run(self, run_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        if self._active:
            try:
                self.mlflow.end_run()
            except Exception:
                pass
        try:
            self.mlflow.start_run(run_name=run_name)
            self._active = True
            if params:
                self.log_params(params)
        except Exception:
            self._active = False
        return self

    def log_params(self, params: Dict[str, Any]):
        try:
            self.mlflow.log_params(params)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        try:
            self.mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def set_tags(self, tags: Dict[str, Any]):
        try:
            self.mlflow.set_tags(tags)
        except Exception:
            pass

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        try:
            self.mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception:
            pass

    def end_run(self):
        if not self._active:
            return
        try:
            self.mlflow.end_run()
        except Exception:
            pass
        finally:
            self._active = False


class _WandbTracker:
    def __init__(self, project: Optional[str] = None, entity: Optional[str] = None):
        import wandb  # type: ignore
        self.wandb = wandb
        self.project = project or os.environ.get("WANDB_PROJECT")
        self.entity = entity or os.environ.get("WANDB_ENTITY")
        self.enabled = True
        self._run = None

    def start_run(self, run_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        if self._run is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass
            self._run = None
        try:
            self._run = self.wandb.init(project=self.project, entity=self.entity, name=run_name, config=params or {})
        except Exception:
            self._run = None
        return self

    def log_params(self, params: Dict[str, Any]):
        try:
            if self._run is not None:
                self._run.config.update(params, allow_val_change=True)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        try:
            if self._run is not None:
                if step is not None:
                    metrics = dict(metrics)
                    metrics["_step"] = step
                self.wandb.log(metrics)
        except Exception:
            pass

    def set_tags(self, tags: Dict[str, Any]):
        try:
            if self._run is not None and hasattr(self._run, "tags"):
                self._run.tags = list(set(list(self._run.tags or []) + [f"{k}:{v}" for k, v in tags.items()]))
        except Exception:
            pass

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        try:
            # Simple file logging; for directories or typed artifacts, callers can customize further
            self.wandb.save(path)
        except Exception:
            pass

    def end_run(self):
        try:
            if self._run is not None:
                self.wandb.finish()
        except Exception:
            pass
        finally:
            self._run = None


def get_tracker(mode: str = "none",
                experiment_name: Optional[str] = None,
                mlflow_uri: Optional[str] = None,
                wandb_project: Optional[str] = None,
                wandb_entity: Optional[str] = None):
    mode = (mode or "none").lower()
    if mode == "mlflow":
        try:
            return _MLflowTracker(experiment_name=experiment_name, tracking_uri=mlflow_uri)
        except Exception:
            return _NullTracker()
    if mode == "wandb":
        try:
            return _WandbTracker(project=wandb_project or experiment_name, entity=wandb_entity)
        except Exception:
            return _NullTracker()
    return _NullTracker()


