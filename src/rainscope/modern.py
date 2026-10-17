"""Recent models with explicit context budgets and validation-only early stopping."""

import copy
import numpy as np
from sklearn.model_selection import train_test_split
from .models import preprocessing


class TabMModel:
    def __init__(self, config):
        self.config = config

    def fit(self, x, y, xv, yv):
        import torch
        from tabm import TabM

        c = self.config
        torch.manual_seed(c["seed"])
        torch.set_num_threads(c["workers"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = preprocessing()
        a = self.preprocessor.fit_transform(x).astype("float32")
        b = self.preprocessor.transform(xv).astype("float32")
        self.model_args = dict(
            n_num_features=a.shape[1],
            d_out=1,
            n_blocks=2,
            d_block=c["tabm_width"],
            dropout=0.1,
            k=c["tabm_members"],
        )
        self.model = TabM.make(**self.model_args).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        xt = torch.tensor(a, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        best, bad = float("inf"), 0
        self.history = []
        self.training_rows = len(y)
        for epoch in range(c["tabm_epochs"]):
            self.model.train()
            order = torch.randperm(len(y), device=self.device)
            losses = []
            for ix in order.split(c["tabm_batch_size"]):
                opt.zero_grad(set_to_none=True)
                logits = self.model(xt[ix]).squeeze(-1)
                target = yt[ix, None].expand_as(logits)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            p = self._predict_array(b)
            val_loss = float(
                -(
                    yv * np.log(np.clip(p, 1e-7, 1))
                    + (1 - yv) * np.log(np.clip(1 - p, 1e-7, 1))
                ).mean()
            )
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "validation_log_loss": val_loss,
                }
            )
            print(
                f"TabM epoch {epoch + 1}: validation log loss={val_loss:.4f}",
                flush=True,
            )
            if val_loss < best - 0.0001:
                best, bad = val_loss, 0
                state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch + 1
            else:
                bad += 1
                if bad >= c["tabm_patience"]:
                    break
        self.model.load_state_dict(state)
        self.model.to("cpu")  # Other estimators should have access to GPU memory.
        del xt, yt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _predict_array(self, a):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device).eval()
        output = []
        with torch.inference_mode():
            for i in range(0, len(a), 1024):
                x = torch.tensor(a[i : i + 1024], device=self.device)
                output.append(self.model(x).squeeze(-1).sigmoid().mean(1).cpu().numpy())
        return np.concatenate(output)

    def predict_proba(self, x):
        p = self._predict_array(self.preprocessor.transform(x).astype("float32"))
        self.model.to("cpu")
        return np.column_stack([1 - p, p])

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model" in state:
            state["model"] = copy.deepcopy(state["model"]).cpu()
        return state


class TabICLModel:
    def __init__(self, config):
        self.config = config

    def fit(self, x, y, xv=None, yv=None):
        import torch
        from tabicl import TabICLClassifier

        c = self.config
        self.preprocessor = preprocessing("ordinal")
        # Choose context from training only. Never use validation/test labels as context.
        ix = np.arange(len(y))
        if len(ix) > c["tabicl_context_rows"]:
            ix, _ = train_test_split(
                ix,
                train_size=c["tabicl_context_rows"],
                stratify=y,
                random_state=c["seed"],
            )
        self.training_rows = len(ix)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        a = self.preprocessor.fit_transform(x.iloc[ix]).astype("float32")
        self.model = TabICLClassifier(
            n_estimators=c["tabicl_estimators"],
            batch_size=1,
            device=self.device,
            checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
            random_state=c["seed"],
            n_jobs=c["workers"],
            verbose=False,
        )
        self.model.fit(a, y[ix])
        return self

    def predict_proba(self, x):
        a = self.preprocessor.transform(x).astype("float32")
        batch = self.config["tabicl_prediction_batch"]
        return np.concatenate(
            [
                self.model.predict_proba(a[i : i + batch])
                for i in range(0, len(a), batch)
            ]
        )
