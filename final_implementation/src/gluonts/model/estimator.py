# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import NamedTuple, Optional

import gluonts
from gluonts.core.component import from_hyperparameters, validated
from gluonts.dataset.common import Dataset
from gluonts.model.predictor import Predictor, DeepEnsemblePredictor


class Estimator:
    """
    An abstract class representing a trainable model.

    The underlying model is trained by calling the `train` method with
    a training `Dataset`, producing a `Predictor` object.
    """

    __version__: str = gluonts.__version__

    prediction_length: int
    freq: str
    lead_time: int

    def __init__(self, lead_time: int = 0, **kwargs) -> None:
        # TODO validation of prediction_length and freq could also
        # TODO be bubbled-up here from subclasses classes
        assert lead_time >= 0, "The value of `lead_time` should be >= 0"

        self.lead_time = lead_time

    def train(
        self, training_data: Dataset, validation_data: Optional[Dataset] = None
    ) -> Predictor:
        """
        Train the estimator on the given data.

        Parameters
        ----------
        training_data
            Dataset to train the model on.
        validation_data
            Dataset to validate the model on during training.

        Returns
        -------
        Predictor
            The predictor containing the trained model.
        """
        raise NotImplementedError

    @classmethod
    def from_hyperparameters(cls, **hyperparameters):
        return from_hyperparameters(cls, **hyperparameters)

    @classmethod
    def derive_auto_fields(cls, train_iter):
        return {}

    @classmethod
    def from_inputs(cls, train_iter, **params):
        # auto_params usually include `use_feat_dynamic_real`, `use_feat_static_cat` and `cardinality`
        auto_params = cls.derive_auto_fields(train_iter)
        # user specified 'params' will take precedence:
        params = {**auto_params, **params}
        return cls.from_hyperparameters(**params)


class DummyEstimator(Estimator):
    """
    An `Estimator` that, upon training, simply returns a pre-constructed
    `Predictor`.

    Parameters
    ----------
    predictor_cls
        `Predictor` class to instantiate.
    **kwargs
        Keyword arguments to pass to the predictor constructor.
    """

    @validated()
    def __init__(self, predictor_cls: type, **kwargs) -> None:
        super().__init__(**kwargs)
        self.predictor = predictor_cls(**kwargs)

    def train(
        self,
        training_data: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> Predictor:
        return self.predictor


class DeepEnsembleEstimator(Estimator):

    def __init__(
        self,
        estimator: Estimator,
        num_deep_models: 5,
        **kwargs,
    ) -> None:
        """
        Ensemble of deep leaning models where each component model is trained with the same dataset and has same hyper-
        parameters. The variance comes from the randomness in the training.

        Parameters
        ----------
        estimator
            Base deep learning model.
        num_deep_models
            Number of models in the ensemble.
        kwargs
        """
        super().__init__(**kwargs)
        self.estimator = estimator
        self.num_deep_models = num_deep_models

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Predictor:

        # To avoid dependency on joblib.
        from joblib import Parallel, delayed

        predictors = Parallel(n_jobs=self.num_deep_models, verbose=10, prefer="threads")(
            # TODO: Fix different seed for each run?
            delayed(self.estimator.train)(
                training_data,
                validation_data,
                num_workers,
                num_prefetch,
                shuffle_buffer_length,
                **kwargs,
            )
            for _ in range(self.num_deep_models)
        )

        return DeepEnsemblePredictor(predictors=predictors)