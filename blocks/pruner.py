import yaml
import tensorflow as tf
import tensorflow_model_optimization as tfmot


class ModelPruner:
    def __init__(self):
        with open('configs/pruning_parameters.yaml', 'r') as f:
            self.pruning_config = yaml.safe_load(f)

        self.pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.pruning_config['initial_sparsity'],
                final_sparsity=self.pruning_config['final_sparsity'],
                begin_step=self.pruning_config['begin_step'],
                end_step=self.pruning_config['end_step']
            )
        }

    def prune_model(self, model):
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        model_for_pruning = prune_low_magnitude(model, **self.pruning_params)
        model_for_pruning.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )
        self.pruned_model = model_for_pruning

    def finetune_model(self, X_train, y_train, X_val, y_val):
        self.pruned_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]  # Required for pruning
        )

        final_model = tfmot.sparsity.keras.strip_pruning(self.pruned_model)

        return final_model