import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import yaml
from tensorflow import keras


class MagnitudePruner:
    def __init__(self):
        """
        Initialize the MagnitudePruner.

        Reads the pruning parameters from 'configs/pruning_parameters.yaml' and sets them as instance variables.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

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
        """Prune the given model using the MagnitudePruner's pruning parameters.

        Args:
            model (tf.keras.Model): The model to be pruned.

        Returns:
            None
        """

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        model_for_pruning = prune_low_magnitude(model, **self.pruning_params)
        model_for_pruning.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )
        self.pruned_model = model_for_pruning


    def finetune_model(self, X_train, y_train, X_val, y_val):
        """
        Finetune the pruned model using the given training data.

        Args:
            X_train (numpy.ndarray): The training input data.
            y_train (numpy.ndarray): The training labels.
            X_val (numpy.ndarray): The validation input data.
            y_val (numpy.ndarray): The validation labels.

        Returns:
            tf.keras.Model: The finetuned model.
        """

        self.pruned_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]  # Required for pruning
        )

        final_model = tfmot.sparsity.keras.strip_pruning(self.pruned_model)

        return final_model


class LTHPruner:
    """
    Pruning algorithm implementation using Lottery Ticket Hypothesis, based on:
    Frankle, Jonathan, and Michael Carbin. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. 2019,
    https://arxiv.org/abs/1803.03635
    Using: **Strategy 1: Iterative pruning with resetting.**
    """

    def _get_weights(self, model):
        """
        Returns a list of the model's trainable weights as numpy arrays.

        Parameters
        ----------
        model : tf.keras.Model
            The model whose weights are to be retrieved.

        Returns
        -------
        List[numpy.ndarray]
            A list of the model's trainable weights as numpy arrays.
        """

        return [w.numpy().copy() for w in model.trainable_weights]


    def _apply_masks(self, model, masks):
        """
        Applies the given masks to the model's trainable weights.

        Parameters
        ----------
        model : tf.keras.Model
            The model whose weights are to be masked.
        masks : List[numpy.ndarray]
            A list of the masks to be applied to the model's trainable weights.

        Returns
        -------
        None
        """

        for weight, mask in zip(model.trainable_weights, masks):
            weight.assign(weight * mask)


    def _compute_masks(self, model, sparsity_percent):
        """
        Computes masks for pruning the model based on the given sparsity percent.

        Parameters
        ----------
        model : tf.keras.Model
            The model whose weights are to be pruned.
        sparsity_percent : int
            The percentage of weights to be pruned.

        Returns
        -------
        List[numpy.ndarray]
            A list of masks to be applied to the model's trainable weights. The masks are numpy arrays of the same shape as the corresponding weights, with values of 0 (pruned) or 1 (not pruned).

        Notes
        -----
        This method uses a global threshold across all non-bias weights to determine which weights to prune. The threshold is the sparsity_percent-th percentile of all non-bias weights.
        """

        masks = []
        all_weights = []

        for w in model.trainable_weights:
            if 'bias' not in w.name:
                all_weights.append(np.abs(w.numpy().flatten()))

        # Global threshold across all non-bias weights
        all_flat = np.concatenate(all_weights)
        threshold = np.percentile(all_flat, sparsity_percent)

        for w in model.trainable_weights:
            if 'bias' in w.name:
                masks.append(np.ones_like(w.numpy()))  # never prune biases
            else:
                masks.append((np.abs(w.numpy()) > threshold).astype(np.float32))

        return masks


    def iterative_pruning_with_reset(
        self,
        model,
        x_train, y_train,
        x_val, y_val,
        prune_percent_per_round=20,  # s% per round
        train_iterations=1000,        # j iterations
        num_rounds=5,
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    ):
        # save initial weights θ₀
        theta_0 = self._get_weights(model)

        # initialize mask: all ones (nothing pruned)
        masks = [np.ones_like(w.numpy()) for w in model.trainable_weights]

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # track cumulative sparsity for reporting
        cumulative_sparsity = 0

        for round_num in range(num_rounds):
            print(f"\n--- Pruning Round {round_num + 1}/{num_rounds} ---")
            print(f"Current sparsity: {cumulative_sparsity:.1f}%")

            # train for j iterations (apply mask after each batch)
            # custom training loop -> enforce masks during training
            steps = 0
            batch_size = 64
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            dataset = dataset.shuffle(10000).batch(batch_size).repeat()

            loss_fn = keras.losses.SparseCategoricalCrossentropy()
            opt = keras.optimizers.get(optimizer)

            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    preds = model(x_batch, training=True)
                    loss_val = loss_fn(y_batch, preds)

                grads = tape.gradient(loss_val, model.trainable_weights)
                opt.apply_gradients(zip(grads, model.trainable_weights))

                # reapply masks after each gradient update (keep pruned weights at 0)
                self._apply_masks(model, masks)

                steps += 1
                if steps >= train_iterations:
                    break

            # eval
            val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
            print(f"Val accuracy after training: {val_acc:.4f}")

            # prune s% of *remaining* weights
            # update cumulative sparsity: P_m' = (P_m - s)%
            cumulative_sparsity = 100 - (100 - cumulative_sparsity) * (1 - prune_percent_per_round / 100)
            masks = self._compute_masks(model, cumulative_sparsity)

            total_params = sum(m.size for m in masks)
            active_params = sum(m.sum() for m in masks)
            print(f"Params remaining: {active_params:.0f}/{total_params} ({100*active_params/total_params:.1f}%)")

            # reset weights to θ₀
            for weight, w0, mask in zip(model.trainable_weights, theta_0, masks):
                weight.assign(w0 * mask)  # reset AND apply mask



        print("\nDone! Final network is a sparse 'lottery ticket'.")
        return model, masks
