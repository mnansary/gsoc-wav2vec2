"""TensorFlow implementation of Wav2Vec2"""

from distutils.command.build import build
import tensorflow as tf

from .encoder import Wav2Vec2Encoder
from .feature_extractor import FeatureExtractorLayer, FeatureProjection
from .spec_augment import apply_spec_augmentation


class Wav2Vec2(tf.keras.layers.Layer):
    def __init__(self,layer_config):
        super().__init__(name="wav2vec2")
        
        self.layer_config = layer_config
        self.hidden_size = layer_config.hidden_size
        self.is_robust = layer_config.is_robust
        self.kernal_sizes = layer_config.kernal_sizes
        self.strides = layer_config.strides

        # spec-augmentation
        self.apply_spec_augment = layer_config.apply_spec_augment
        self.mask_time_prob = layer_config.mask_time_prob
        self.mask_time_length = layer_config.mask_time_length

        num_feature_extractor_layers = len(layer_config.filter_sizes)

        self.feature_extractor = [
            FeatureExtractorLayer(
                layer_config.filter_sizes,
                layer_config.kernal_sizes,
                layer_config.strides,
                conv_bias=layer_config.conv_bias,
                is_gelu_approx=layer_config.is_gelu_approx,
                feature_extractor_norm_type=layer_config.feature_extractor_norm_type,
                layer_id=i,
                name=f"feature_extractor/conv_layers/{i}",
            )
            for i in range(num_feature_extractor_layers)
        ]
        self.feature_projection = FeatureProjection(
            layer_config.hidden_size,
            layer_norm_eps=layer_config.layer_norm_eps,
            dropout=layer_config.dropout,
            name="feature_projection",
        )
        self.encoder = Wav2Vec2Encoder(
            layer_config.hidden_size,
            layer_config.num_heads,
            layer_config.num_layers,
            layer_config.intermediate_size,
            layer_config.num_conv_pos_embeddings,
            layer_config.num_conv_pos_embedding_groups,
            survival_prob=layer_config.survival_prob,
            dropout=layer_config.dropout,
            layer_norm_eps=layer_config.layer_norm_eps,
            is_gelu_approx=layer_config.is_gelu_approx,
            attention_norm_type=layer_config.attention_norm_type,
            name="encoder",
        )
    
    def build(self,input_shape):
        self.masked_spec_augment = self.add_weight(
            name="masked_spec_embed",
            shape=(self.hidden_size,),
            initializer="uniform",
            trainable=True,
        )

    def call(self, batch, attention_mask= None, training=False):
        """
        Args:
            batch (:obj: `tf.Tensor`) of shape (batch_size, seqlen):
                Sound tensor obtained from `Wav2Vec2Processor.__call__`.
            attention_mask (:obj: `tf.Tensor`, `optional`) of shape (batch_size, seqlen):
                Don't pass `attention_mask` when working with checkpoints based on `wav2vec2-base`
                otherwise you should pass this argument.
            training (:obj: `bool`, `optional`):
                Whether to use model for training.

        Returns:
            Logits from the model of shape (batch_size, seqlen, hidden_dim).
        """
        
        batch = tf.expand_dims(batch, axis=-1)
        for feature_extractor_layer in self.feature_extractor:
            batch = feature_extractor_layer(batch)
        batch = self.feature_projection(batch, training=training)

        if training and self.apply_spec_augment:
            batch = apply_spec_augmentation(
                batch,
                self.masked_spec_augment,
                self.mask_time_prob,
                self.mask_time_length,
            )

        if attention_mask is not None:
            input_length = tf.reduce_sum(attention_mask, axis=-1)
            for kernal_size, stride in zip(self.kernal_sizes, self.strides):
                input_length = 1 + (input_length - kernal_size) // stride

            attention_mask = tf.sequence_mask(input_length, maxlen=batch.shape[1])

        batch = self.encoder(batch, attention_mask=attention_mask, training=training)
        return batch

    def freeze_feature_extractor(self):
        """This will freeze the feature extractor layers (Recommended to use for fine-tuning)."""
        for i in range(len(self.feature_extractor)):
            self.feature_extractor[i].trainable = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_config": self.layer_config
            }
        )
        return config

