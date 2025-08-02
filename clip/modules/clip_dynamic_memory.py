import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip.modules.vision_transformer_memory as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from clip.modules import clip_utils, heads, objectives, clip
import copy

def load_clip_to_cpu(backbone_name, memory_length, memory_depth):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")  # .eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict(), memory_length, memory_depth)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.memory_length = clip_model.memory_length

    def forward(self, tokenized_texts, all_memories_text, missing_type):
        x = self.token_embedding(tokenized_texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, all_memories_text, 0, missing_type]  # third argument is the counter which denotes depth of memory
        outputs = self.transformer(combined)
        x = outputs[0][self.memory_length:]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ResidualDynamicNet(nn.Module):
    """
    A dynamic residual network module designed to adaptively transform input features while preserving residual connections.
    This module dynamically adjusts the dimensionality of input features and applies transformations to enhance feature representations.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
        hidden_dim (int): Dimensionality of the hidden layer, default is 256.

    Forward Pass:
        The input `x` is passed through a fully connected layer, LayerNorm, activation function, and dropout layer.
        A residual connection is added to combine the transformed features with the original input, ensuring information preservation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Dynamically adjust input feature dimensionality
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Dynamically adjust output feature dimensionality
        self.activation = nn.GELU()  # Activation function
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization
        self.layernorm = nn.LayerNorm(hidden_dim)  # Layer normalization for stability
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()  # Residual connection

    def forward(self, x):
        residual = self.skip(x)  # Preserve the original input
        x = self.fc1(x)  # Transform input features
        x = self.layernorm(x)  # Normalize features
        x = self.activation(x)  # Apply activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Transform to output dimensionality
        return x + residual  # Combine with residual connection


class MultiModalMemoryLearner(nn.Module):
    def __init__(self, memory_length, memory_depth, clip_model):
        super().__init__()

        dtype = clip_model.dtype
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim_common = 512
        embed_dim = embed_dim_text + embed_dim_image

        memory_length_half = memory_length // 3  # Half the length of the memory buffer
        self.memory_depth = memory_depth  # Depth of the memory buffer

        # Initialize memory buffers for text and image modalities
        self.text_memory_complete = nn.Parameter(self.initialize_text_memory(memory_length_half, embed_dim_text, clip_model, dtype=dtype))
        self.text_memory_missing = nn.Parameter(self.initialize_text_memory(memory_length_half, embed_dim_text, clip_model, dtype=dtype))
        self.visual_memory_complete = nn.Parameter(self.initialize_visual_memory(memory_length_half, embed_dim_image, clip_model, dtype=dtype))
        self.visual_memory_missing = nn.Parameter(self.initialize_visual_memory(memory_length_half, embed_dim_image, clip_model, dtype=dtype))

        # Initialize shared memory buffers
        self.common_memory_complete = nn.Parameter(self.initialize_memory(memory_length_half, embed_dim_common, dtype=dtype))
        self.common_memory_image = nn.Parameter(self.initialize_memory(memory_length_half, embed_dim_common, dtype=dtype))
        self.common_memory_text = nn.Parameter(self.initialize_memory(memory_length_half, embed_dim_common, dtype=dtype))

        # Dynamic projection networks
        self.dynamic_common_token_net = ResidualDynamicNet(
            input_dim=embed_dim_text + embed_dim_image,
            output_dim=embed_dim_common,
            hidden_dim=256
        )
        # Dynamic projection networks for text and image modalities
        self.compound_memory_projections_text = nn.ModuleList([
            ResidualDynamicNet(embed_dim, embed_dim_text) for _ in range(self.memory_depth)
        ])
        self.compound_memory_projections_image = nn.ModuleList([
            ResidualDynamicNet(embed_dim, embed_dim_image) for _ in range(self.memory_depth)
        ])

        # Shared projection networks
        self.common_memory_projection_image = ResidualDynamicNet(embed_dim_common, embed_dim_image)
        self.common_memory_projection_text = ResidualDynamicNet(embed_dim_common, embed_dim_text)

        # Layer normalization for each layer
        self.layernorm_text = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.memory_depth)])
        self.layernorm_image = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.memory_depth)])

        # Residual factor for dynamic memory updates
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))  # Controls the balance between new and previous memory

    def initialize_text_memory(self, memory_length_half, embed_dim_text, clip_model, noise_scale=0.2, dtype=None):
        """
        Initialize text memory buffer with noise for variability.
        """
        text_token = clip_model.token_embedding(torch.tensor([49407]))  # Shape: [1, 512]
        text_token = text_token.repeat(memory_length_half, 1)  # Shape: [memory_length_half, 512]
        text_noise = torch.randn(memory_length_half, embed_dim_text)  # Add noise
        text_noise = text_noise / text_noise.norm(dim=-1, keepdim=True)  # Normalize noise
        text_token += noise_scale * text_noise  # Add scaled noise
        return text_token.type(dtype) if dtype is not None else text_token

    def initialize_visual_memory(self, memory_length_half, embed_dim_image, clip_model, noise_scale=0.2, dtype=None):
        """
        Initialize visual memory buffer with noise for variability.
        """
        visual_token = clip_model.visual.class_embedding  # Shape: [1, 768]
        visual_token = visual_token.repeat(memory_length_half, 1)  # Shape: [memory_length_half, 768]
        visual_noise = torch.randn(memory_length_half, embed_dim_image)  # Add noise
        visual_noise = visual_noise / visual_noise.norm(dim=-1, keepdim=True)  # Normalize noise
        visual_token += noise_scale * visual_noise  # Add scaled noise
        return visual_token.type(dtype) if dtype is not None else visual_token

    def initialize_memory(self, memory_length_half, embed_dim, noise_scale=0.2, dtype=None):
        """
        Initialize a generic memory buffer with noise for variability.
        """
        token = torch.randn(memory_length_half, embed_dim)  # Random initialization
        token = token / token.norm(dim=-1, keepdim=True)  # Normalize
        token += noise_scale * torch.randn(memory_length_half, embed_dim)  # Add scaled noise
        return token.type(dtype) if dtype is not None else token

    def generate_layer_memories(self, prev_image_memory, prev_text_memory, index):
        """
        Generate memory buffers for the current layer by combining and transforming previous layer memories.
        """
        # Combine previous image and text memories
        combined_memory = torch.cat([prev_image_memory, prev_text_memory], dim=-1)

        # Update image memory using dynamic projection and residual connection
        current_image_memory = self.compound_memory_projections_image[index](
            self.layernorm_image[index](combined_memory)
        )
        current_image_memory = self.residual_alpha * current_image_memory + (1 - self.residual_alpha) * prev_image_memory

        # Update text memory using dynamic projection and residual connection
        current_text_memory = self.compound_memory_projections_text[index](
            self.layernorm_text[index](combined_memory)
        )
        current_text_memory = self.residual_alpha * current_text_memory + (1 - self.residual_alpha) * prev_text_memory

        return current_image_memory, current_text_memory

    def forward(self, missing_type, visual_features=None, text_features=None):
        """
        Generate common_token and memory buffers based on the input features and missing modality type.
        """
        # Generate common_token
        if visual_features is not None and text_features is not None:
            combined_features = torch.cat([visual_features, text_features],
                                          dim=-1)  # [batch_size, embed_dim_text + embed_dim_image]
            dynamic_common_token = self.dynamic_common_token_net(combined_features)  # [batch_size, embed_dim_common]
            dynamic_common_token = dynamic_common_token.unsqueeze(1).repeat(1, self.common_memory_complete.shape[0],
                                                                            1)  # [batch_size, memory_length_half, embed_dim_common]
            common_token = self.common_memory_complete.unsqueeze(
                0) + dynamic_common_token  # [batch_size, memory_length_half, embed_dim_common]
        else:
            common_token = self.common_memory_complete  # Use default common memory if no visual or text features are provided

        # Initialize memory buffers
        batch_size = len(missing_type)
        memories_buffer_image = [[] for _ in range(self.memory_depth)]  # List to store image memories for each layer
        memories_buffer_text = [[] for _ in range(self.memory_depth)]  # List to store text memories for each layer

        for i in range(batch_size):
            # Initialize memory based on missing modality type
            if missing_type[i] == 0:  # Complete modality
                initial_memory_image = self.visual_memory_complete
                initial_memory_text = self.text_memory_complete
                common_memory = common_token[i] if visual_features is not None else self.common_memory_complete
            elif missing_type[i] == 1:  # Missing text modality
                initial_memory_image = self.visual_memory_complete
                initial_memory_text = self.text_memory_missing
                common_memory = common_token[i] if visual_features is not None else self.common_memory_image
            elif missing_type[i] == 2:  # Missing image modality
                initial_memory_image = self.visual_memory_missing
                initial_memory_text = self.text_memory_complete
                common_memory = common_token[i] if visual_features is not None else self.common_memory_text

            # Generate initial memory for the first layer
            memories_buffer_image[0].append(self.compound_memory_projections_image[0](
                self.layernorm_image[0](torch.cat([initial_memory_image, initial_memory_text], -1))))
            memories_buffer_text[0].append(self.compound_memory_projections_text[0](
                self.layernorm_text[0](torch.cat([initial_memory_image, initial_memory_text], -1))))

            # Generate layer-wise memories for subsequent layers
            for index in range(1, self.memory_depth):
                current_image_memory, current_text_memory = self.generate_layer_memories(
                    memories_buffer_image[index - 1][-1], memories_buffer_text[index - 1][-1], index
                )
                memories_buffer_image[index].append(current_image_memory)
                memories_buffer_text[index].append(current_text_memory)

            # Append common memory to the first layer
            memories_buffer_image[0][i] = torch.cat([
                memories_buffer_image[0][i],
                self.common_memory_projection_image(common_memory)]
                , 0)
            memories_buffer_text[0][i] = torch.cat([
                memories_buffer_text[0][i],
                self.common_memory_projection_text(common_memory)]
                , 0)

        # Stack memories for each layer
        memories_buffer_image = [torch.stack(memories) for memories in memories_buffer_image]
        memories_buffer_text = [torch.stack(memories) for memories in memories_buffer_text]

        return memories_buffer_image, memories_buffer_text
class CustomCLIP(nn.Module):
    def __init__(self, memory_length, memory_depth, clip_model, config):
        super().__init__()
        self.memory_learner = MultiModalMemoryLearner(memory_length, memory_depth, clip_model)  # Memory learning module
        self.image_encoder = clip_model.visual  # Image encoder
        self.text_encoder = TextEncoder(clip_model)  # Text encoder
        self.logit_scale = clip_model.logit_scale  # Temperature parameter
        self.dtype = clip_model.dtype  # Data type
        self.config = config

        #
        if self.config["loss_names"]["food101"] > 0 or self.config["loss_names"]["mmimdb"] > 0:

            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.logit_scale.requires_grad = False

        if self.config["loss_names"]["hatememes"] > 0:
            for param in self.image_encoder.parameters():
                param.requires_grad = True
            for param in self.text_encoder.parameters():
                param.requires_grad = True
            self.logit_scale.requires_grad = True

    def forward(self, image, text, missing_type):
        """
        Forward pass that combines input features with dynamic memory buffers.
        """
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(
            image.get_device()).squeeze(1)  # Tokenize text
        all_memories_image, all_memories_text = self.memory_learner(missing_type)  # Generate memory buffers
        text_features = self.text_encoder(tokenized_texts, all_memories_text, missing_type)  # Encode text
        image_features = self.image_encoder(image.type(self.dtype), all_memories_image, missing_type)  # Encode image
        return torch.cat([image_features, text_features], -1)  # Concatenate features


class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        clip_model = load_clip_to_cpu(config['vit'], config['memory_length'], config['memory_depth'])

        print("Building custom CLIP")
        hidden_size = 512 * 2
        self.model = CustomCLIP(config['memory_length'], config['memory_depth'], clip_model, config)

        # ===================== Downstream ===================== #
        if (
                self.hparams.config["load_path"] != ""
                and not self.hparams.config["test_only"]
                and not self.hparams.config["finetune_first"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Linear(hidden_size, cls_num)
            self.hatememes_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Linear(hidden_size, cls_num)
            self.food101_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights)

        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print("use pre-finetune model")

        if not self.hparams.config["test_only"]:
            for name, param in self.model.named_parameters():
                if "memory_learner" not in name and "memory" not in name and 'ln_final' not in name and 'ln_post' not in name and \
                        name.split('.')[-1] != 'proj':
                    param.requires_grad_(False)

        clip_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        self.records = {}

    def infer(
            self,
            batch,
    ):
        text = batch["text"]
        img = batch["image"][0]  # extract the first view (total 1)
        if self.hparams.config["test_only"]:
            self.model.eval()
            if self.hparams.config["loss_names"]["hatememes"] > 0:
                self.hatememes_classifier.eval()

            if self.hparams.config["loss_names"]["food101"] > 0:
                self.food101_classifier.eval()

            if self.hparams.config["loss_names"]["mmimdb"] > 0:
                self.mmimdb_classifier.eval()
        both_feats = self.model(img, text, batch["missing_type"])
        feature_dim = both_feats.shape[1] // 2
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                pass
            elif batch["missing_type"][idx] == 1:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 2:
                both_feats[idx, :feature_dim].zero_()

        ret = {
            "cls_feats": both_feats,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))

        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    #         print('missing_img:', self.missing_img_memory[0,0:3,0:8])
    #         print('missing_text:', self.missing_text_memory[0,0:3,0:8])
    #         print('complete:', self.complete_memory[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        clip_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return clip_utils.set_schedule(self)
