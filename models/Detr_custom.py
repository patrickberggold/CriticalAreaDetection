import torch
from transformers import DetrForObjectDetection, DetrConfig
from transformers.models.detr.configuration_detr import DetrConfig
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.detr.modeling_detr import BaseModelOutput, DetrObjectDetectionOutput, DETR_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, DetrModel, DetrHungarianMatcher, DetrLoss, DetrModelOutput, DetrAttention
import math

class Detr_custom(DetrForObjectDetection):
    def __init__(self, config: dict = None):

        if config is None:
            config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        
        super().__init__(config)
        self.overwrite_functions()

    def overwrite_functions(self):
        self.forward = self.forward
        self.model.forward = self.model_forward
        
        for enc_layer in self.model.encoder.layers:
            self_attn: DetrAttention = enc_layer._modules['self_attn']
            self_attn.with_pos_embed = self.with_pos_embed_overwritten
        for dec_layer in self.model.decoder.layers:
            self_attn: DetrAttention = dec_layer._modules['self_attn']
            self_attn.with_pos_embed = self.with_pos_embed_overwritten
            enc_attn: DetrAttention = dec_layer._modules['encoder_attn']
            enc_attn.with_pos_embed = self.with_pos_embed_overwritten


    def update_model(self, custom_config: dict):
        
        # unload pretrained and initialize weights
        if not custom_config['facebook_pretrained']:
            print('Initializing random weights...')
            self.model = DetrModel(self.config)
            self.overwrite_functions()

        self.additional_queries = custom_config['additional_queries']
        
        assert self.additional_queries in ['vanilla', 'vanilla_imgAugm', 'before_encoder', 'after_encoder', 'before_encoder+', 'after_encoder+']
        
        # adjust number of classes
        self.class_labels_classifier = torch.nn.Linear(self.model.config.d_model, custom_config['num_classes'] + 1) # detection + no_detection
        self.model.config.num_labels = custom_config['num_classes']
        torch.nn.init.normal_(self.class_labels_classifier.weight, mean=0.0, std=self.config.init_std)
        torch.nn.init.constant_(self.class_labels_classifier.bias, 0.0)
        self.embed_dim = 256
        
        # if not the original DETR implementation...
        if self.additional_queries not in ['vanilla', 'vanilla_imgAugm']:
            self.num_extra_queries = 100
            if not self.additional_queries.endswith('+'):
                self.central_ascent_embeddings, self.sides_ascent_embeddings = None, None
                self.simulation_embeddings = torch.nn.Embedding(3, self.embed_dim)
            else:
                self.simulation_embeddings = torch.nn.Embedding(3, self.embed_dim) 
                self.central_ascent_embeddings = torch.nn.Embedding(7, self.embed_dim) # additional 7 options (E=1/2/3 + S=0/2.4 + None) both for vertical ascent in the center and from the sides
                self.sides_ascent_embeddings = torch.nn.Embedding(7, self.embed_dim)
                self.obstacle_presence_embeddings = torch.nn.Embedding(2, self.embed_dim)
                torch.nn.init.normal_(self.central_ascent_embeddings.weight.data, mean=0.0, std=self.config.init_std)
                torch.nn.init.normal_(self.sides_ascent_embeddings.weight.data, mean=0.0, std=self.config.init_std)
                torch.nn.init.normal_(self.obstacle_presence_embeddings.weight.data, mean=0.0, std=self.config.init_std)
            torch.nn.init.normal_(self.simulation_embeddings.weight.data, mean=0.0, std=self.config.init_std)
            self.model.backbone.position_embedding = DetrSinePositionEmbedding_custom(self.config.d_model // 2, normalize=True, num_extra_queries=self.num_extra_queries)
        else:
            self.num_extra_queries = None

        # for tuning...
        k_decoder_queries = custom_config['top_k']
        if k_decoder_queries != self.model.query_position_embeddings.num_embeddings:
            self.model.query_position_embeddings = torch.nn.Embedding(k_decoder_queries, self.model.config.d_model)
            torch.nn.init.normal_(self.model.query_position_embeddings.weight.data, mean=0.0, std=self.config.init_std)

    
    def with_pos_embed_overwritten(self, tensor: torch.Tensor, position_embeddings: torch.Tensor):
        if position_embeddings is None:
            return tensor
        elif tensor.size() == position_embeddings.size():
            return tensor + position_embeddings
        elif tensor.shape[1] < position_embeddings.shape[1]:
            return tensor + position_embeddings[:, :tensor.size(1), :]
        else:
            raise NotImplementedError

    
    #################################################################################
    # MODEL FORWARD #################################################################
    #################################################################################
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def model_forward(
        self,
        pixel_values,
        simulation_embeddings=None, # added to original code
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DetrFeatureExtractor, DetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")

        >>> # prepare image for the model
        >>> inputs = feature_extractor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 100, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, position_embeddings_list = self.model.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.model.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        
        # encoding extension must be considered here
        if position_embeddings_list[-1].ndim == 3:
            # here, flatten() has already been performed
            position_embeddings = position_embeddings_list[-1].permute(0, 2, 1)
        elif position_embeddings_list[-1].ndim == 4:
            # default setting
            position_embeddings = position_embeddings_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # append extra queries before the encoder
        if self.additional_queries == 'before_encoder':
            flattened_features = torch.cat((flattened_features, simulation_embeddings), dim=1)
            flattened_mask = torch.cat((flattened_mask, torch.ones(simulation_embeddings.size(0), simulation_embeddings.size(1), dtype=torch.bool, device=flattened_mask.device)), dim=1)

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # append extra queries after the encoder
        if self.additional_queries == 'after_encoder':
            queries = torch.zeros_like(query_position_embeddings)
            assert (encoder_outputs[0].requires_grad == simulation_embeddings.requires_grad)
            encoder_outputs['last_hidden_state'] = torch.cat((encoder_outputs[0], simulation_embeddings), dim=1)
            flattened_mask = torch.cat((flattened_mask, torch.ones(simulation_embeddings.size(0), simulation_embeddings.size(1), dtype=torch.bool, device=flattened_mask.device)), dim=1)
        else:
            queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )

    
    
    #################################################################################
    # DETR_OD FORWARD ###############################################################
    #################################################################################
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values,
        information_ids, # added to original code
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import DetrFeatureExtractor, DetrForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to COCO API
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     # let's only keep detections with score > 0.9
        ...     if score > 0.9:
        ...         print(
        ...             f"Detected {model.config.id2label[label.item()]} with confidence "
        ...             f"{round(score.item(), 3)} at location {box}."
        ...         )
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.additional_queries in ['vanilla_imgAugm', 'vanilla']:
            # vanilla and imgAugm
            simulation_embeddings = None
        else:
            agent_ids, central_ids, sides_ids, obstacle_ids = torch.split(information_ids, 1, dim=1)
            # before_encoder and after_encoder
            if self.central_ascent_embeddings is None and self.sides_ascent_embeddings is None:
                simulation_embeddings = self.simulation_embeddings(agent_ids).repeat(1, self.num_extra_queries, 1)
            # before_encoder+ and after_encoder+
            elif self.central_ascent_embeddings is not None and self.sides_ascent_embeddings is not None:
                agent_embeddings = self.simulation_embeddings(agent_ids).repeat(1, self.num_extra_queries//4, 1)
                central_ascent_embeddings = self.central_ascent_embeddings(central_ids).repeat(1, self.num_extra_queries//4, 1)
                sides_ascent_embeddings = self.sides_ascent_embeddings(sides_ids).repeat(1, self.num_extra_queries//4, 1)
                obstacle_presence_embeddings = self.obstacle_presence_embeddings(obstacle_ids).repeat(1, self.num_extra_queries//4, 1)
                simulation_embeddings = torch.cat((agent_embeddings, central_ascent_embeddings, sides_ascent_embeddings, obstacle_presence_embeddings), dim=1)
            assert simulation_embeddings.shape == (information_ids.size(0), self.num_extra_queries, self.embed_dim), f'Wrong dimensions given, got ({simulation_embeddings.shape}() but should be ({information_ids.size(0), self.num_extra_queries, self.embed_dim})'

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            simulation_embeddings,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
           
            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return DetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    
class DetrSinePositionEmbedding_custom(torch.nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None, num_extra_queries=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.num_extra_queries = num_extra_queries
        assert self.num_extra_queries % 2 == 0, 'self.num_extra_queries must be divisible by 2 (sin-cos positional encodings)'

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)

        normalizer_pix_x = x_embed[:, :, -1:][0,0,0]
        normalizer_pix_y = y_embed[:, -1:, :][0,0,0]
        normalizer_pix = max(normalizer_pix_x, normalizer_pix_y)

        normalizer = normalizer_pix + self.num_extra_queries

        extend = torch.arange(normalizer_pix+1, normalizer+1, dtype=torch.float32, device=pixel_values.device)
        extend = torch.tile(extend, (pixel_values.shape[0], 1))

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

            extend = extend / normalizer_pix * self.scale # or normalizer

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        extend = extend[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        extend = torch.stack((extend[:, :, 0::2].sin(), extend[:, :, 1::2].cos()), dim=3).flatten(2)
        extend = torch.cat((extend, extend), dim=2).permute(0, 2, 1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2)
        pos = torch.cat((pos, extend), dim=2)
        return pos

