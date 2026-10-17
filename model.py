import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
from transformers import BertConfig, BertModel


class BERTResNetClassifier(nn.Module):
    def __init__(self, num_classes=6, load_pretrained_weights=True):

        super(BERTResNetClassifier, self).__init__()

        self.num_classes = num_classes

        # Image processing (ResNet)
        image_weights = ResNet50_Weights.IMAGENET1K_V1 if load_pretrained_weights else None
        self.image_model = resnet50(weights=image_weights)

        # Image processing (Fully Connected Layer)
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)

        # Dropout layer
        self.drop = nn.Dropout(p=0.3)

        # Text processing (using the 768-dimensional BERT arrays)
        if load_pretrained_weights:
            self.text_model = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.text_model = BertModel(BertConfig())

        # Text processing (Fully Connected Layer)
        self.fc_text = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=num_classes, bias=True)

    def forward(self, image, text_input_ids, text_attention_mask):
        # Image branch
        x_img = self.image_model(image)
        x_img = self.drop(x_img)
        x_img = self.fc_image(x_img)

        # Text branch
        x_text_last_hidden_states = self.text_model(
            input_ids = text_input_ids,
            attention_mask = text_attention_mask,
            return_dict=False
        )
        x_text_pooled_output = x_text_last_hidden_states[0][:, 0, :]
        x_text = self.drop(x_text_pooled_output)
        x_text = self.fc_text(x_text)

        # Fusion and max merge
        x = torch.max(x_text, x_img)

        # Classification (softmax omitted; CrossEntropyLoss applies it internally during training)
        return x
