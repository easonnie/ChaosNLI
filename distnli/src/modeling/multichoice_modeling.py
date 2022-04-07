import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from modeling.soft_crossentropy import softCrossEntropy


class MultichoiceModel(nn.Module):
    def __init__(self, encoder, number_of_choice, with_token_type=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultichoiceModel, self).__init__()
        # encoder pass model
        self.encoder = encoder  # assert self.encoder.num_labels == 1
        self.number_of_choice = number_of_choice
        self.with_token_type = with_token_type

    def forward(self, input_item, labels=None, soft_loss=False):

        # assert len(input_list) == self.number_of_choice

        out_list = []
        for i in range(self.number_of_choice):
            if self.with_token_type:
                current_out = self.encoder(
                    input_ids=input_item[f'input_ids_{i}'],
                    attention_mask=input_item[f'attention_mask_{i}'],
                    token_type_ids=input_item[f'token_type_ids_{i}'],
                )
                current_logits = current_out[0]     # B, logits
                out_list.append(current_logits.squeeze(1))      # [B, B, ... number of choices]
            else:
                raise NotImplementedError

        combined_logits = torch.stack(out_list, dim=1)  # think about it, it might not be the shape we want it to be

        outputs = (combined_logits,)

        loss = None
        if labels is not None:
            if soft_loss == False:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(combined_logits.view(-1, self.number_of_choice), labels.view(-1))  # hard code 2 here.
            else:
                loss_fct = softCrossEntropy
                loss = loss_fct(combined_logits.view(-1, self.number_of_choice), labels.view(-1, self.number_of_choice))

        if loss is not None:
            outputs = (loss,) + outputs

        return outputs



